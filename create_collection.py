from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tqdm import tqdm
import torch
import os
import gc
from uuid import uuid4
from datetime import datetime
import time
import psutil
from typing import List

# ===== Configuration =====
CHUNK_SIZE = 512
CHUNK_OVERLAP = 32
EMBEDDING_BATCH_SIZE = 64
MAX_ARTICLE_LENGTH = 50000
MAX_CHUNKS_PER_ARTICLE = 20
CHECKPOINT_INTERVAL = 50
LOG_FILE = "processing_log.txt"
PERSIST_DIRECTORY = "/home/rust/Downloads/IBMC/NLP/BioCreative_IX/medrag_chroma_2"
EMBEDDING_MODEL_PATH = "abhinand/MedEmbed-small-v0.1"
CHECKPOINT_FILE = "processing_checkpoint.txt"  # File to store the last processed line
CHUNKED_INSERT_SIZE = 1000  # Number of documents to insert at once

# ===== Memory Monitoring =====
def memory_safe():
    """Check if we have sufficient memory to continue"""
    mem = psutil.virtual_memory()
    if mem.available < 1 * 1024**3:  # Less than 1GB available
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        time.sleep(5)
        mem = psutil.virtual_memory()
        return mem.available >= 1.5 * 1024**3
    return True

# ===== GPU Configuration =====
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# ===== Logging and Checkpointing =====
def log(message):
    with open(LOG_FILE, 'a') as f:
        f.write(f"{time.ctime()}: {message}\n")
    print(message)

def save_checkpoint(line_number):
    """Save the current line number to resume from later"""
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(line_number))

def load_checkpoint():
    """Load the last processed line number, returns 0 if no checkpoint exists"""
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0

# ===== ChromaDB Setup =====
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={
            "batch_size": EMBEDDING_BATCH_SIZE,
            "normalize_embeddings": True
        }
    )

def split_list(splits, chunk_size):
    """Split a list into smaller chunks for memory-safe processing"""
    for i in range(0, len(splits), chunk_size):
        yield splits[i:i + chunk_size]

# ===== Text Processing =====
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def process_article(article: dict) -> List[Document]:
    """Process a single article and return Document objects"""
    if not memory_safe():
        raise MemoryError("Insufficient memory before processing article")
        
    try:
        text = article.get("text", "")[:MAX_ARTICLE_LENGTH]
        if not text.strip():
            return []
            
        chunks = text_splitter.split_text(text)[:MAX_CHUNKS_PER_ARTICLE]
        if not chunks:
            return []
            
        # Prepare documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            categories = article.get("categories", [])
            categories_str = "|".join(str(cat) for cat in categories)[:1000]
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "title": article.get("title", "Untitled")[:200],
                    "url": article.get("url", ""),
                    "categories": categories_str
                }
            ))
            
        return documents
            
    except Exception as e:
        log(f"Article processing error: {str(e)}")
        return []

def process_file(file_path: str):
    """Main processing loop with robust error handling and checkpointing"""
    embeddings = get_embeddings()
    total_chunks = 0
    processed_articles = 0
    accumulated_docs = []
    
    # Load the last checkpoint
    start_line = load_checkpoint()
    if start_line > 0:
        log(f"Resuming from line {start_line}")
    
    try:
        with open(file_path, 'r') as f:
            # Skip already processed lines
            for _ in range(start_line):
                next(f)
            
            # Create a progress bar starting from the checkpoint
            pbar = tqdm(total=os.path.getsize(file_path), desc="Processing articles", unit='B', unit_scale=True)
            pbar.update(f.tell())  # Set initial position
            
            for line in f:
                try:
                    if not memory_safe():
                        raise MemoryError("Insufficient system memory")
                        
                    article = json.loads(line)
                    documents = process_article(article)
                    if documents:
                        accumulated_docs.extend(documents)
                        total_chunks += len(documents)
                    processed_articles += 1
                    
                    # Update checkpoint every line (for maximum safety)
                    current_line = start_line + processed_articles
                    save_checkpoint(current_line)
                    
                    # Update progress bar
                    pbar.update(len(line.encode('utf-8')))
                    
                    # Process in chunks to avoid memory issues
                    if len(accumulated_docs) >= CHUNKED_INSERT_SIZE:
                        for docs_chunk in split_list(accumulated_docs, CHUNKED_INSERT_SIZE):
                            _ = Chroma.from_documents(
                                documents=docs_chunk,
                                embedding=embeddings,
                                persist_directory=PERSIST_DIRECTORY
                            )
                            torch.cuda.empty_cache()
                            gc.collect()
                        accumulated_docs = []
                    
                    # Periodic cleanup and logging
                    if processed_articles % CHECKPOINT_INTERVAL == 0:
                        log(f"Checkpoint: Processed {current_line} lines, {total_chunks} chunks total")
                        gc.collect()
                        if device == "cuda":
                            torch.cuda.empty_cache()
                            
                except json.JSONDecodeError:
                    log(f"JSON decode error at line {current_line}")
                    continue
                except MemoryError as e:
                    log(f"Memory error at line {current_line}: {str(e)}")
                    time.sleep(30)  # Longer sleep for memory issues
                    # Rewind the file pointer to retry the same line
                    f.seek(pbar.n - len(line.encode('utf-8')))
                    continue
                except Exception as e:
                    log(f"Error at line {current_line}: {str(e)}")
                    continue
                    
        # Process any remaining documents
        if accumulated_docs:
            for docs_chunk in split_list(accumulated_docs, CHUNKED_INSERT_SIZE):
                _ = Chroma.from_documents(
                    documents=docs_chunk,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIRECTORY
                )
                torch.cuda.empty_cache()
                gc.collect()
                    
    except Exception as e:
        log(f"Fatal error: {str(e)}")
        raise  # Re-raise to exit the program
    finally:
        pbar.close()
        log(f"Processing completed up to line {start_line + processed_articles} with {total_chunks} chunks")

if __name__ == "__main__":
    log(f"Starting processing on {device}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    
    start_time = time.time()
    
    try:
        process_file("medical_articles.jsonl")
        # If we complete successfully, remove the checkpoint file
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        log("\nProcessing completed successfully!")
    except Exception as e:
        log(f"\nProcessing interrupted due to: {str(e)}")
        log("The program can be restarted to resume from the last checkpoint")
    
    log(f"Time elapsed: {(time.time() - start_time)/60:.2f} minutes")
