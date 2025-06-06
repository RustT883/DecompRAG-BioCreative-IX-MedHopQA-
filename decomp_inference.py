from langchain_ollama import ChatOllama
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from custom_langchain.retrievers import BM25SRetriever
from langchain.retrievers import EnsembleRetriever
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import Dict, Any
import gc
import warnings
from tqdm import tqdm
import torch
import pandas as pd
import csv
import os
from time import sleep

# ==== CONFIG ====
device = torch.device("cuda")
warnings.filterwarnings("ignore")
callbacks = CallbackManager([StreamingStdOutCallbackHandler()])
handler = StdOutCallbackHandler()

# CSV output configuration
CHECKPOINT_INTERVAL = 2
RETRY_DELAY = 5 
MAX_RETRIES = 3
OUTPUT_CSV_PATH = "./qa_results.csv"

# ==== DIRECTORIES ====
core_embeddings_model = HuggingFaceEmbeddings(
    model_name="abhinand/MedEmbed-small-v0.1",
    model_kwargs={
        'device': "cpu", 
        'trust_remote_code': True,
    },
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 8
    }
)

persist_directory = "./medrag_chroma_2"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=core_embeddings_model)

llm = ChatOllama(
    base_url = "http://localhost:11434",
    model="thewindmom/llama3-med42-8b", 
    timeout=300, 
    temperature = 0.0,
    disable_streaming = True,
    num_ctx=8192
)

# ==== PROMPT TEMPLATES ====
prompt_template = """Context information is below.
---------------------
{context}
---------------------
You are an expert in medicine, molecular biology and biochemistry. Answer the question below based strictly on the context above and using common sense. 
If the answer cannot be found in the context, say "I couldn't find a definitive answer in my sources."
For complex questions, break them down into logical sub-questions.
Query: {question}
Answer: """

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

DECOMPOSITION_PROMPT = ChatPromptTemplate.from_template("""
Break down this medical question into simple, factual sub-questions that can be answered independently from medical literature and be used to answer the main question.
Each sub-question should:
1. Be answerable with a specific fact or short answer
2. Build logically toward answering the main question
3. Use clear medical terminology. Don't turn "diagnostic procedure" into "test"
4. Duplicate the initial question in the numbered list of sub-questions
5. Don't make the questions redundant
6. Always make no more than 4 subquestions
7. NEVER loose context (e.g. never make "Which chromosome does this disorder primarily affect?" for "Which gene is associated with a rare genetic disorder characterized by bilateral congenital hearing loss and brain malformations?", since you will loose "brain malformations" with "bilateral congenital hearing loss")
8. NEVER make the sub-quesitons more complex than the initial question

Output ONLY the sub-questions as a numbered list, nothing else.

Question: {question}
Sub-questions:
""")

COMPOSITION_PROMPT = ChatPromptTemplate.from_template("""
Combine these answers to sub-questions into a coherent final answer.
Be precise and cite sources when available using the provided source references.

Sub-question Answers:
{intermediate_answers}

Source References:
{source_references}

Original question: {main_question}

ALWAYS Extract a concise SINGLE and SHORT final answer (1-3 word long) following these rules:
1. Use complete formal names (e.g., "Diabetes Mellitus, type 2")
2. For chromosome questions: format as "Chromosome X"
3. For syndromes named after people: use full name (e.g., "Carpenter's syndrome")
4. For yes/no: answer only "Yes" or "No"
5. For true/false: answer only "TRUE" or "FALSE"
6. Pick a single answer without writing synonyms (e.g., either write "PID" or "Pelvic Inflammatory Disease", NOT "PID (Pelvic Inflammatory Disease)")
7. Don't overthink or overengineer the answers (e.g., write "Cardiology", NOT "Adult Congenital Cardiology")
8. Make sure the answers are not recursive (e.g. for question "What is the primary cause of the physical changes observed in males with Klinefelter syndrome during puberty?" the answer should be not "Klinefelter Syndrome", but the cause of it, like "X Chromosome")
9. For drug questions: don't mention drug form (e.g., "Calamine" but NOT "Calamine lotion")
11. Answers like "Indirectly" or "Probably" are forbidden.
10. If you don't know the answer, just write "N/A" and nothing else
11. For protein questions: just give a protein name (e.g., "Myelin", but not "Myelin protein")

Structure your final answer as follows:

SHORT ANSWER: [your 1-3 word answer here]

DETAILED ANSWER:
[your detailed explanation here, citing sources like [1], [2] where appropriate]

SOURCE REFERENCES:
{source_references}
""")

# ==== HELPER FUNCTIONS ====
def format_source_references(sources: list[dict]) -> str:
    """Format sources with numbering for citation in the answer"""
    source_refs = []
    for idx, source in enumerate(sources, 1):
        if 'wikipedia.org' in source['source_url']:
            source_refs.append(
                f"[{idx}] {source['source_title']} - {source['source_url']}"
            )
    return "\n".join(source_refs) if source_refs else ""

def improve_retrieval(query: str, is_subquestion: bool = False, main_question: str = "") -> str:
    """Optimize the query for better retrieval while avoiding circular references"""
    if is_subquestion and main_question:
        optimization_prompt = ChatPromptTemplate.from_template("""
You are a medical question optimizer. Optimize this specific medical sub-question for document retrieval but never answer them (e.g. "Which medical specialty is likely involved in the diagnosis and treatment of angiolipomas?" is not "Dermatology involvement in angiolipoma diagnosis and management" but "Angliolipoma medical specialty")
Focus only on the specific aspect asked about. Don't incorporate the main question.
NEVER lose context (e.g. "Which medical subspecialty primarily focuses on the diagnosis and management of skin lesions?" is not "Dermatology" but "Dermatology skin lesions medical specialty")

Don't explain your reasoning.

Sub-question: {query}
Optimized:""")
    else:
        optimization_prompt = ChatPromptTemplate.from_template("""
Optimize this medical question for better document retrieval. 
Don't explain your reasoning.

Original: {query}
Optimized:""")
    
    optimizer = optimization_prompt | llm | StrOutputParser()
    return optimizer.invoke({"query": query})

def format_document(doc: Document) -> str:
    """Custom document formatter that safely handles metadata"""
    base_content = f"Content: {doc.page_content}\n"
    
    if not hasattr(doc, 'metadata') or not doc.metadata:
        return base_content
    
    metadata = doc.metadata
    title = metadata.get('title', 'N/A')
    url = metadata.get('url', 'N/A')
    
    return f"Content: {doc.page_content[:500]}\nSource: {doc.metadata.get('title','')}\n"

def extract_short_answer(long_answer: str) -> str:
    """Extract the short answer from the long answer"""
    # Look for the SHORT ANSWER: pattern
    if "SHORT ANSWER:" in long_answer:
        return long_answer.split("SHORT ANSWER:")[1].split("\n")[0].strip()
    # Fallback to first line that meets criteria
    for line in long_answer.split('\n'):
        line = line.strip()
        if line and not line.startswith('[') and len(line.split()) <= 3:
            return line
    return "N/A"

def write_to_csv(qidx: str, question: str, short_answer: str, long_answer: str, file_path: str):
    """Write or append results to CSV file"""
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['QIDX', 'Question', 'Short Answer', 'Long Answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'QIDX': qidx,
            'Question': question,
            'Short Answer': short_answer,
            'Long Answer': long_answer
        })

# ==== CHAIN SETUP ====
class CustomStuffDocumentsChain(StuffDocumentsChain):
    def _get_inputs(self, docs, **kwargs):
        doc_strings = [format_document(doc) for doc in docs]
        return {**{self.document_variable_name: "\n\n".join(doc_strings)}, **kwargs}

llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, callbacks=None, verbose=False)

vectorstore_retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={
    	"k": 4
    }
)

combine_documents_chain = CustomStuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    callbacks=callbacks,
)

keyword_retriever = BM25SRetriever.from_persisted_directory("bm25_medrag", k=4)

ensemble_retriever = EnsembleRetriever(
	retrievers=[vectorstore_retriever, keyword_retriever],
	weights=[0.6, 0.4]
)

qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    retriever=ensemble_retriever,
    verbose=False,
    return_source_documents=True
)

# ==== MAIN PROCESSING FUNCTIONS ====
def decompose_question(question: str) -> List[str]:
    """Decomposes a complex question into sub-questions while keeping the original question"""
    decomposition_chain = DECOMPOSITION_PROMPT | llm | StrOutputParser()
    result = decomposition_chain.invoke({"question": question})
    
    sub_questions = []
    for line in result.split('\n'):
        line = line.strip()
        if line and line[0].isdigit() and '. ' in line:
            sub_questions.append(line.split('. ', 1)[1])
    
    if question not in sub_questions:
        sub_questions.insert(0, question)
    
    return sub_questions

def format_for_json(docs: list[Document]) -> list[dict]:
    formatted = []
    for doc in docs:
        metadata = getattr(doc, 'metadata', {})
        formatted.append({
            "content": doc.page_content,
            "source_title": metadata.get('title', 'N/A'),
            "source_url": metadata.get('url', 'N/A')
        })
    return formatted

def process_question(question: str, qidx: str = "0"):
    try:
        # Parallelize sub-question processing
        from concurrent.futures import ThreadPoolExecutor
        
        sub_questions = decompose_question(question)
        print(f"\nDecomposed into sub-questions:")
        for i, q in enumerate(sub_questions, 1):
            print(f"{i}. {q}")
            
        intermediate_answers = {}
        all_sources = []
        source_map = {} 
        
        # Process questions in parallel
        def process_subquestion(q, i):
            is_subquestion = i > 0
            optimized_query = improve_retrieval(q, is_subquestion, sub_questions[0])
            print(f"Optimized query: {optimized_query}")
            
            qa_result = qa.invoke({"query": optimized_query[:512]})
            return {
                "idx": i,
                "question": q,
                "optimized_query": optimized_query,
                "answer": qa_result['result'],
                "sources": format_for_json(qa_result['source_documents'][:2])  # Reduced from 3
            }
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_subquestion, q, i) 
                      for i, q in enumerate(sub_questions)]
            for future in futures:
                result = future.result()
                i = result["idx"]
                intermediate_answers[f"q{i}"] = {
                    "question": result["question"],
                    "optimized_query": result["optimized_query"],
                    "answer": result["answer"],
                    "sources": result["sources"]
                }
                source_map[f"q{i}"] = result["sources"]
                all_sources.extend(result["sources"])
        
        # Prepare source references
        source_references = format_source_references(all_sources[:10])
        
        # Generate final answer
        composition_chain = COMPOSITION_PROMPT | llm | StrOutputParser()
        final_answer = composition_chain.invoke({
            "intermediate_answers": "\n".join(
                f"Q{i}: {v['question']}\nA: {v['answer']}" 
                for i, v in enumerate(intermediate_answers.values(), 1)
            ),
            "source_references": source_references,
            "main_question": question
        })
        
        # Ensure sources are included even if LLM didn't add them
        if "SOURCE REFERENCES:" not in final_answer:
            final_answer += f"\n\nSOURCE REFERENCES:\n{source_references}"
        
        # Extract short answer
        short_answer = extract_short_answer(final_answer)
        
        # Write to CSV
        write_to_csv(qidx, question, short_answer, final_answer, OUTPUT_CSV_PATH)
        
        return {
            "Short_Answer": short_answer,
            "Long_Answer": final_answer,
            "Sources": all_sources[:10],
            "Intermediate_steps": intermediate_answers
        }
            
    except Exception as e:
        print(f"Error during QA processing: {str(e)}")
        write_to_csv(qidx, question, f"ERROR: {str(e)}", "", OUTPUT_CSV_PATH)
        return {
            "Short_Answer": f"ERROR: {str(e)}",
            "Long_Answer": "",
            "Sources": [],
            "Intermediate_steps": {}
        }

# ==== INTERFACE FUNCTIONS ====
# ==== INTERFACE FUNCTIONS ====
def process_question_list(input_csv: str):
    """Process questions sequentially in batches from a CSV file"""
    try:
        df = pd.read_csv(input_csv)
        if len(df) == 0:
            print("Input CSV is empty")
            return
    except Exception as e:
        print(f"Error reading input CSV: {str(e)}")
        return

    # Initialize processed questions tracking
    processed_qidx = set()
    if os.path.exists(OUTPUT_CSV_PATH):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV_PATH, usecols=['QIDX'])
            processed_qidx = set(existing_df['QIDX'].astype(str).unique())
            print(f"Resuming processing with {len(processed_qidx)} already completed questions")
        except Exception as e:
            print(f"Warning: Could not read existing output file - {str(e)}")

    # Create list of unprocessed questions
    unprocessed = []
    for _, row in df.iterrows():
        qidx = str(row['QIDX']) if 'QIDX' in row else str(row.name)
        if qidx not in processed_qidx:
            unprocessed.append((qidx, row['Question']))

    total_to_process = len(unprocessed)
    if total_to_process == 0:
        print("No new questions to process")
        return

    print(f"Starting sequential processing of {total_to_process} questions")
    
    # Process in batches (but sequentially within each batch)
    batch_size = 4
    with tqdm(total=total_to_process, desc="Processing questions") as pbar:
        for batch_start in range(0, total_to_process, batch_size):
            batch = unprocessed[batch_start:batch_start + batch_size]
            batch_results = []
            
            for qidx, question in batch:
                try:
                    result = process_question(question, qidx)
                    batch_results.append({
                        'QIDX': qidx,
                        'Question': question,
                        'Short Answer': result.get("Short_Answer", "N/A"),
                        'Long Answer': result.get("Long_Answer", "")
                    })
                    pbar.update(1)
                except Exception as e:
                    print(f"\nError processing question {qidx}: {str(e)}")
                    batch_results.append({
                        'QIDX': qidx,
                        'Question': question,
                        'Short Answer': f"ERROR: {str(e)}",
                        'Long Answer': ""
                    })
                    pbar.update(1)

            # Save batch results
            if batch_results:
                pd.DataFrame(batch_results).to_csv(
                    OUTPUT_CSV_PATH,
                    mode='a',
                    header=not os.path.exists(OUTPUT_CSV_PATH),
                    index=False
                )
            
            # Memory management
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nProcessing complete. Results saved to {OUTPUT_CSV_PATH}")

def interactive_mode():
    """Run in interactive question-answer mode"""
    qidx = 0
    while True:
        try:
            question = input("\nEnter your medical question (or 'exit' to quit): ").strip()
            if question.lower() in ['exit', 'quit']:
                print("Ending session. Goodbye!")
                break
            if not question:
                continue
                
            qidx += 1
            print("\nProcessing your question...")
            result = process_question(question, str(qidx))
            
            print("\n" + "="*80)
            print("FINAL ANSWER:")
            print("-"*80)
            print(result["Long_Answer"])
            
            print("\n" + "="*80)
            print("INTERMEDIATE STEPS:")
            print("-"*80)
            for step in result.get("Intermediate_steps", {}).values():
                print(f"\nQ: {step['question']}")
                print(f"A: {step['answer']}")
            
            print("\n" + "="*80)
            print("SOURCE ATTRIBUTION:")
            print("-"*80)
            for i, source in enumerate(result.get("Sources", [])[:5], 1):
                print(f"\n[{i}] {source.get('source_title', 'N/A')}")
                print(f"URL: {source.get('source_url', 'N/A')}")
                print(f"Content: {source.get('content', '')[:200]}...")
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\nERROR: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical QA System with Source Attribution")
    parser.add_argument('--input', type=str, help='Input CSV file with questions (columns: QIDX,Question)')
    parser.add_argument('--output', type=str, help='Output CSV file path', default=OUTPUT_CSV_PATH)
    args = parser.parse_args()
    
    if args.output:
        OUTPUT_CSV_PATH = args.output
    
    if args.input:
        process_question_list(args.input)
    else:
        interactive_mode()
