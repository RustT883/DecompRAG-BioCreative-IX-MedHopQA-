from custom_langchain.retrievers import BM25SRetriever
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pprint import pprint

def metadata_func(record: str, metadata: str) -> str:
    metadata['title'] = record.get('title')
    metadata['url'] = record.get('url')
    metadata['categories'] = record.get('categories')
    return metadata

loader = JSONLoader(
    file_path="medical_articles.jsonl",
    jq_schema=".",
    content_key="text",
    metadata_func=metadata_func,
    json_lines=True
)

docs = loader.load()

for doc in docs:
    if 'source' in doc.metadata and 'seq_num' in doc.metadata:
        del doc.metadata['source']
        del doc.metadata['seq_num']

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, 
    chunk_overlap=32, 
    separators=["\n\n", "\n", ". ", " ", ""]
)

split_docs = text_splitter.split_documents(docs)

# Extract just the page_content from each Document to get a list of strings
text_contents = [doc.page_content for doc in split_docs]
metadata = [doc.metadata for doc in split_docs]

# Create the retriever once, outside the loop
retriever = BM25SRetriever.from_texts(text_contents, metadata, k=2, persist_directory='bm25_medrag')
