import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredCSVLoader
)
from langchain_core.documents import Document
import pandas as pd

# === config ===
DOC_FOLDER = "/home/ikanbesar/Documents/ai_chatbot/test_files"
DB_PATH = "/home/ikanbesar/Documents/ai_chatbot/chroma_db"
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# === load docs from folder ===
all_docs = []

def load_csv_fallback(filepath):
    try:
        file = pd.read_csv(filepath)
        docs = []
        for _, row in file.iterrows():
            content = "\n".join(f"{col}: {row[col]}" for col in file.columns)
            docs.append(Document(page_content=content))
        return docs
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return []
    
for filename in os.listdir(DOC_FOLDER):
    filepath = os.path.join(DOC_FOLDER, filename)
    
    docs = [] # reset docs for each file
    
    try:
        # loads either .pdf, .docx, or . xlsx files
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith("docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
        elif filename.endswith("xlsx"):
            loader = UnstructuredExcelLoader(filepath)
        elif filename.endswith("csv"):
            try:
                loader = UnstructuredCSVLoader(filepath)
                docs = loader.load()
            except Exception as e:
                print(f"Fallback loading {filename} due to: {e}")
                docs = load_csv_fallback(filepath)
                
        else: # skips any files which are not the supported file types
            print(f"Skipping unsupported file: {filename}")
            continue
    
        print(f" Loaded {filename} with {len(docs)} docs")
        all_docs.extend(docs)
        
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        
 
    
# === SPLIT TEXT INTO CHUNKS ===
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(all_docs)
print(f"Total split chunks: {len(split_docs)}")

# === CREATE VECTOR ====
vectorstore = Chroma.from_documents(split_docs, embedding=EMBEDDING_MODEL, persist_directory=DB_PATH)

print(f" ChromaDB built and saved to: {DB_PATH}")
