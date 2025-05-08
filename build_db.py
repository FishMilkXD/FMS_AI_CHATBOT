import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredCSVLoader
)
from langchain_core.documents import Document
import pandas as pd

# === config ===
DOC_FOLDER = "/Users/sphiere/Desktop/Khayri's codes/FMS_AI_CHATBOT/FMS_AI_CHATBOT/test_files"
DB_PATH = "/Users/sphiere/Desktop/Khayri's codes/FMS_AI_CHATBOT/FMS_AI_CHATBOT/chroma_db"
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

def load_excel_fallback(filepath):
    try:
        xl = pd.read_excel(filepath, sheet_name=None)
        docs = []
        for sheet_name, sheet in xl.items():
            for _, row in sheet.iterrows():
                content = "\n".join(f"{col}: {row[col]}" for col in sheet.columns)
                docs.append(Document(page_content=content))
        return docs
    except Exception as e:
        print(f"Failed to load Excel fallback for {filepath}: {e}")
        return []



for filename in os.listdir(DOC_FOLDER):
    filepath = os.path.join(DOC_FOLDER, filename)
    
    docs = [] # reset docs for each file
    
    try:
        if filename.endswith(".pdf"):
            loader = PDFPlumberLoader(filepath)
        elif filename.endswith("docx"):
            loader = Docx2txtLoader(filepath)
        elif filename.endswith("xlsx"):
            try:
                loader = UnstructuredExcelLoader(filepath)
                docs = loader.load()
            except Exception as e:
                print(f"Fallback loading {filename} due to: {e}")
                docs = load_excel_fallback(filepath)
                
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
            
        # prints the loaded document back (debug)
        if len(docs) == 0:
            print(f"  {filename} loaded 0 docs. Check file content or fallback loaders.")
        else:
            print(f" Loaded {filename} with {len(docs)} docs. Sample content from first doc:")
            print("=" * 60)
            print(docs[0].page_content[:500])  # preview first 500 chars of first doc
            print("=" * 60)

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
