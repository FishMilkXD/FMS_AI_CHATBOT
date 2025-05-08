import os
import requests
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader
)

# === API stuff ===
# export the api key into the terminal first:
# export GROQ_API_KEY="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"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# === config ===
DOC_FOLDER = "/home/ikanbesar/Documents/ai_chatbot/test_files"
DB_PATH = "/home/ikanbesar/Documents/ai_chatbot/chroma_db"
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# === load docs from folder ===
all_docs = []

for filename in os.listdir(DOC_FOLDER):
    filepath = os.path.join(DOC_FOLDER, filename)
    
    # loads either .pdf, .docx, or . xlsx files
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif filename.endswith("docx"):
        loader = UnstructuredWordDocumentLoader(filepath)
    elif filename.endswith("xlsx"):
        loader = UnstructuredExcelLoader(filepath)
    else: # skips any files which are not the supported file types
        print(f"Skipping unsupported file: {filename}")
        continue
    
    # exception handler
    try:
        docs = loader.load()
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

# retrieves docs for context
def retrieve_relevant_context(query, k = 3):
    docs = vectorstore.similarity_search(query, k = k)
    return "\n\n".join([doc.page_content for doc in docs])

def chat_with_bot(prompt):
    
    # get relevant docs
    context = retrieve_relevant_context(prompt)
    
    # format prompt for groq
    final_prompt = f"""You are a helpful assistant. Use the information below to answer the user's question.

Context:
{context}

User Question:
{prompt}
"""
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": final_prompt}
        ],
        "temperature": 0.7
    }
    
    response = requests.post(GROQ_API_URL, headers = headers, json = data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")
        
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            break
            
        try:
            response = chat_with_bot(user_input)
            print("Chatbot: ", response)
        except Exception as e:
            print("An error has occured", e)