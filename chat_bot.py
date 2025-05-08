import os
import requests
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



# === API stuff ===
# export the api key into the terminal first:
# export GROQ_API_KEY="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"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# === file paths ===
DOC_FOLDER = "/Users/sphiere/Desktop/Khayri's codes/FMS_AI_CHATBOT/FMS_AI_CHATBOT/test_files"
DB_PATH = "/Users/sphiere/Desktop/Khayri's codes/FMS_AI_CHATBOT/FMS_AI_CHATBOT/chroma_db"

# === embedding model intialization ===
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# === call chroma db ===
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=EMBEDDING_MODEL)

# === retrieves docs for context ====
def retrieve_relevant_context(query, k = 8):
    docs = vectorstore.similarity_search(query, k = k)
    return "\n\n".join([doc.page_content for doc in docs])


# === chat bot ===
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