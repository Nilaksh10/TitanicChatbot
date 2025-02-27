import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.vectorstores import FAISS  # Import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Step 1: Download the Titanic Dataset from GitHub if not present
csv_file = "titanic.csv"
if not os.path.exists(csv_file):
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    response = requests.get(url)
    with open(csv_file, "wb") as file:
        file.write(response.content)

# Initialize FastAPI app
app = FastAPI()

# Step 2: Get Hugging Face API Token
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the .env file")

# Step 3: Load CSV and Convert to Documents
loader = CSVLoader(file_path=csv_file)
data = loader.load()

documents = [
    Document(page_content=", ".join([f"{key}: {value}" for key, value in row.items()])) 
    for row in data
]

# Step 4: Initialize Embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 5: Load or Create FAISS Index
faiss_path = "faiss_index"
if os.path.exists(faiss_path):
    faiss_index = FAISS.load_local(faiss_path, embedding)
else:
    faiss_index = FAISS.from_documents(documents, embedding)
    faiss_index.save_local(faiss_path)  # Save for future use

# Step 6: Set Up Falcon Model
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=hf_api_token,
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Step 7: Define Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Based on the Titanic dataset and the following context:\n\n{context}\n\nAnswer this question: {question}"
)

# Step 8: Create a LangChain LLMChain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Define API Request and Response Models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

# API Endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        question = request.question

        # Retrieve relevant documents
        docs = faiss_index.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # Get answer from LLM
        answer = chain.run({"context": context, "question": question})

        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI App
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

