import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Step 1: Download the Titanic Dataset from GitHub
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
response = requests.get(url)
with open("titanic.csv", "wb") as file:
    file.write(response.content)

# Initialize FastAPI app
app = FastAPI()

# Step 2: Set Hugging Face API Token from .env
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the .env file")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# Step 3: Load Titanic Dataset from Local CSV
loader = CSVLoader(file_path="titanic.csv")

# Step 4: Specify an Embedding Model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 5: Create a Vectorstore Index with the Embedding Model
index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])

# Step 6: Set Up the Chatbot with Falcon Model
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",  # Falcon model repository ID
    model_kwargs={"temperature": 0.7, "max_length": 512}  # Adjust parameters
)

# Step 7: Define a Prompt Template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question based on the Titanic dataset: {question}"
)

# Step 8: Create a LangChain LLMChain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Define a request model for the API
class ChatRequest(BaseModel):
    question: str

# Define a response model for the API
class ChatResponse(BaseModel):
    answer: str

# API endpoint to handle chatbot requests
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Get the question from the request
        question = request.question

        # Get the answer from the chatbot
        answer = chain.run(question)

        # Return the answer
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))  # Render assigns a dynamic port
    uvicorn.run(app, host="0.0.0.0", port=port)
