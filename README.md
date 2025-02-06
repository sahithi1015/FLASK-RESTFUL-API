# FLASK-RESTFUL-API
!pip install --upgrade pip
!pip install langchain==0.0.148 chromadb pydantic==1.10.7
# Step 1: Import necessary modules
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from flask import Flask, request, jsonify
import os
# Step 2: Extract data from the technical courses page using WebBaseLoader
url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
documents = loader.load()
# Example: print the first 500 characters of the first document
print(documents[0].page_content[:500])
# Step 3: Create embeddings and store them in Chroma vector store
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(documents, embeddings)
# Step 4: Create a conversation chain with OpenAI LLM
llm = OpenAI(temperature=0)  # Low temperature for deterministic responses
conversation_chain = ConversationChain(llm=llm, memory=vector_store)
# Step 5: Set up Flask API
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    # Extract user input from the JSON request
    user_input = request.json.get("message")

    # Get the chatbot's response
    response = conversation_chain.run(user_input)

    # Return the response as JSON
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
