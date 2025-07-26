# ConversAI – LLM-Powered Chatbot for PDF-Based Question Answering

ConversAI is an intelligent chatbot that leverages OpenAI's language models and Pinecone's vector database to answer user queries based on the content of uploaded PDF documents. It uses LangChain to orchestrate document loading, chunking, embedding, storage, retrieval, and response generation — all with natural language.

---

## 🧠 Key Features

- Ingests and parses multiple PDF documents
- Splits large documents into manageable, overlapping chunks
- Uses OpenAI embeddings to vectorize text
- Stores and searches document chunks via Pinecone vector store
- Answers user queries using OpenAI’s GPT models with LangChain’s QA chains

---

## 🧰 Tech Stack

| Tool           | Purpose                              |
|----------------|--------------------------------------|
| **LangChain**  | Framework for chaining LLM tasks     |
| **OpenAI**     | Embedding + language generation      |
| **Pinecone**   | Vector storage and similarity search |
| **PyPDF**      | Document parsing                     |
| **dotenv**     | Secret key management                |


---

## Add Your PDF Documents   
Place your PDF documents inside a folder (e.g., ./pdf_docs) and modify the path in the code:   
documents = read_doc("./pdf_docs")

---
## Configure API Keys   
Create a .env file in the project root:   

OPENAI_API_KEY=your_openai_api_key   
PINECONE_API_KEY=your_pinecone_api_key   
PINECONE_ENVIRONMENT=your_pinecone_env

---
## How It Works   
Load PDFs – Uses PyPDFDirectoryLoader to read PDF files.   
Split into Chunks – Breaks long texts into manageable chunks with overlaps using RecursiveCharacterTextSplitter.   
Generate Embeddings – Converts text chunks into high-dimensional vectors using OpenAI embeddings.   
Store in Pinecone – Embeddings are stored and indexed in a Pinecone vector database.   
Query & Retrieve – Your natural language query is embedded and compared with stored chunks using cosine similarity.   
Answer with LLM – The top relevant documents are passed into an OpenAI-powered LLM to generate a final answer.   
 
