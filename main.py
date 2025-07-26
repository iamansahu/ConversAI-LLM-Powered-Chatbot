import openai
import langchain
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

#Read the document
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    file_loader.load()
    return documents

#Split the doc into chunks

def chunk_data(docs, chunk_size=1000, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc

documents = chunk_data(docs=doc)

#Embedding technique of OPEN AI
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

#Initialize the Pinecone vector store
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

index_name = "langchainvector"

index = Pinecone.from_document(doc, embeddings, index_name = index_name )

#Cosines Similarity retreive results
def retrieve_results(query, top_k=3):
    matching_results = index.similarity_search(query, k=top_k)
    return matching_results


from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

llm = OpenAI(model_name = "text-davinci-003",temperature=0.5)
chain = load_qa_chain(llm, chain_type="stuff")

def retrieve_answer(query):
    doc_serch = retrieve_results(query)
    print(doc_serch)
    response = chain.run(input_documents=doc_serch, question=query)
    return response

# Example
our_query = "What is the main topic of the document?"
answer = retrieve_answer(our_query)
print(f"Query: {our_query}\nAnswer: {answer}")