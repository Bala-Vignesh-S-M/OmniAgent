import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_resume(pdf_path: str = "BalaVignesh_SM_Resume.pdf", index_name: str = "faiss_index"):
    """Loads a PDF resume, chunks it, and creates a local FAISS index."""
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found. Please place your resume in the same directory.")
        return

    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)

    print("Generating embeddings and creating FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_name)
    print(f"Success! Knowledge base saved to directory: {index_name}")

if __name__ == "__main__":
    ingest_resume()
