import sys
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from logger.logging import setup_logging
from exceptions.exception import PipelineException

logger = setup_logging(log_file_path='logs/build_faiss.log')

def build_local_index():
    try:
        logger.info("Starting FAISS index build")
        
        
        loader = TextLoader("data/raw/sherlock_mega_corpus.txt", encoding="utf-8")
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(docs)} chunks")

        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("Embeddings model loaded")

        
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("data/processed/faiss_sherlock")
        logger.info("FAISS index saved to data/processed/faiss_sherlock")
        
    except Exception as e:
        raise PipelineException("Failed to build FAISS index", sys)

if __name__ == "__main__":
    build_local_index()