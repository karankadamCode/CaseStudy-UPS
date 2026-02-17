"""
faiss_ingest.py

Purpose:
Loads a PDF, splits it into overlapping text chunks, embeds the chunks using
OpenAI embeddings, and builds a FAISS index saved to disk. This index is later
used by the RAG runtime for retrieval.

Workflow:
1) Load PDF pages using PyPDFLoader
2) Chunk pages using RecursiveCharacterTextSplitter
3) Create FAISS vectorstore from chunks + embeddings
4) Persist FAISS index under vectorstore/faiss_index

Author:
Karan Kadam
"""

import os
import logging
from dotenv import load_dotenv
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------
# Environment & Config
# ----------------------------
load_dotenv()

PDF_PATH = "data/AI Enginner Use Case Document.pdf"
VECTORSTORE_PATH = "vectorstore/faiss_index"

EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ----------------------------
# Document Loading
# ----------------------------
def load_documents(pdf_path: str) -> List[Document]:
    """
    Load a PDF into a list of LangChain Document objects (one per page).

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of Document objects with page content and metadata.

    Raises:
        FileNotFoundError: If PDF path does not exist.
        RuntimeError: If PDF loading fails.
    """
    
    logger.info("Starting PDF load")
    logger.info(f"PDF path: {pdf_path}")

    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found at path: {pdf_path}")
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        logger.exception("Failed to load PDF")
        raise RuntimeError("PDF loading failed") from e

    if not documents:
        logger.warning("PDF loaded but no pages were extracted")

    logger.info(f"Successfully loaded {len(documents)} pages from PDF")
    return documents

# ----------------------------
# Chunking
# ----------------------------
def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping text chunks for embedding.

    Args:
        documents: List of Document objects (typically per page).
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap characters between adjacent chunks.

    Returns:
        List of chunked Document objects.

    Raises:
        ValueError: If documents is empty or chunk params are invalid.
        RuntimeError: If chunking fails.
    """
    
    logger.info("Starting document chunking")
    logger.info(
        f"Chunk size: {CHUNK_SIZE}, Chunk overlap: {CHUNK_OVERLAP}"
    )

    if not documents:
        logger.error("No documents provided for chunking")
        raise ValueError("Document list is empty")

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)
    except Exception as e:
        logger.exception("Document chunking failed")
        raise RuntimeError("Chunking failed") from e

    if not chunks:
        logger.warning("Chunking completed but produced zero chunks")

    logger.info(f"Chunking completed: {len(chunks)} chunks created")
    return chunks

# ----------------------------
# FAISS Vector Store Creation
# ----------------------------
def build_faiss_index(chunks: List[Document]):
    """
    Build and persist a FAISS index from chunked documents.

    Args:
        chunks: Chunked Document list.
        vectorstore_path: Directory path where FAISS index will be saved.
        embedding_model: OpenAI embedding model name.

    Returns:
        The path where the FAISS index was saved.

    Raises:
        ValueError: If chunks is empty or vectorstore_path is invalid.
        RuntimeError: If FAISS creation or saving fails.
    """
    
    logger.info("Starting FAISS index creation")

    if not chunks:
        logger.error("No chunks provided for FAISS indexing")
        raise ValueError("Chunk list is empty")

    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings,
        )
    except Exception as e:
        logger.exception("Failed to create FAISS vector store")
        raise RuntimeError("FAISS index creation failed") from e

    try:
        os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
        vectorstore.save_local(VECTORSTORE_PATH)
    except Exception as e:
        logger.exception("Failed to save FAISS index to disk")
        raise RuntimeError("FAISS persistence failed") from e

    logger.info(f"FAISS index successfully saved at: {VECTORSTORE_PATH}")

# ----------------------------
# Execution
# ----------------------------
# if __name__ == "__main__":
#     logger.info("===== FAISS INGESTION PIPELINE STARTED =====")

#     try:
#         documents = load_documents(PDF_PATH)
#         chunks = split_documents(documents)
#         build_faiss_index(chunks)
#     except Exception as e:
#         logger.error("FAISS ingestion pipeline failed")
#         logger.error(str(e))
#         raise
#     else:
#         logger.info("FAISS ingestion pipeline completed successfully")

#     logger.info("===== PIPELINE FINISHED =====")