import tempfile
import os
import uuid
import logging
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import streamlit as st

from config.config import openai_api_key

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    """
    Configure a retriever from uploaded PDF files.
    Uses ChromaDB as the vector store.
    """
    try:
        # Create a unique directory for this session
        if "chroma_db_dir" not in st.session_state:
            st.session_state.chroma_db_dir = os.path.join(tempfile.gettempdir(), f"chroma_db_{uuid.uuid4().hex}")
            logger.info(f"Creating ChromaDB at {st.session_state.chroma_db_dir}")
        
        # Process documents
        docs = process_documents(uploaded_files)
        if not docs:
            logger.warning("No documents were successfully loaded")
            return None

        # Create embeddings
        embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
        
        # Create vector store
        vectordb = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings_model,
            persist_directory=st.session_state.chroma_db_dir
        )
        
        # Create retriever
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        logger.info(f"Retriever created successfully with {len(docs)} document chunks")
        return retriever
        
    except Exception as e:
        logger.error(f"Error configuring retriever: {str(e)}")
        st.error(f"Error setting up document retrieval: {str(e)}")
        return None

def process_documents(uploaded_files) -> List:
    """Process the uploaded PDF files into document chunks."""
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    
    logger.info(f"Processing {len(uploaded_files)} PDF files")
    
    for file in uploaded_files:
        try:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = PyMuPDFLoader(temp_filepath)
            file_docs = loader.load()
            logger.info(f"Loaded {len(file_docs)} documents from {file.name}")
            docs.extend(file_docs)
        except Exception as e:
            logger.error(f"Error loading file {file.name}: {str(e)}")
    
    if not docs:
        return []
        
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(docs)
    logger.info(f"Created {len(doc_chunks)} document chunks")
    
    return doc_chunks
