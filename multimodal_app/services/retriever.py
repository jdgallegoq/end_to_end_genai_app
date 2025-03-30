import tempfile
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import streamlit as st

from config.config import openai_api_key

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyMuPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split into documents chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                                    chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(docs)

    # Create document embeddings and store in Vector DB
    embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
    vectordb = Chroma.from_documents(doc_chunks, embeddings_model)

    # Define retriever object
    retriever = vectordb.as_retriever()
    return retriever
