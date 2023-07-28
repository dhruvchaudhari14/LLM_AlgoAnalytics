
import langchain
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain.document_loaders import WebBaseLoader

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter

from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import pickle

# Step 1: Define the file path where you saved the document
file_path = "data"

# Step 2: Open the file in binary mode and use pickle to load the object
with open(file_path, "rb") as file:
    data = pickle.load(file)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
texts = text_splitter.split_documents(data)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
final_embeddings= HuggingFaceEmbeddings()
docsearch = FAISS.from_documents(texts, final_embeddings)
llm=HuggingFaceHub(repo_id="google/flan-t5-xxl",model_kwargs={"temperature": 0.4, "max_length": 500,"batch_size":32})
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":7})
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


query = "Give details about nursemaid's elbow injury "
qa(query)
