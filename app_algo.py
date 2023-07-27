
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
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# from langchain.chains.question_answering import load_qa_chain



# loader = TextLoader("/kaggle/input/qatext-processed/final.txt")
# data = loader.load()
# data



# print (f'You have {len(data)} document(s) in your data')
# print (f'There are {len(data[0].page_content)} characters in your document')



# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
# texts = text_splitter.split_documents(data)
# print (f'Now you have {len(texts)} documents')



# from getpass import getpass


# HUGGINGFACEHUB_API_TOKEN = getpass()




# final_embeddings= HuggingFaceEmbeddings()
# docsearch = FAISS.from_documents(texts, final_embeddings)
# llm=HuggingFaceHub(repo_id="google/flan-t5-xxl",model_kwargs={"temperature": 0.4, "max_length": 500,"batch_size":32})
# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":7})
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)



# query = "Give details about nursemaid's elbow injury "
# qa(query)


# # Function to process the documents
# def process_documents(data):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
#     texts = text_splitter.split_documents(data)

#     HUGGINGFACEHUB_API_TOKEN = st.text_input("Enter your HuggingFace Hub API Token:", type="password")

#     # Create the HuggingFaceEmbeddings and FAISS vector store
#     final_embeddings = HuggingFaceEmbeddings(huggingface_api_token=HUGGINGFACEHUB_API_TOKEN)
#     docsearch = FAISS.from_texts(texts, final_embeddings)

#     # Load the HuggingFace model for QA
#     llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.4, "max_length": 500, "batch_size": 32})

#     # Create the RetrievalQA chain
#     retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 7})
#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

#     return qa

# # Create a Streamlit app
# def main():
#     # Add a title to the app
#     st.title("Interactive QA with Streamlit")

#     # File uploader widget to upload the text file
#     uploaded_file = st.file_uploader("Upload a text file")
#     text = uploaded_file.read()
# #     st.write(text)

#     if uploaded_file is not None:
        
#         # Load data from the uploaded file
#         loader = TextLoader("text")
#         data = loader.load()

#         if data:
#             # Process the documents and create the QA chain
#             st.write(f'You have {len(data)} document(s) in your data')
#             st.write(f'There are {len("".join(data))} characters in your document')
#             qa = process_documents(data)

#             # Input widget to take user query
#             query = st.text_input("Enter your query:")

#             # Button to trigger the QA process
#             if st.button("Ask"):
#                 if query:
#                     # Perform the QA and display the results
#                     results = qa(query)
#                     st.write("Answer:")
#                     st.write(results["answer"])
#                     st.write("Context:")
#                     st.write(results["context"])
#                 else:
#                     st.warning("Please enter a query.")
#         else:
#             st.error("Failed to load data from the uploaded file.")
#     else:
#         st.info("Please upload a text file.")

# # Run the app
# if __name__ == "__main__":
#     main()


loader = TextLoader("final.txt")
data = loader.load()
data

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
texts = text_splitter.split_documents(data)

HUGGINGFACEHUB_API_TOKEN = st.text_input("Enter your HuggingFace Hub API Token:", type="password")

    # Create the HuggingFaceEmbeddings and FAISS vector store
final_embeddings = HuggingFaceEmbeddings(huggingface_api_token=HUGGINGFACEHUB_API_TOKEN)
docsearch = FAISS.from_texts(texts, final_embeddings)

    # Load the HuggingFace model for QA
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.4, "max_length": 500, "batch_size": 32})

    # Create the RetrievalQA chain
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 7})
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
