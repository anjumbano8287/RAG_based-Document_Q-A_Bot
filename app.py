from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
import streamlit as st
from time import sleep

llm=ChatGoogleGenerativeAI(model="gemini-3.7-flash")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db=None

def document_process(path):
    ## document loader


    loder=PyPDFLoader(path)
    doc=loder.load()

    ## text-splitting

    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    doc=splitter.split_documents(doc)

    ## embedding and storing in vector db

    embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_db=InMemoryVectorStore.from_documents(documents=doc,embedding=embeddings)
    st.session_state.vector_db=vector_db
    st.session_state.document_uploaded = True
# ## query:

# query="Whats the name of physician"
# documents=vector_db.similarity_search(query=query,k=2)
# print(len(documents),documents[0].page_content)


## 
# context=" "
# for doc in documents:
#     context= context+doc.page_content+"/n/n"

# prompt=f"""
#     You are a helpful assitant and provide answer based on the provided context
#     context: {context}, query: {query}
# """

llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# result=llm.invoke(prompt)
# print(result.content[0]["text"])

st.header("📃 Document Q&A Chatbot - Ask Anything")


if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
    



## document upload
if not st.session_state.document_uploaded:
    file = st.file_uploader(label="Select your pdf ",type="pdf")
    if file:
        with open("uploaded_file.pdf","wb") as f:
            f.write(file.getvalue())
        st.markdown("Document Uploaded Successfully !")

        with st.spinner("processing...."):
            document_process("./uploaded_file.pdf")

        st.markdown("Document process successfully !")
        sleep(2)
        st.rerun()
if st.session_state.document_uploaded and  st.session_state.vector_db:
    query = st.chat_input("Ask Anything related to document ? ")

    for onemessage in st.session_state.messages:
        role=onemessage['role']
        content=onemessage['content']

        st.chat_message(role).markdown(content)
    
    
    if query:

        st.chat_message('user').markdown(query)
        st.session_state.messages.append({'role':'user','content':query})
        document = st.session_state.vector_db.similarity_search(query,k=2)
        context=" "

        for doc in document:
            context+=doc.page_content+"/n/n"

        prompt=f""" You are a helpful assistant and you provide answers based on provided context : {context} , user question is :  {query}"""
        result = llm.invoke(prompt)

        st.chat_message('ai').markdown(result.content[0]['text'])
        st.session_state.messages.append({'role':'ai','content':result.content[0]['text']})
        print(result.content[0]['text'])