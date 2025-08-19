import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever

#load the documents
loader= WebBaseLoader("http://www.paulgraham.com/worked.html")
docs=loader.load()

#splits 

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits=text_splitter.split_documents(docs)

#store

load_dotenv() 
embeddings_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore=FAISS.from_documents(splits,embeddings_model)

# create the advanced retriever

model=ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

multi_query_retriver=MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever() ,
    llm=model
)

#creating the RAG chain for multiqueryretriever
print("now we are going to create the RAG chain")

prompt=ChatPromptTemplate.from_template(
    """answer the following question based only on 
      <context>
       {context}
        <context>
        Question:{input}"""
    
)

rag_chain=(
    {"context": multi_query_retriver, "input": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

question="what was his early life like?"
answer=rag_chain.invoke(question)
print(f"Answer:{answer}")
