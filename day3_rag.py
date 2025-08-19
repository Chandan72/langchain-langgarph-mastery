from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS


#create a loader instance by calling the WebBaseLoader class, by adding the url of website

loader=WebBaseLoader("https://paulgraham.com/hwh.html")

#Call the .load() method to featch and parse method

print("loadimg document........")

docs=loader.load()
print("document loaded succesfully")
#print(docs[0].page_content[:500])


text_splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
print("\n splitting document into chunks")
splits=text_splitter.split_documents(docs)

#print(splits[5].page_content[:500])

embedding_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
sample_embed=embedding_model.aembed_query("chandan")
#print(sample_embed[:5])

print("creating and storing the embedding vector into vectordatabase called FAISS")

vectorstore=FAISS.from_documents(
    documents=splits,
    embedding=embedding_model
)
print("vectordatabase created sucessfully")

retriever=vectorstore.as_retriever(kwargs={"k":3})
#query="what hard work?"
#print(retriever.invoke(query))
prompt=ChatPromptTemplate.from_template(
    """you are assistant, who is expert in finding answer from the documents
     give me the answer of this {query}
    from {context}"""
)
model=GoogleGenerativeAI(model="gemini-1.5-pro-latest", temprature=0)
chain=(
    {"context":retriever, "query":RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()

)

query="what is the real hard work"

response=chain.invoke(query)
print(response)
     



