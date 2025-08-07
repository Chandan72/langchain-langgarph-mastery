import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.4)
prompt = ChatPromptTemplate.from_template(
    "you are expert in identifying the best opportunities for workflow automation with AI,"
    "your work is to analyze the full workflow of {job_profession} profession "
    "and try to indentify the best opportunities for workflow automation with AI."
    "make sure that analsis is for {target_audience} audience."
)
output_parser = StrOutputParser()

chain = prompt | model | output_parser
print("welcome to the workflow automation analysis tool that streaming the response")
input_data ={
    "job_profession": "software developer",
    "target_audience": "ceo of AI Agency"
}
for chunk in chain.stream(input_data):
    sanitize_chunk= chunk.replace("\n", " ")
    print(sanitize_chunk, end="", flush=True)