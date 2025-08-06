import os
from dotenv import load_dotenv
# load the environment variables from .env file
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model= ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.4)
prompt= ChatPromptTemplate.from_template(
    "you are a survvy market analyst with a knack for clear communication."
    "Provide a brief, one-paragraph analysis of the impact of the '{market_trend}' "
    " trend on the '{company_name}'."
)

output_parser= StrOutputParser()

chain= prompt | model | output_parser

print("welcome to the market analysis tool")

response=chain.invoke({"company_name": "meta",
                      "market_trend": "AI and AI agents future planing"})
print(f"Response: {response}")
