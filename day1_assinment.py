import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.4)
prompt= ChatPromptTemplate.from_template(
    "you are a proffessional researcher in python programming"
    "your work is to explain this code snippet {code_snippet} in"
    "{language} language. make sure that explaination is for {target_audience} audience."
    "please try to explain in one paragraph."
)
output_parser = StrOutputParser()

chain= prompt | model | output_parser

print("welcome to the code explanation tool that streaming the response")

input_data={
    "code_snippet": 
       """ async def get_analysis(tech, trend):
        
         An asynchronous function to run a single analysis chain.

        print(f"--- Starting analysis for: {tech}... ---")
        # Step 2: Use 'await' to call the non-blocking .ainvoke() method.
        # The program can work on other tasks while this is "awaiting".
        response = await chain.ainvoke({"tech_name": tech, "market_trend": trend})
        print(f"\n✅ === Analysis Complete for: {tech} === ✅\n{response}\n")"""
    
     
,
    "language": "English",
    "target_audience": "AI Developers"
}
for chunk in chain.stream(input_data):
    sanitize_chunk = chunk.replace("\n", " ")
    print(sanitize_chunk, end="", flush=True)
