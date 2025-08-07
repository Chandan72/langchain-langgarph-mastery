import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.4)
prompt = ChatPromptTemplate.from_template(
    "you are a professional researcher in python programming "
    "your work is to explain this code snippet {code_snippet} in "
    "{language} language. make sure that explanation is for {target_audience} audience. "
    "please try to explain in one paragraph."
)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

async def get_analysis(code , language, audience):
    """
    An asynchronous function to run a single analysis chain.
    """
    print(f"--- Starting analysis for: {audience}... ---")
    # Step 2: Use 'await' to call the non-blocking .ainvoke() method.
    response = await chain.ainvoke({"code_snippet": code, "language": language, "target_audience": audience})
    print(f"\n✅ === Analysis Complete for: {audience} === ✅\n{response}\n")

async def main():
    """
    The main function that runs multiple analyses concurrently.
    """
    start_time = time.time()
    
    # Step 4: Use asyncio.gather to run all our 'get_analysis' tasks concurrently.
    await asyncio.gather(
        get_analysis(
            'async def get_analysis(tech, trend):An asynchronous function to run a single analysis chain print(f"--- Starting analysis for: {tech}... ---") response = await chain.ainvoke({"tech_name": tech, "market_trend": trend}) print(f"\n✅ === Analysis Complete for: {tech} === ✅\n{response}\n" ',
        
         
            'English',
            'AI Developers'
        ),
        
        get_analysis(
            'async def get_analysis(tech, trend):An asynchronous function to run a single analysis chain print(f"--- Starting analysis for: {tech}... ---") response = await chain.ainvoke({"tech_name": tech, "market_trend": trend}) print(f"\n✅ === Analysis Complete for: {tech} === ✅\n{response}\n" ',
            'English',
            'non-technical audience'
        ) ,
       get_analysis(
            'async def get_analysis(tech, trend):An asynchronous function to run a single analysis chain print(f"--- Starting analysis for: {tech}... ---") response = await chain.ainvoke({"tech_name": tech, "market_trend": trend}) print(f"\n✅ === Analysis Complete for: {tech} === ✅\n{response}\n" ',
            'Hindi',
            'for indian rular audience'
        )
    )
    
    end_time = time.time()
    print(f"--- All analyses completed in {end_time - start_time:.2f} seconds ---")


if __name__== "__main__":
    asyncio.run(main())

        # This command tells Python to run our main asynchronous function.
    