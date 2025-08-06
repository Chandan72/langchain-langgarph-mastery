import asyncio
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Standard Setup (Same as before) ---
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
prompt = ChatPromptTemplate.from_template(
    "Provide a brief, one-paragraph analysis of the impact of the '{market_trend}' "
    "to best oppertunities to start a new start-up using '{tech_name}' technologies."
)
output_parser = StrOutputParser()
chain = prompt | model | output_parser


# --- New Asynchronous Structure ---

# Step 1: Define an 'async def' function. This is a coroutine.
async def get_analysis(tech, trend):
    """
    An asynchronous function to run a single analysis chain.
    """
    print(f"--- Starting analysis for: {tech}... ---")
    # Step 2: Use 'await' to call the non-blocking .ainvoke() method.
    # The program can work on other tasks while this is "awaiting".
    response = await chain.ainvoke({"tech_name": tech, "market_trend": trend})
    print(f"\n✅ === Analysis Complete for: {tech} === ✅\n{response}\n")


# Step 3: Create a main async function to orchestrate our tasks.
async def main():
    """
    The main function that runs multiple analyses concurrently.
    """
    start_time = time.time()
    
    # Step 4: Use asyncio.gather to run all our 'get_analysis' tasks concurrently.
    # It doesn't run them one by one. It starts them all at once.
    await asyncio.gather(
        get_analysis("Langchain and langgraph", "improvement in the thinking in LLM"),
        get_analysis("trasformers", "the future of AI and LLMs"),
        get_analysis("MCP server", "Agentic AI and LLMs")
    )
    
    end_time = time.time()
    print(f"--- All analyses completed in {end_time - start_time:.2f} seconds ---")


# Step 5: The entry point for the script.
if __name__ == "__main__":
    # This command tells Python to run our main asynchronous function.
    asyncio.run(main())