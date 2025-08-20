import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.tools.tavily_search import TavilySearchResults

load_dotenv()

@tool
def multiply(a:int, b:int)->int:
    """Multiplies two numbers together. use this for math questions."""
    print("Debug: using multply tool")
    return a * b

@tool
def get_customer_details(customer_name:str)-> str:
    """looks up key information about a custorm from the company's private CRM database.
        use this tool when you need to find a customer's status, account managet, or contact details."""
    print(f"DEBUG: Searching mock database for {customer_name}....")
    mock_crm={
        "Innovate crop": {"status": "active", "account_managet": "Bob", "contact": "chandan875792@gmail.com"},
        "Tech solutions ltd": {"status": "Trial", "account_manager": "alice", "contact": "kundan8757@gmail.com"}
    }
    return str(mock_crm.get(customer_name, "Customer not found in the database."))

search_tool= TavilySearchResults()

tools= [search_tool, multiply, get_customer_details]
model= ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

prompt= ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant with access to several tools."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]

)

agent=create_tool_calling_agent(model, tools, prompt)
agent_executor= AgentExecutor(agent=agent, tools=tools, verbose=True)

response1=agent_executor.invoke({"input": "what is the current population of kolkata?"})
print(f"answer: {response1["output"]}")

response2= agent_executor.invoke({"input": "what is 35 multiplied by 42?"})
print(f"answer:{response2["output"]}")

question="what company did sam altman co-found before OpenAI?"
question2="what is the status of our customer 'Innovate crop'?"
response3=agent_executor.invoke({"input": question})
response4=agent_executor.invoke({"input": question2})
print(f"Answer:{response4["output"]}")
