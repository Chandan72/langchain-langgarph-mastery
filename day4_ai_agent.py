from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()

# set our brain(llm)the ceo
model= ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

#set ours tools(the employee)
search_tool=TavilySearchResults()
tools=[search_tool]
# the prompt
# the ReAct framework needs a specific set of instructions.
prompt= ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant.messages="),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
# create the agent
# this function combines the llm and the prompt to create the core of the  agent
agent=create_tool_calling_agent(model,tools, prompt)
# create the agent executor
# this is the final engine that will run the "thought"-> Action ->Observation
agent_executor=AgentExecutor(agent=agent, tools=tools, verbose=True)
# run the agent


print("\n--- Running the Agent Executor ---")
question = "Who is the current Prime Minister of India, and what is the current weather in their capital city?"
response = agent_executor.invoke({"input": question})

print("\n--- Final Answer from Agent ---")
print(response["output"])