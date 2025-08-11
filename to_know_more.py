from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from langchain_core.memory import ConversationBufferMemory

# --- Standard Setup ---
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.8)
# --- Step 1: Define Your Data Schema using Pydantic ---
class userProfile(BaseModel):
    name: str = Field(description="The name of the user.")
    age: int = Field(description="The age of the user.")
    proffession: str = Field(description="The profession or job title of the user.")
    skills: List[str] = Field(description="A list of skills or expertise of the user.")
    experience_years: int = Field(description="The number of years of experience in their field.")
    education: str = Field(description="The highest level of education attained by the user.")
    interests: List[str] = Field(description="A list of interests or hobbies of the user.")
    location: str = Field(description="The geographical location of the user.")

print("--- Data Schema 'userProfile' defined successfully. ---")
# --- Step 2: Instantiate the Parser ---
parser = PydanticOutputParser(pydantic_object=userProfile)
format_instructions = parser.get_format_instructions()

# --- Step 3: Craft the Prompt Template ---
prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "you are a friendly AI assistant conducting a user profile interview"
    "your goal is to gather detailed information about the user, having a natural conversation."
    "IMPORTANT: IF the user's input is the single word 'summarize',you must stop the conversation."
    "your only job then is to summarize the information gathered so far"
     "The response should be in this format: {format_instructions}\n"
),
        ("human", " the conversation till now {history}n/n{input}"),
        ("ai", "Please provide your response in the following format:\n{format_instructions}")
    ],
    partial_variables={"format_instructions": format_instructions}
)

print("--- Prompt Template created successfully. ---")

Chain= prompt | model | parser
async def user_profile_interview():
    """
    An asynchronous function to conduct a user profile interview.
    """
    print("--- Starting user profile interview... ---")
    memory = ConversationBufferMemory()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "summarize":
            break
        
        response = await Chain.ainvoke({"input": user_input, "history": memory.get_history()})
        print(f"AI: {response}")
        memory.add_message("human", user_input)
        memory.add_message("ai", response)

    summary = await Chain.ainvoke({"input": "summarize", "history": memory.get_history()})
    print(f"\nSummary of User Profile:\n{summary}")



