from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

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
string_parser= StrOutputParser()

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
    ]
)

print("--- Prompt Template created successfully. ---")
chat_chain= prompt | model | string_parser

summary_chain= prompt | model | parser
def run_interview():
    memory = ConversationBufferMemory()
    
    print("AI: Hello! I'm here to help build your professional profile. To start, what's your name?")
    
    while True:
        history_variables = memory.load_memory_variables({})
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("AI: Goodbye!")
            break

        # Check if we need to summarize or just chat
        if user_input.lower() == 'summarize':
            print("\nAI: Understood. Generating your profile summary...")
            # Use the summary_chain
            response = summary_chain.invoke({
                "history": history_variables.get("history", ""),
                "input": user_input,
                "format_instructions": format_instructions
            })
            print("\n--- Your User Profile Summary ---")
            print(response.model_dump_json(indent=2))
            break
        else:
            # Use the chat_chain
            response = chat_chain.invoke({
                "history": history_variables.get("history", ""),
                "input": user_input,
                "format_instructions": "" # Pass empty instructions for chat
            })
            print(f"AI: {response}")
            memory.save_context({"input": user_input}, {"output": response})

# --- Run the main function ---
if __name__ == "__main__":
    run_interview()