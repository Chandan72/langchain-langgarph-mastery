from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

# This class defines the structure of our "fillable form"
class QualifiedLead(BaseModel):
    """A lead that has been qualified by an AI assistant."""

    contact_name: str = Field(description="The full name of the person who sent the inquiry.")
    company_name: Optional[str] = Field(description="The name of the company the person works for. Can be null if not mentioned.")
    company_size: Optional[int] = Field(description="The number of employees at the company. Can be null if not mentioned.")
    summary: str = Field(description="A one-sentence summary of the user's request or problem.")
    is_qualified: bool = Field(description="Set to True if this seems like a genuine business inquiry for our services, otherwise set to False.")
    priority: str = Field(description="Assign a priority level: 'High', 'Medium', or 'Low'.")

print("Blueprint for QualifiedLead has been defined!")

model=ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

# (This code goes after your QualifiedLead class definition)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# First, create the parser instance from our blueprint
parser = PydanticOutputParser(pydantic_object=QualifiedLead)

# Now, build the prompt, including the instructions from the parser
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert assistant for a tech startup. Your job is to analyze new customer inquiry emails and extract key information in a structured format.",
        ),
        (
            "human",
            "Please extract the key information from this email: {inquiry_email}\n\n{format_instructions}",
        ),
    ]
)

print("Prompt and Parser are ready!")

chain= prompt| model | parser

sample_email ="""
hey,
my name is chandan, i am recently graduted from IIT kharagpur, i am building a new AI startup that focuses on AI agents and LLMs,
we are a team of 5 people, we are looking for a AI agency that can help us in building our product and also help us in the marketing of our product.
i am founder and ceo of this company, the company name is Introverted.AI so please let me know if you are interested in working with us.

"""

print("Analyzing email....")
lead_object= chain.invoke({
    "inquiry_email": sample_email,
    "format_instructions": parser.get_format_instructions()
})

print("\n--- Lead analysis complete!---")
print(lead_object)

print("\n---Automated Action---")
if lead_object.is_qualified and lead_object.priority == "High":
    print(f"Send a follow-up email to {lead_object.contact_name} at {lead_object.company_name} to schedule a meeting.")
    print("ACTION: Automatically adding to salesforce and notifying the sales team.")
else:
    print(f"Lead from {lead_object.contact_name} is not qualified. No further action required.")
    print("ACTION: Automatically archiving the email and notifying the team.")