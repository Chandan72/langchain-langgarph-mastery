import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# --- Standard Setup ---
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.4)


# --- Step 1: Define Your Data Schema using Pydantic ---

# Here, we define what our structured output should look like.
# We inherit from Pydantic's 'BaseModel'.
class Recipe(BaseModel):
    # Field provides a way to add a description, which the LLM uses for better results.
    # We use type hints to specify the expected data type (e.g., str, List[str]).
    recipe_name: str = Field(description="The name of the recipe.")
    ingredients: List[str] = Field(description="A list of ingredients required for the recipe.")
    instructions: List[str] = Field(description="A step-by-step list of instructions to prepare the recipe.")
    serving_size: int = Field(description="The number of people the recipe is intended to serve.")
    difficulty: str = Field(description="The estimated difficulty level, e.g., 'Easy', 'Medium', or 'Hard'.")


print("--- Data Schema 'Recipe' defined successfully. ---")

# We will continue from here...
# (Your existing code from the last step should be above this)
# ...
# class Recipe(BaseModel):
#     ...

# --- Step 2: Instantiate the Parser ---
# We create an instance of the parser and tell it what Pydantic model to use.
parser = PydanticOutputParser(pydantic_object=Recipe)

# --- Step 3: Craft the Prompt Template ---
# Here's the key step. We get formatting instructions from the parser.
# This method returns a string with instructions on how the LLM should format its output.
format_instructions = parser.get_format_instructions()

# Now we create our prompt template. Notice the new "{format_instructions}" partial variable.
prompt = PromptTemplate(
    template="You are a helpful assistant who generates recipes. Answer the user's query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions}
)

# --- Step 4: Assemble the Chain ---
# The chain structure remains elegant and simple with LCEL.
chain = prompt | model | parser

# --- Step 5: Invoke the Chain and Observe ---
query = "What is a good recipe that uses fresh tomatoes, basil, and mozzarella cheese?"
print(f"\nSending query: '{query}'\n")

# Run the chain. The output will not be a string, but a Pydantic 'Recipe' object.
recipe_object = chain.invoke({"query": query})

# --- Final Analysis ---
print("--- Structured Output Received ---")
print(recipe_object)

# You can now access the data like a regular Python object!
print("\n--- Accessing individual fields ---")
print(f"Recipe Name: {recipe_object.recipe_name}")
print(f"Serves: {recipe_object.serving_size}")
print(f"First ingredient: {recipe_object.ingredients[0]}")