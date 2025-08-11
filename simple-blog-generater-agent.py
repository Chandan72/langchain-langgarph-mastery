"""I am going to build a simple blog generator agent that can create a blog post based on a given topic with structured content for my company's blog."""
import asyncio
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()

# Define the model and prompt
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)
# Define the data schema using Pydantic
class BlogPost(BaseModel):
    title: str = Field(description="The title of the blog post.")
    content: str = Field(description="The main content of the blog post.")
    conclusion: str = Field(description="A concluding section summarizing the blog post.")  
    tags: list[str] = Field(description="A list of tags related to the blog post.")
parser= PydanticOutputParser(pydantic_object=BlogPost)
format_instructions= parser.get_format_instructions()

prompt = ChatPromptTemplate.from_template(
    "create a creative and engaging blog post on the topic '{topic}'. "
    "the blog post should we in this format:{format_instructions}\n"
)

Chain= prompt | model | parser
async def generate_blog_post(topic):
    """
     An asynchronous function to generate a blog post based on the given topic.
   """
    print(f"--- Starting blog post generation for: {topic}... ---")
    response = await Chain.ainvoke({"topic": topic, "format_instructions": format_instructions})
    print(f"\n✅ === Blog Post Generated for: {topic} === ✅\n{response}\n")

async def main():
    """
    The main function that runs the blog post generation concurrently.
    """
    start_time = time.time()
    
    # Run multiple blog post generations concurrently
    await asyncio.gather(
        generate_blog_post("The Future of AI in Business"),
        generate_blog_post("How to Leverage AI for Marketing"),
        generate_blog_post("AI and the Evolution of Customer Service")
    )
    
    end_time = time.time()
    print(f"--- All blog posts generated in {end_time - start_time:.2f} seconds ---")
if __name__ == "__main__":
    # This command tells Python to run our main asynchronous function.
    asyncio.run(main())
reponse= Chain.invoke({"topic": "The Future of AI in Business", "format_instructions": format_instructions})
print(f"Blog Post Title: {reponse.title}")
print(f"Blog Post Content: {reponse.content}")
print(f"Blog Post Conclusion: {reponse.conclusion}")
print(f"Blog Post Tags: {', '.join(reponse.tags)}")
