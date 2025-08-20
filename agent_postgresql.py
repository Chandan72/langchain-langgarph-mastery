# firstly we have to install dependencies so that python can talk to sql
# pip install SQLAlchemy psycopg2-binary
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# load important variable

load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

#connect to sql database
db_url=os.getenv("DATABASE_URL")
if not db_url:
    raise ValueError("DATABASE_URL environment variable not set.")
db= SQLDatabase.from_uri(db_url)

print(f"Database dialect: {db.dialect}")
print(f" Sample tables:{db.get_table_info()}")

# create the sql agent---

agent_executor= create_sql_agent(
    llm=model,
    db=db,
    agent_type="tool-calling",
    verbose=True
)
print("--Running the sql agnet ---")

question= "number of user in block A multiply number of user in block A?"
response=agent_executor.invoke({"input": question})
print(response["output"])