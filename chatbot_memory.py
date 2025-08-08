from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
model= ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.4)
memory= ConversationBufferMemory()
conversation= ConversationChain(
    llm=model,
    memory=memory,
    verbose=True
)

while True:
    user_input= input("You: ")
    if user_input.lower()== "exit":
        break
    response= conversation.predict(input=user_input)
    print(f"AI: {response}")