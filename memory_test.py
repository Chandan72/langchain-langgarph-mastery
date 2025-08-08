from langchain.memory import ConversationBufferMemory

# First, let's create an instance of the memory
memory = ConversationBufferMemory()

# Let's simulate a conversation and save it to the memory
# The user says "Hi!"
memory.save_context({"input": "Hi there!"}, {"output": "Hello! How can I help you today?"})

# The user says "I want to learn about AI."
memory.save_context({"input": "I want to learn about AI."}, {"output": "That's a great topic! AI is a broad field. Where should we start?"})

# Now, let's see what's inside the memory
print(memory.load_memory_variables({}))