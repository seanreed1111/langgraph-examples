#!/usr/bin/env python
# coding: utf-8

# # Prompt Generator
# 
# In this example we will create a chat bot that helps a user generate a prompt.
# It will first collect requirements from the user, and then will generate the prompt (and refine it based on user input).
# These are split into two separate states, and the LLM decides when to transition between them.
# 
# A graphical representation of the system can be found below.
# 
# ![](imgs/prompt-generator.png)

# ## Gather information
# 
# First, let's define the part of the graph that will gather user requirements. This will be an LLM call with a specific system message. It will have access to a tool that it can call when it is ready to generate the prompt.

# In[6]:


from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List


# In[7]:


template = """Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discerne this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discerne all the information, call the relevant tool"""


# In[8]:


llm = ChatOpenAI(temperature=0)


# In[9]:


def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages


class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""
    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]


# In[10]:


llm_with_tool = llm.bind_tools([PromptInstructions])

chain = get_messages_info | llm_with_tool


# ## Generate Prompt
# 
# We now set up the state that will generate the prompt.
# This will require a separate system message, as well as a function to filter out all message PRIOR to the tool invocation (as that is when the previous state decided it was time to generate the prompt

# In[11]:


# Helper function for determining if tool was called
def _is_tool_call(msg):
    return hasattr(msg, "additional_kwargs") and 'tool_calls' in msg.additional_kwargs


# In[12]:


# New system prompt
prompt_system = """Based on the following requirements, write a good prompt template:

{reqs}"""

# Function to get the messages for the prompt
# Will only get messages AFTER the tool call
def get_prompt_messages(messages):
    tool_call = None
    other_msgs = []
    for m in messages:
        if _is_tool_call(m):
            tool_call = m.additional_kwargs['tool_calls'][0]['function']['arguments']
        elif tool_call is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs
    


# In[ ]:


prompt_gen_chain = get_prompt_messages | llm


# ## Define the state logic
# 
# This is the logic for what state the chatbot is in.
# If the last message is a tool call, then we are in the state where the "prompt creator" (`prompt`) should respond.
# Otherwise, if the last message is not a HumanMessage, then we know the human should respond next and so we are in the `END` state.
# If the last message is a HumanMessage, then if there was a tool call previously we are in the `prompt` state.
# Otherwise, we are in the "info gathering" (`info`) state.

# In[14]:


def get_state(messages):
    if _is_tool_call(messages[-1]):
        return "prompt"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    for m in messages:
        if _is_tool_call(m):
            return "prompt"
    return "info"


# ## Create the graph
# 
# We can now the create the graph.
# We will use a SqliteSaver to persist conversation history.

# In[2]:


from langgraph.graph import MessageGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")

nodes = {k:k for k in ['info', 'prompt', END]}
workflow = MessageGraph()
workflow.add_node("info", chain)
workflow.add_node("prompt", prompt_gen_chain)
workflow.add_conditional_edges("info", get_state, nodes)
workflow.add_conditional_edges("prompt", get_state, nodes)
workflow.set_entry_point("info")
graph = workflow.compile(checkpointer=memory)


# ## Use the graph
# 
# We can now use the created chatbot.

# In[5]:


import uuid
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": str(uuid.uuid4())}}
while True:
    user = input('User (q/Q to quit): ')
    if user in {'q', 'Q'}:
        print('AI: Byebye')
        break
    for output in graph.stream([HumanMessage(content=user)], config=config):
        if "__end__" in output:
            continue
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")


# In[ ]:




