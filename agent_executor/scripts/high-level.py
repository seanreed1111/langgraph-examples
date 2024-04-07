#!/usr/bin/env python
# coding: utf-8

# # Agent Executor
# 
# This notebook walks through an example creating an agent executor to work with an existing LangChain agent.
# This is useful for getting started quickly.
# However, it is highly likely you will want to customize the logic - for information on that, check out the other examples in this folder.

# ## Setup
# 
# First we need to install the packages required

# In[ ]:


get_ipython().system('pip install --quiet -U langchain langchain_openai tavily-python')


# Next, we need to set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)

# In[ ]:


import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API Key:")


# Optionally, we can set API key for [LangSmith tracing](https://smith.langchain.com/), which will give us best-in-class observability.

# In[ ]:


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key:")


# ## Set up LangChain Agent
# 
# First, will set up our LangChain Agent. 
# See documentation [here](https://python.langchain.com/docs/modules/agents/) for more information on what these agents are and how to think about them

# In[2]:


from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults


# In[3]:


tools = [TavilySearchResults(max_results=1)]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

# Construct the OpenAI Functions agent
agent_runnable = create_openai_functions_agent(llm, tools, prompt)


# ## Create agent executor
# 
# Now we will use the high level method to create the agent executor

# In[4]:


from langgraph.prebuilt import create_agent_executor


# In[5]:


app = create_agent_executor(agent_runnable, tools)


# In[6]:


inputs = {"input": "what is the weather in sf", "chat_history": []}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")


# In[18]:


s["__end__"]["agent_outcome"]


# ## Custom Input Schema
# 
# By default, the `create_agent_executor` assumes that the input will be a dictionary with two keys: `input` and `chat_history`. 
# If this is not the case, you can easily customize the input schema.
# You should do this, by defining a schema as a TypedDict.
# 
# For this example, we will create a new agent that expects `question` and `language` as inputs.

# ### Create New Agent

# In[7]:


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Respond to the user question: {question}. Answer in this language: {language}",
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
agent_runnable = create_openai_functions_agent(llm, tools, prompt)


# ### Define Input Schema

# In[8]:


from typing import TypedDict


# In[9]:


class InputSchema(TypedDict):
    question: str
    language: str


# ### Create new agent executor

# In[10]:


app = create_agent_executor(agent_runnable, tools, input_schema=InputSchema)


# In[11]:


inputs = {"question": "what is the weather in sf", "language": "italian"}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")


# In[25]:


s["__end__"]["agent_outcome"]


# In[ ]:




