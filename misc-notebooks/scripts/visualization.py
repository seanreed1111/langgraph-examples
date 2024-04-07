#!/usr/bin/env python
# coding: utf-8

# # Visualization
# 
# This notebook walks through how to visualize the graphs you create. For this example we will use a prebuilt graph, but this works with ANY graphs.

# ## Set up the chat model and tools
# 
# Here we will define the chat model and tools that we want to use.
# Importantly, this model MUST support OpenAI function calling.

# In[1]:


from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import chat_agent_executor
from langchain_core.messages import HumanMessage


# In[2]:


tools = [TavilySearchResults(max_results=1)]
model = ChatOpenAI()


# ## Create executor
# 
# We can now use the high level interface to create the executor

# In[3]:


app = chat_agent_executor.create_function_calling_executor(model, tools)


# ## Ascii
# 
# We can easily visualize this graph in ascii

# In[4]:


app.get_graph().print_ascii()


# ## PNG
# 
# We can also visualize this as a `.png`
# 
# Note that this requires having graphviz installed

# In[ ]:


# !pip install pygraphviz 


# In[6]:


from IPython.display import Image


# In[7]:


Image(app.get_graph().draw_png())


# In[ ]:




