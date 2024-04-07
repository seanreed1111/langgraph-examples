#!/usr/bin/env python
# coding: utf-8

# # Get/Update State
# 
# When running LangGraph agents, you can easily get or update the state of the agent at any point in time. This allows for several things. Firstly, it allows you to inspect the state and take actions accordingly. Second, it allows you to modify the state - this can be useful for changing or correcting potential actions.
# 
# **Note:** this requires passing in a checkpointer.

# ## Setup
# 
# First we need to install the packages required

# In[1]:


get_ipython().system('pip install --quiet -U langchain langchain_openai tavily-python')


# Next, we need to set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)

# In[2]:


import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API Key:")


# Optionally, we can set API key for [LangSmith tracing](https://smith.langchain.com/), which will give us best-in-class observability.

# In[ ]:


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key:")


# ## Set up the tools
# 
# We will first define the tools we want to use.
# For this simple example, we will use a built-in search tool via Tavily.
# However, it is really easy to create your own tools - see documentation [here](https://python.langchain.com/docs/modules/agents/tools/custom_tools) on how to do that.
# 

# In[1]:


from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]


# We can now wrap these tools in a simple ToolExecutor.
# This is a real simple class that takes in a ToolInvocation and calls that tool, returning the output.
# A ToolInvocation is any class with `tool` and `tool_input` attribute.
# 

# In[2]:


from langgraph.prebuilt import ToolExecutor

tool_executor = ToolExecutor(tools)


# ## Set up the model
# 
# Now we need to load the chat model we want to use.
# Importantly, this should satisfy two criteria:
# 
# 1. It should work with messages. We will represent all agent state in the form of messages, so it needs to be able to work well with them.
# 2. It should work with OpenAI function calling. This means it should either be an OpenAI model or a model that exposes a similar interface.
# 
# Note: these model requirements are not requirements for using LangGraph - they are just requirements for this one example.

# In[3]:


from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0)


# 
# After we've done this, we should make sure the model knows that it has these tools available to call.
# We can do this by converting the LangChain tools into the format for OpenAI function calling, and then bind them to the model class.
# 

# In[4]:


from langchain_core.utils.function_calling import convert_to_openai_function

functions = [convert_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)


# ## Define the nodes
# 
# We now need to define a few different nodes in our graph.
# In `langgraph`, a node can be either a function or a [runnable](https://python.langchain.com/docs/expression_language/).
# There are two main nodes we need for this:
# 
# 1. The agent: responsible for deciding what (if any) actions to take.
# 2. A function to invoke tools: if the agent decides to take an action, this node will then execute that action.
# 
# We will also need to define some edges.
# Some of these edges may be conditional.
# The reason they are conditional is that based on the output of a node, one of several paths may be taken.
# The path that is taken is not known until that node is run (the LLM decides).
# 
# 1. Conditional Edge: after the agent is called, we should either:
#    a. If the agent said to take an action, then the function to invoke tools should be called
#    b. If the agent said that it was finished, then it should finish
# 2. Normal Edge: after the tools are invoked, it should always go back to the agent to decide what to do next
# 
# Let's define the nodes, as well as a function to decide how what conditional edge to take.

# In[5]:


from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage, AIMessage


# Define the function that determines whether to continue or not
def should_continue(messages):
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function to execute tools
def call_tool(messages, config):
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action, config)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return function_message


# ## Define the graph
# 
# We can now put it all together and define the graph!

# In[6]:


from langgraph.graph import MessageGraph, END

# Define a new graph
workflow = MessageGraph()

# Define the two nodes we will cycle between
workflow.add_node("agent", model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")


# **Persistence**
# 
# To add in persistence, we pass in a checkpoint when compiling the graph

# In[7]:


from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")


# In[8]:


# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile(checkpointer=memory)


# ## Preview the graph

# In[9]:


from IPython.display import Image

Image(app.get_graph().draw_png())


# ## Interacting with the Agent
# 
# We can now interact with the agent. Between interactions you can get and update state.
# 

# In[10]:


from langchain_core.messages import HumanMessage

thread = {"configurable": {"thread_id": '3'}}
for event in app.stream("hi! I'm bob", thread):
    for v in event.values():
        print(v)


# See LangSmith example run here https://smith.langchain.com/public/48931f97-3a33-453a-9150-e9cef6ec8cef/r
# 
# Here you can see the "agent" node ran, and then "should_continue" returned "end" so the graph stopped execution there.

# Let's now get the current state

# In[11]:


app.get_state(thread).values


# ### Let's get it to execute a tool

# In[12]:


for event in app.stream("what is the weather in sf currently", thread):
    for v in event.values():
        print(v)


# See LangSmith example run here https://smith.langchain.com/public/12ac5112-f550-4dc2-ae7c-4b813b9f65dd/r

# We can see it planned the tool execution (ie the "agent" node), then "should_continue" edge returned "continue" so we proceeded to "action" node, which executed the tool, and then "agent" node emitted the final response, which made "should_continue" edge return "end". Let's see how we can have more control over this.

# ### Pause before tools

# If you notice below, we now will add `interrupt_before=["action"]` - this means that before any actions are taken we pause. This is a great moment to allow the user to correct and update the state! This is very useful when you want to have a human-in-the-loop to validate (and potentially change) the action to take. 

# In[16]:


app_w_interrupt = workflow.compile(checkpointer=memory, interrupt_before=["action"])


# In[14]:


thread = {"configurable": {"thread_id": '4'}}
for event in app_w_interrupt.stream("what is the weather in sf currently", thread):
    for v in event.values():
        print(v)


# See LangSmith example run here https://smith.langchain.com/public/54d37892-9ff4-48a8-8be9-bb9ff537de68/r
# This time it executed the "agent" node same as before, and you can see in the LangSmith trace that "should_continue" returned "continue", but it paused execution per our setting above.

# This is the function call the model produced

# In[17]:


current_values = app_w_interrupt.get_state(thread)
current_values.values[-1].additional_kwargs


# Let's update the search string before proceeding

# In[18]:


current_values.values[-1].additional_kwargs = {'function_call': {'arguments': '{"query":"weather in San Francisco today"}',
  'name': 'tavily_search_results_json'}}


# In[19]:


app_w_interrupt.update_state(thread, current_values.values)


# This actually produces a LangSmith run too! See it here https://smith.langchain.com/public/0f3ed712-35a3-4a34-8353-85780dc1bd05/r
# 
# This is a shorter run that allows you to inspect the edges that reacted to the state update, you can see "should_continue" returned "continue" as before, given this is still a function call.

# The current state now reflects our updated search query!

# In[22]:


app_w_interrupt.get_state(thread).values


# In[23]:


app_w_interrupt.get_state(thread).next


# If we start the agent again it will pick up from the state we updated.

# In[24]:


for event in app_w_interrupt.stream(None, thread):
    for v in event.values():
        print(v)


# See this run in LangSmith here https://smith.langchain.com/public/51c7843b-039a-4efc-bcbd-3f93ad3cbd13/r
# 
# This continues where we left off, with "action" node, followed by "agent" node, which terminates the execution.

# ## Checking history
# 
# Let's browse the history of this thread, from newest to oldest.

# In[25]:


for state in app_w_interrupt.get_state_history(thread):
    print(state)
    print('--')
    if len(state.values) == 2:
        to_replay = state


# We can go back to any of these states and restart the agent from there!

# In[26]:


to_replay.values


# In[27]:


to_replay.next


# ### Replay a past state
# 
# To replay from this place we just need to pass its config back to the agent.

# In[28]:


for event in app_w_interrupt.stream(None, to_replay.config):
    for v in event.values():
        print(v)


# See this run in LangSmith here https://smith.langchain.com/public/f26e9e1d-16df-48ae-98f7-c823d6942bf7/r
# 
# This is similar to the previous run, this time with the original search query, instead of our modified one. 

# ### Branch off a past state

# In[29]:


branch_config = app_w_interrupt.update_state(to_replay.config, AIMessage(content='All done here!', id=to_replay.values[-1].id))


# In[30]:


branch_state = app_w_interrupt.get_state(branch_config)


# In[31]:


branch_state.values


# In[32]:


branch_state.next


# You can see the snapshot was updated and now correctly reflects that there is no next step.
# 
# You can see this in LangSmith update run here https://smith.langchain.com/public/65104717-6eda-4a0f-93c1-4755c6f929ed/r
# 
# This shows the "should_continue" edge now reacting to this replaced message, and now changing the outcome to "end" which finishes the computation.

# In[ ]:




