#!/usr/bin/env python
# coding: utf-8

# # Reflexion
# 
# [Reflexion](https://arxiv.org/abs/2303.11366) by Shinn, et. al., is an architecture designed to learn through verbal feedback and self-reflection. The agent explicitly critiques its responses for tasks to generate a higher quality final response, at the expense of longer execution time.
# 
# ![reflexion diagram](./img/reflexion.png)
# 
# The paper outlines 3 main components:
# 
# 1. Actor (agent) with self-reflection
# 2. External evaluator (task-specific, e.g. code compilation steps)
# 3. Episodic memory that stores the reflections from (1).
# 
# In their code, the last two components are very task-specific, so in this notebook, you will build the _actor_ in LangGraph.
# 
# To skip to the graph definition, see the [Construct Graph section](#Construct-Graph) below.

# ## 0. Prerequisites
# 
# Install `langgraph` (for the framework), `langchain_openai` (for the LLM), and `langchain` + `tavily-python` (for the search engine).
# 
# We will use tavily search as a tool. You can get an API key [here](https://app.tavily.com/sign-in) or replace with a different tool of your choosing.

# In[1]:


# %pip install -U --quiet  langchain langgraph langchain_openai
# %pip install -U --quiet tavily-python


# In[2]:


import getpass
import os


def _set_if_undefined(var: str) -> None:
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var)


# Optional: Configure tracing to visualize and debug the agent
_set_if_undefined("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Reflexion"

_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")


# ## 1. Actor (with reflection)
# 
# The main component of Reflexion is the "actor", which is an agent that reflects on its response and re-executes to improve based on self-critique. It's main sub-components include:
# 1. Tools/tool execution
# 2. Initial responder: generate an initial response (and self-reflection)
# 3. Revisor: re-respond (and reflec) based on previous reflections
# 
# We'll first define the tool execution context.
# 
# #### Construct tools

# In[3]:


from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)


# The tools are invoked _in context_. Create a function that invokes all the requested tools.

# In[28]:


from collections import defaultdict
from typing import List

from langchain.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

# This a helper class we have that is useful for running tools
# It takes in an agent action and calls that tool and returns the result
tool_executor = ToolExecutor([tavily_tool])
# Parse the tool messages for the execution / invocation
parser = JsonOutputToolsParser(return_id=True)


def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    tool_invocation: AIMessage = state[-1]
    parsed_tool_calls = parser.invoke(tool_invocation)
    ids = []
    tool_invocations = []
    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            tool_invocations.append(
                ToolInvocation(
                    # We only have this one for now. Would want to map it
                    # if we change
                    tool="tavily_search_results_json",
                    tool_input=query,
                )
            )
            ids.append(parsed_call["id"])

    outputs = tool_executor.batch(tool_invocations)
    outputs_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        outputs_map[id_][invocation.tool_input] = output

    return [
        ToolMessage(content=json.dumps(query_outputs), tool_call_id=id_)
        for id_, query_outputs in outputs_map.items()
    ]


# #### Initial responder

# In[63]:


import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langsmith import traceable

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


llm = ChatOpenAI(model="gpt-4-turbo-preview")
initial_answer_chain = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
) | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
validator = PydanticToolsParser(tools=[AnswerQuestion])


class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    @traceable
    def respond(self, state: List[BaseMessage]):
        response = []
        for attempt in range(3):
            try:
                response = self.runnable.invoke({"messages": state})
                self.validator.invoke(response)
                return response
            except ValidationError as e:
                state = state + [HumanMessage(content=repr(e))]
        return response


# In[64]:


first_responder = ResponderWithRetries(
    runnable=initial_answer_chain, validator=validator
)


# In[65]:


example_question = "Why is reflection useful in AI?"
initial = first_responder.respond([HumanMessage(content=example_question)])


# In[ ]:


parsed = parser.invoke(initial)
parsed


# #### Revision
# 
# The second part of the actor is a revision step.

# In[67]:


revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""


# Extend the initial answer schema to include references.
# Forcing citation in the model encourages grounded responses
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )


revision_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

revisor = ResponderWithRetries(runnable=revision_chain, validator=revision_validator)


# In[68]:


import json

revised = revisor.respond(
    [
        HumanMessage(content=""),
        initial,
        ToolMessage(
            tool_call_id=initial.additional_kwargs["tool_calls"][0]["id"],
            content=json.dumps(
                tavily_tool.invoke(str(parsed[0]["args"]["search_queries"]))
            ),
        ),
    ]
)


# In[69]:


parsed = parser.invoke(revised)
parsed


# ## Construct Graph
# 
# 
# Now we can wire all our components together.

# In[71]:


from langgraph.graph import END, MessageGraph

MAX_ITERATIONS = 5
builder = MessageGraph()
builder.add_node("draft", first_responder.respond)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor.respond)
# draft -> execute_tools
builder.add_edge("draft", "execute_tools")
# execute_tools -> revise
builder.add_edge("execute_tools", "revise")

# Define looping logic:


def _get_num_iterations(state: List[BaseMessage]):
    i = 0
    for m in state[::-1]:
        if not isinstance(m, (ToolMessage, AIMessage)):
            break
        i += 1
    return i


def event_loop(state: List[BaseMessage]) -> str:
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state)
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


# revise -> execute_tools OR end
builder.add_conditional_edges("revise", event_loop)
builder.set_entry_point("draft")
graph = builder.compile()


# In[72]:


events = graph.stream(
    [HumanMessage(content="How should we handle the climate crisis?")]
)
for i, step in enumerate(events):
    node, output = next(iter(step.items()))
    print(f"## {i+1}. {node}")
    print(str(output)[:100] + " ...")
    print("---")


# In[77]:


print(parser.invoke(step[END][-1])[0]["args"]["answer"])


# ## Conclusion
# 
# Congrats on building a Reflexion actor! I'll leave you with a few observations to save you some time when choosing which parts of this agent ot adapt to your workflow:
# 1. This agent trades off execution time for quality. It explicitly forces the agent to critique and revise the output over several steps, which usually (not always) increases the response quality but takes much longer to return a final answer
# 2. The 'reflections' can be paired with additional external feedback (such as validators), to further guide the actor.
# 3. In the paper, 1 environment (AlfWorld) uses external memory. It does this by storing summaries of the reflections to an external store and using them in subsequent trials/invocations.

# In[ ]:




