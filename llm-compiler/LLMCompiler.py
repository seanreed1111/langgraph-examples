#!/usr/bin/env python
# coding: utf-8

# # LLMCompiler
#
# This notebook shows how to implement [LLMCompiler, by Kim, et. al](https://arxiv.org/abs/2312.04511) in LangGraph.
#
# LLMCompiler is an agent architecture designed to **speed up** the execution of agentic tasks by eagerly-executed tasks within a DAG. It also saves costs on redundant token usage by reducing the number of calls to the LLM. Below is an overview of its computational graph:
#
# ![LLMCompiler Graph](./img/llm-compiler.png)
#
# It has 3 main components:
#
# 1. Planner: stream a DAG of tasks.
# 2. Task Fetching Unit: schedules and executes the tasks as soon as they are executable
# 3. Joiner: Responds to the user or triggers a second plan
#
#
# This notebook walks through each component and shows how to wire them together using LangGraph. The end result will leave a trace [like the following](https://smith.langchain.com/public/218c2677-c719-4147-b0e9-7bc3b5bb2623/r).
#
#
# **First,** install the dependencies, and set up LangSmith for tracing to more easily debug and observe the agent.
import datetime
import itertools
import json
import os
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

from langchain import hub
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import (
    chain as as_runnable,
)
from langchain_core.tools import BaseTool#, tool
# from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, MessageGraph
from loguru import logger
from math_tools import get_math_tool
from output_parser import LLMCompilerPlanParser, Task
from typing_extensions import TypedDict

if "src" not in sys.path:
    sys.path.append("../src")  # needed to get the azure config imports to run
from config import LOCAL_CONFIG_DIR, run_azure_config

RECURSION_LIMIT = 80

run_azure_config(LOCAL_CONFIG_DIR)

now = str(datetime.date.today())
temp_dir_path = tempfile.mkdtemp(prefix=now)
# log_file_name = "quickstart.ipynb.log"  # only for notebooks
log_file_name = Path(__file__).stem + ".log"  # only for scripts
log_file_path = (
    Path(temp_dir_path) / log_file_name
)  # appends automatically if file exists

# logger.info(f"created {temp_dir_path=}")
log_level = "DEBUG"
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
logger.add(
    sys.stderr,
    level=log_level,
    format=log_format,
    colorize=True,
    backtrace=True,
    diagnose=True,
)
logger.add(
    log_file_path,
    level=log_level,
    format=log_format,
    colorize=False,
    backtrace=True,
    diagnose=True,
)


use_gpt_4 = input("Use GPT 4? (y or n)").lower()
if use_gpt_4 == "y":
    model_name = os.getenv("MODEL_NAME_GPT")
    deployment_name = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT")
    logger.info("using gpt 4")
else:
    model_name = os.getenv("MODEL_NAME_GPT35")
    deployment_name = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35")
    logger.info("using gpt 3.5")

os.environ["LANGCHAIN_PROJECT"] = (
    f"{Path(__file__).stem}-langgraph-examples-dir-{model_name}"
)


# ## Part 1: Tools
#
# We'll first define the tools for the agent to use in our demo. We'll give it the class search engine + calculator combo.
#
# If you don't want to sign up for tavily, you can replace it with the free [DuckDuckGo](https://python.langchain.com/docs/integrations/tools/ddg).

# Imported from the https://github.com/langchain-ai/langgraph/tree/main/examples/plan-and-execute repo

# EXAMPLE 1
llm = AzureChatOpenAI(
    temperature=0.05,
    streaming=True,
    model_name=model_name,
    azure_deployment=deployment_name,
    azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    request_timeout=120,
    verbose=False,
)

calculate = get_math_tool(llm)
search = TavilySearchResults(
    max_results=5,
    description='tavily_search_results_json(query="the search query") - a search engine.',
)

tools = [search, calculate]

calculate.invoke(
    {
        "problem": "What's the temp of Denver + 5?",
        "context": ["The temperature of Denver is 32 degrees"],
    }
)

# # Part 2: Planner
#
#
# Largely adapted from [the original source code](https://github.com/SqueezeAILab/LLMCompiler/blob/main/src/llm_compiler/output_parser.py), the planner  accepts the input question and generates a task list to execute.
#
# If it is provided with a previous plan, it is instructed to re-plan, which is useful if, upon completion of the first batch of tasks, the agent must take more actions.
#
# The code below composes constructs the prompt template for the planner and composes it with LLM and output parser, defined in [output_parser.py](./output_parser.py). The output parser processes a task list in the following form:
#
# ```plaintext
# 1. tool_1(arg1="arg1", arg2=3.5, ...)
# Thought: I then want to find out Y by using tool_2
# 2. tool_2(arg1="", arg2="${1}")'
# 3. join()<END_OF_PLAN>"
# ```
#
# The "Thought" lines are optional. The `${#}` placeholders are variables. These are used to route tool (task) outputs to other tools.

prompt = hub.pull("wfh/llm-compiler")
print(prompt.pretty_print())


@logger.catch
def create_planner(
    llm: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate
):
    tool_descriptions = "\n".join(
        f"{i+1}. {included_tool.description}\n"
        for i, included_tool in enumerate(
            tools
        )  # +1 to offset the 0 starting index, we want it count normally from 1.
    )
    planner_prompt = base_prompt.partial(
        replan="",
        num_tools=len(tools)
        + 1,  # Add one because we're adding the join() tool at the end.
        tool_descriptions=tool_descriptions,
    )
    replanner_prompt = base_prompt.partial(
        replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
        "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
        'You MUST use these information to create the next plan under "Current Plan".\n'
        ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
        " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
        " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    def should_replan(state: list):
        # Context is passed as a system message
        return isinstance(state[-1], SystemMessage)

    def wrap_messages(state: list):
        return {"messages": state}

    def wrap_and_get_last_index(state: list):
        next_task = 0
        for message in state[::-1]:
            if isinstance(message, FunctionMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
        return {"messages": state}

    return (
        RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            wrap_messages | planner_prompt,
        )
        | llm
        | LLMCompilerPlanParser(tools=tools)
    )


# This is the primary "agent" in our application
planner = create_planner(llm, tools, prompt)

example_question = "What's the temperature in Denver raised to the 2rd power?"

for task in planner.stream([HumanMessage(content=example_question)]):
    print(task["tool"], task["args"])
    print("---")


# ## 3. Task Fetching Unit
#
# This component schedules the tasks. It receives a stream of tools of the following format:
#
# ```typescript
# {
#     tool: BaseTool,
#     dependencies: number[],
# }
# ```
#
#
# The basic idea is to begin executing tools as soon as their dependencies are met. This is done through multi-threading. We will combine the task fetching unit and exector below:
#
# ![diagram](./img/diagram.png)
@logger.catch
def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    # Get all previous tool responses
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    return results


class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Task]


@logger.catch
def _execute_task(task, observations, config):
    tool_to_use = task["tool"]
    if isinstance(tool_to_use, str):
        return tool_to_use
    args = task["args"]
    try:
        if isinstance(args, str):
            resolved_args = _resolve_arg(args, observations)
        elif isinstance(args, dict):
            resolved_args = {
                key: _resolve_arg(val, observations) for key, val in args.items()
            }
        else:
            # This will likely fail
            resolved_args = args
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}.)"
            f" Args could not be resolved. Error: {repr(e)}"
        )
    try:
        return tool_to_use.invoke(resolved_args, config)
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}."
            + f" Args resolved to {resolved_args}. Error: {repr(e)})"
        )


@logger.catch
def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    # $1 or ${1} -> 1
    ID_PATTERN = r"\$\{?(\d+)\}?"

    def replace_match(match):
        # If the string is ${123}, match.group(0) is ${123}, and match.group(1) is 123.

        # Return the match group, in this case the index, from the string. This is the index
        # number we get back.
        idx = int(match.group(1))
        return str(observations.get(idx, match.group(0)))

    # For dependencies on other tasks
    if isinstance(arg, str):
        return re.sub(ID_PATTERN, replace_match, arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


@as_runnable
@logger.catch
def schedule_task(task_inputs, config):
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = _execute_task(task, observations, config)
    except Exception:

        observation = traceback.format_exception()  # repr(e) +
    observations[task["idx"]] = observation


@logger.catch
def schedule_pending_task(
    task: Task, observations: Dict[int, Any], retry_after: float = 0.2
):
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            # Dependencies not yet satisfied
            time.sleep(retry_after)
            continue
        schedule_task.invoke({"task": task, "observations": observations})
        break


@as_runnable
@logger.catch
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    """Group the tasks into a DAG schedule."""
    # For streaming, we are making a few simplifying assumption:
    # 1. The LLM does not create cyclic dependencies
    # 2. That the LLM will not generate tasks with future deps
    # If this ceases to be a good assumption, you can either
    # adjust to do a proper topological sort (not-stream)
    # or use a more complicated data structure
    tasks = scheduler_input["tasks"]
    args_for_tasks = {}
    messages = scheduler_input["messages"]
    # If we are re-planning, we may have calls that depend on previous
    # plans. Start with those.
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    # ^^ We assume each task inserts a different key above to
    # avoid race conditions...
    futures = []
    retry_after = 0.25  # Retry every quarter second
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            deps = task["dependencies"]
            task_names[task["idx"]] = (
                task["tool"] if isinstance(task["tool"], str) else task["tool"].name
            )
            args_for_tasks[task["idx"]] = task["args"]
            if (
                # Depends on other tasks
                deps and (any([dep not in observations for dep in deps]))
            ):
                futures.append(
                    executor.submit(
                        schedule_pending_task, task, observations, retry_after
                    )
                )
            else:
                # No deps or all deps satisfied
                # can schedule now
                schedule_task.invoke(dict(task=task, observations=observations))
                # futures.append(executor.submit(schedule_task.invoke dict(task=task, observations=observations)))

        # All tasks have been submitted or enqueued
        # Wait for them to complete
        wait(futures)
    # Convert observations to new tool messages to add to the state
    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }
    tool_messages = [
        FunctionMessage(
            name=name, content=str(obs), additional_kwargs={"idx": k, "args": task_args}
        )
        for k, (name, task_args, obs) in new_observations.items()
    ]
    return tool_messages


@as_runnable
@logger.catch
def plan_and_schedule(messages: List[BaseMessage], config):
    tasks = planner.stream(messages, config)
    # Begin executing the planner immediately
    try:
        tasks = itertools.chain([next(tasks)], tasks)
    except StopIteration:
        # Handle the case where tasks is empty.
        tasks = iter([])
    scheduled_tasks = schedule_tasks.invoke(
        {
            "messages": messages,
            "tasks": tasks,
        },
        config,
    )
    return scheduled_tasks


# #### Example Plan
#
# We still haven't introduced any cycles in our computation graph, so this is all easily expressed in LCEL.

tool_messages = plan_and_schedule.invoke([HumanMessage(content=example_question)])


# tool_messages


# ## "Joiner"
#
# So now we have the planning and initial execution done. We need a component to process these outputs and either:
#
# 1. Respond with the correct answer.
# 2. Loop with a new plan.
#
# The paper refers to this as the "joiner". It's another LLM call. We are using function calling to improve parsing reliability.


class FinalResponse(BaseModel):
    """The final response/answer."""

    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]


joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(
    examples=""
)  # You can optionally add examples
# llm = ChatOpenAI(model="gpt-4-turbo-preview")

runnable = create_structured_output_runnable(JoinOutputs, llm, joiner_prompt)


# We will select only the most recent messages in the state, and format the output to be more useful for
# the planner, should the agent need to loop.


@logger.catch
def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        return response + [
            SystemMessage(
                content=f"Context from last attempt: {decision.action.feedback}"
            )
        ]
    else:
        return response + [AIMessage(content=decision.action.response)]


@logger.catch
def select_recent_messages(messages: list) -> dict:
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break
    return {"messages": selected[::-1]}


joiner = select_recent_messages | runnable | _parse_joiner_output

input_messages = [HumanMessage(content=example_question)] + tool_messages


joiner.invoke(input_messages)


# ## 5. Compose using LangGraph
#
# We'll define the agent as a stateful graph, with the main nodes being:
#
# 1. Plan and execute (the DAG from the first step above)
# 2. Join: determine if we should finish or replan
# 3. Recontextualize: update the graph state based on the output from the joiner

graph_builder = MessageGraph()
graph_builder.add_node("plan_and_schedule", plan_and_schedule)
graph_builder.add_node("join", joiner)
graph_builder.add_edge("plan_and_schedule", "join")


def should_continue(state: List[BaseMessage]):
    if isinstance(state[-1], AIMessage):
        return END
    return "plan_and_schedule"


graph_builder.add_conditional_edges(
    start_key="join",
    # Next, we pass in the function that will determine which node is called next.
    condition=should_continue,
)
graph_builder.set_entry_point("plan_and_schedule")
chain = graph_builder.compile()


# #### Multi-hop question

for step in chain.stream(
    [
        HumanMessage(
            content="What was the GDP of Australia in 2020 in in Billions of US Dollars? \
     What was the GDP of Canada in 2020 in Billions of US Dollars? Which country has the larger GDP? \
        What is the sum of the GDPs of the two countries, in Billions of US Dollars"
        )
    ],
    {
        "recursion_limit": RECURSION_LIMIT,
    },
):
    print(step)
    print("---")


# #### Multi-hop question
#
# This question requires that the agent perform multiple searches.

steps = chain.stream(
    [
        HumanMessage(
            content="Who is the oldest person alive? and how much older is that person than the average human lifespan \
                in the country where they were born ?"
        )
    ],
    {
        "recursion_limit": RECURSION_LIMIT,
    },
)
for i, step in enumerate(steps):
    print(i, step)
    print("---")


# Next Question
for i,step in enumerate(chain.stream(
    [
        HumanMessage(
            content="What's (3*3245) + 8? What's 32/4.23? What's the sum of those two values?"
        )
    ],
    {
        "recursion_limit": RECURSION_LIMIT,
    },
)):
    # print(f"\nstep{i}:{str(step)}")
    print("---")
    logger.info(f"\nstep{i}:{str(step)}")


# ## Conclusion
#
# Congrats on building your first LLMCompiler agent! 
# I'll leave you with some known limitations to the implementation above:
#
'''
# 1. The planner output parsing format is fragile if your function requires more than 1 or 2 arguments. 
    # We could make it more robust by using streaming tool calling.
# 2. Variable substitution is fragile in the example above. 
    # It could be made more robust by using a fine-tuned model 
    # and a more robust syntax (using e.g., Lark or a tool calling schema)
# 3. The state can grow quite long if you require multiple re-planning runs. 
    # To handle, you could add a message compressor once you go above a certain token limit.
'''