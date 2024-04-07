import datetime
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, MessageGraph
from loguru import logger

sys.path.append("../src")  # needed to get the azure config imports to run
from config import LOCAL_CONFIG_DIR, run_azure_config

run_azure_config(LOCAL_CONFIG_DIR)

now = str(datetime.date.today())
temp_dir_path = tempfile.mkdtemp(prefix=now)
# log_file_name = "quickstart.ipynb.log"  # only for notebooks
log_file_name = Path(__file__).stem + ".log"  # only for scripts
log_file_path = (
    Path(temp_dir_path) / log_file_name
)  # appends automatically if file exists

logger.info(f"created {temp_dir_path=}")
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

os.environ["LANGCHAIN_PROJECT"] = f"{Path(__file__)}-langgraph-examples-dir"
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

graph = MessageGraph()


def invoke_model(state: List[BaseMessage]):
    return llm.invoke(state)


graph.add_node("oracle", invoke_model)
graph.add_edge("oracle", END)

graph.set_entry_point("oracle")

runnable = graph.compile()


logger.info(str(runnable.invoke(HumanMessage("What is 1 + 1?"))))


inp = input("Ready to begin example #2?")

##############################
### EXAMPLE 2
##############################

@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number


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
model_with_tools = llm.bind(tools=[convert_to_openai_tool(multiply)])

graph = MessageGraph()


def invoke_model(state: List[BaseMessage]):
    return model_with_tools.invoke(state)


graph.add_node("oracle", invoke_model)


def invoke_tool(state: List[BaseMessage]):
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    multiply_call = None

    for tool_call in tool_calls:
        if tool_call.get("function").get("name") == "multiply":
            multiply_call = tool_call

    if multiply_call is None:
        raise Exception("No adder input found.")

    res = multiply.invoke(json.loads(multiply_call.get("function").get("arguments")))

    return ToolMessage(tool_call_id=multiply_call.get("id"), content=res)


graph.add_node("multiply", invoke_tool)

graph.add_edge("multiply", END)

graph.set_entry_point("oracle")


def router(state: List[BaseMessage]):
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    if len(tool_calls):
        return "multiply"
    else:
        return "end"


graph.add_conditional_edges(
    "oracle",
    router,
    {
        "multiply": "multiply",
        "end": END,
    },
)


runnable = graph.compile()
logger.info(str(runnable.invoke(HumanMessage("What is 12 * 13?"))))
