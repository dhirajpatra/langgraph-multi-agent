# agent.py
import os
from langgraph.graph import StateGraph
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage
from typing import TypedDict, Annotated, Sequence, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_community.llms import Ollama

from fastapi.middleware.cors import CORSMiddleware


# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Ollama Llama 3 model
llm = Ollama(
    model="deepseek-r1:1.5b",
    base_url="http://ollama_server:11434",  # Ensure correct endpoint
    temperature=0,
)

# Define the state of the agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# define the tools
# these are the tools that the agent can use
@tool
def weather_tool() -> str:
    """Get the current weather."""
    return "Weather is 19°C and Partly Cloudy"

@tool
def calendar_tool() -> str:
    """Check your calendar for meetings on a specific date."""
    return "No meetings scheduled"

all_tools = [weather_tool, calendar_tool]
tool_node = ToolNode(all_tools)

# define the graph workflow
workflow = StateGraph(AgentState) # Add the AgentState here

# get the model
def _get_model():
    # model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
    model = llm
    model = model.bind_tools(all_tools) # add this line
    return model

system_prompt = "You are such a nice helpful bot"

# invoke the model
def call_model(state):
    messages = state["messages"]
    
    # add the system message
    system_message = SystemMessage(content=system_prompt)
    full_messages = [system_message] + messages
    
    model = _get_model()
    response = model.invoke(full_messages)
    
    return {"messages": [response]}

# define nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# entry point
workflow.set_entry_point("agent")

# define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # if there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # if there is, we continue
    else:
        return "continue"

# add the conditional edges 
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

# connect tools back to agent
workflow.add_edge("tools", "agent")

# compile the graph
workflow.compile()
