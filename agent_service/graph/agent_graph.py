# agent_service/graph/agent_graph.py
from __future__ import annotations
import logging
import uuid
import os
from typing import List, Literal, Annotated, Any
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.messages import ToolMessage
from tools.weather_tool import weather_tool
from tools.calendar_tool import calendar_tool
from tools.retriever_tool import retriever_tool

from openai import OpenAI
from agents import Agent, HandoffInputData, Runner, function_tool, handoff, trace
from agents.extensions import handoff_filters

logging.basicConfig(level=logging.INFO)
load_dotenv()

model = os.getenv("REMOTE_MODEL")
base_url = os.getenv("REMOTE_BASE_URL")
token = os.getenv("GITHUB_TOKEN")

temperature = 0.9
top_p = 0.1
max_tokens = 500

llm = OpenAI(
    base_url=base_url,
    api_key=token,
)

class FinalResult(BaseModel):
    result: dict[str, Any]

# Specialized agents
weather_agent = Agent(
    model=llm,
    tools=[weather_tool],
    name="weather_agent",
    instructions="""
    You are a specialized weather assistant. Always respond using the `weather_tool` only. 
    Do not generate any additional explanation or response. Return only the tool output.
    """,
    output_type=FinalResult,
)

calendar_agent = Agent(
    model=llm,
    tools=[calendar_tool],
    name="calendar_agent",
    instructions="""
    You are a calendar assistant. Always respond using the `calendar_tool` only. 
    Do not generate any additional response. Return only the tool output.
    """,
    output_type=FinalResult,
)

retriever_agent = Agent(
    model=llm,
    tools=[retriever_tool],
    name="retriever_agent",
    instructions="""
    You are a retriever assistant. Always respond using the `retriever_tool` only.
    Do not generate any additional response. Return only the tool output.    
    """,
    output_type=FinalResult,
)

def create_custom_handoff_tool(*, agent_name: str, name: str | None, description: str | None) -> BaseTool:
    @tool(name, description=description)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        task_description = state.get("task_description", "")
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
        )
        messages = state.get("messages", [])
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={
                "messages": messages + [tool_message],
                "active_agent": agent_name,
                "task_description": task_description,
            },
        )

    return handoff_to_agent

handoff_to_weather = create_custom_handoff_tool(
    agent_name="weather_agent",
    name="handoff_to_weather",
    description="Handoff control to the weather agent.",
)
handoff_to_calendar = create_custom_handoff_tool(
    agent_name="calendar_agent",
    name="handoff_to_calendar",
    description="Handoff control to the calendar agent.",
)
handoff_to_retriever = create_custom_handoff_tool(
    agent_name="retriever_agent",
    name="handoff_to_retriever",
    description="Handoff control to the retriever agent.",
)

supervisor = Agent(
    model=llm,
    handoffs=[weather_agent, calendar_agent, retriever_agent],
    name="supervisor",
    instructions="""
    You are a supervisor agent managing `weather_agent`, `calendar_agent`, and `retriever_agent`.

    Your job is to:
    1. Analyze the user's query and identify the individual sub-questions by splitting on '?'.
    2. For each distinct sub-question:
        a. Determine the most appropriate agent (only one per sub-question)
        b. Issue exactly one `goto` command for that sub-question
    3. Agent routing rules:
        - route only one sub-question to one agent
        - Weather-related → `weather_agent`
        - Calendar/meetings/schedule → `calendar_agent`
        - Blogs/documents/prompt/LLM topics → `retriever_agent`
    4. Important rules:
        - Never route the same sub-question to multiple agents
        - Never issue multiple `goto` commands for the same sub-question
        - DO NOT answer any sub-questions yourself - only route to agents
    5. For multiple sub-questions:
        - Process each one independently
        - Ensure each gets exactly one `goto` command to one agent
        - After all agents have responded with their results 
        - put together the final output from all agents and respond with FINISH.
    """,
)

store = InMemoryStore()
checkpointer = InMemorySaver()

def llm_call(content: str) -> str:
    session_id = str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}")
    try:
        response = Runner.run_sync(
            starting_agent=supervisor,
            handoff_filters=[
                handoff_filters.handoff_to_weather,
                handoff_filters.handoff_to_calendar,
                handoff_filters.handoff_to_retriever,
            ],
            input={"messages": [{"role": "user", "content": content}]},
            config={"configurable": {"thread_id": session_id}}
        )
        
        if not response.get("messages"):
            raise ValueError("Empty response from agent")

        for m in response["messages"]:
            logging.info(m.pretty_print())
        final_output = response["messages"][-1].content if response.get("messages") else "No output"
        return final_output
    except Exception as e:
        logging.error(f"Error in llm_call: {str(e)}")
        raise
