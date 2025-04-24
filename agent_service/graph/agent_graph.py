# agent_service/graph/agent_graph.py

import logging
import uuid
import os
from typing import List, Literal, Annotated
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

logging.basicConfig(level=logging.INFO)
load_dotenv()

# model = "llama3.1:8b"
model = os.getenv("MODEL")
base_url = os.getenv("BASE_URL")

llm = ChatOllama(
    model=model,
    base_url=base_url,
    temperature=0.0,
    max_tokens=500,
    top_p=0.1,
)

weather_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a weather assistant. Use only `weather_tool` to get the weather info. Return only the tool output."),
    MessagesPlaceholder("messages"),
])
calendar_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a calendar assistant. Use only `calendar_tool` to get the meeting info. Return only the tool output."),
    MessagesPlaceholder("messages"),

])
retriever_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a retriever assistant. Use only `retriever_tool` to get the blog info. Return only the tool output."),
    MessagesPlaceholder("messages"),

])

# Create specialized agents
weather_agent = create_react_agent(
    model=llm,
    tools=[weather_tool],
    name="weather_agent",
    prompt=weather_prompt
)

calendar_agent = create_react_agent(
    model=llm,
    tools=[calendar_tool],
    name="calendar_agent",
    prompt=calendar_prompt
)

retriever_agent = create_react_agent(
    model=llm,
    tools=[retriever_tool],
    name="retriever_agent",
    prompt=retriever_prompt
)

# Custom Handoff Tool
def create_custom_handoff_tool(*, agent_name: str, name: str | None, description: str | None) -> BaseTool:
    @tool(name, description=description)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        # Get task description from the state (you should store it somewhere earlier in the flow)
        task_description = state.get("task_description", "")

        # Ensure the key exists in state
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
                "task_description": task_description,  # Store task_description in the state
            },
        )

    return handoff_to_agent

# Create handoff tools
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

# Supervisor agent using standard LangGraph supervisor
supervisor = create_supervisor(
    model=llm,
    agents=[weather_agent, calendar_agent, retriever_agent],
    prompt="""
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
    output_mode="last_message",
)

# Memory components
store = InMemoryStore()
checkpointer = InMemorySaver()

# Compile the agent graph
compiled_agent = supervisor.compile(
    checkpointer=checkpointer,
    store=store,
)

# Agent execution wrapper
def llm_call(content: str) -> str:
    """LLM decides whether to call a tool or not"""
    session_id = str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}")
    try:
        response = compiled_agent.invoke(
            {"messages": [{"role": "user", "content": content}]},
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
