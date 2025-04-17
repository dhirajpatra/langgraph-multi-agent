# agent_service/graph/agent_graph.py

import logging
import uuid
from typing import List, Literal, Annotated
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

model = "llama3.1:8b"

llm = ChatOllama(
    model=model,
    base_url="http://ollama_server:11434",
    temperature=0.0,
)

# Create specialized agents
weather_agent = create_react_agent(
    model=llm,
    tools=[weather_tool],
    name="weather_agent",
    prompt="""
    You are a specialized weather assistant. Always respond using the `weather_tool` only. 
    Do not generate any additional explanation or response. Return only the tool output.
    """
)

calendar_agent = create_react_agent(
    model=llm,
    tools=[calendar_tool],
    name="calendar_agent",
    prompt="""
    You are a calendar assistant. Always respond using the `calendar_tool` only. 
    Do not generate any additional response. Return only the tool output.
    """
)

retriever_agent = create_react_agent(
    model=llm,
    tools=[retriever_tool],
    name="retriever_agent",
    prompt="""
    You are a retriever assistant. Always respond using the `retriever_tool` only.
    Do not generate any additional response. Return only the tool output.    
    """
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
    1. Analyze the user's query and identify the individual sub-questions.
    2. Route each sub-question to the appropriate agent by issuing only one `goto` command per sub-question.
        - If the sub-question is about weather → route to `weather_agent`.
        - If it's about calendar, meetings, or schedule → route to `calendar_agent`.
        - If it's about blogs, documents, or technical topics → route to `retriever_agent`.
    3. If there are multiple distinct sub-questions, split questions by '?' and issue a `goto` command for each, but ensure each sub-question is routed to only one agent.
    4. Do not answer or summarize the question yourself. Only route sub-questions to the appropriate agents.
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

