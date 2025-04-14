# agent_service/graph/agent_graph.py
import logging
import uuid
from typing import List, Literal, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.graph import MessageGraph, END
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
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

# Custom Handoff Tool
def create_custom_handoff_tool(*, agent_name: str, name: str | None, description: str | None) -> BaseTool:

    @tool(name, description=description)
    def handoff_to_agent(
        task_description: Annotated[str, "Detailed description of what the next agent should do, including all of the relevant context."],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
        )
        messages = state["messages"]
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

# Create agents
weather_agent = create_react_agent(
    model=llm,
    tools=[weather_tool, handoff_to_calendar, handoff_to_retriever],
    name="weather_agent",
    prompt="""
    You are a specialized weather assistant. If the user asks about weather, use `weather_tool`.
    If the user asks about calendar or meeting, use `handoff_to_calendar`.
    If the user asks about blog, use `handoff_to_retriever`.
    Only answer weather related questions.
    """
)

calendar_agent = create_react_agent(
    model=llm,
    tools=[calendar_tool, handoff_to_weather, handoff_to_retriever],
    name="calendar_agent",
    prompt="""
    You are a dedicated calendar assistant. Use `calendar_tool` to check calendar entries.
    If the user asks about weather, use `handoff_to_weather`.
    If the user asks about blog, use `handoff_to_retriever`.
    Only answer calendar related questions.
    """
)

retriever_agent = create_react_agent(
    model=llm,
    tools=[retriever_tool, handoff_to_weather, handoff_to_calendar],
    name="retriever_agent",
    prompt="""
    You are an expert retriever agent, focused on information from a RAG database.
    Use `retriever_tool` for information about Lilian Weng's blog posts.
    If the user asks about weather, use `handoff_to_weather`.
    If the user asks about calendar or meeting, use `handoff_to_calendar`.
    Only answer questions related to blog.
    """
)

# Supervisor agent
supervisor = create_supervisor(
    [weather_agent, calendar_agent, retriever_agent],
    model=llm,
    prompt="""
    You are a supervisor that managing `weather_agent`, `calendar_agent` and `retriever_agent`.
    Combine the responses into a clear and concise summary.
    """,
    output_mode="last_message",
)

# Initialize memory components
store = InMemoryStore()
checkpointer = InMemorySaver()

# Compile the graph
compiled_agent = supervisor.compile(
    checkpointer=checkpointer,
    store=store,
)

# Node Call (Synchronous)
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
        final_output = response.get("messages")[-1].content if "messages" in response else response
        return final_output
    except Exception as e:
        logging.error(f"Error in llm_call: {str(e)}")
        raise