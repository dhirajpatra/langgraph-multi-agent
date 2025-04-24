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
    max_tokens=1000,
    top_p=0.9,  # Increased from 0.1 to allow more diversity
    top_k=40,   # Added to help with tool selection
)

weather_prompt = """You are a weather assistant. Follow these rules STRICTLY:
1. ONLY answer weather-related questions
2. Use ONLY the weather_tool
3. Return ONLY factual weather data
4. For non-weather questions, respond: "I only handle weather queries"
5. After completing your task, ALWAYS transfer back to supervisor
"""

calendar_prompt = """You are a calendar assistant. Follow these rules STRICTLY:
1. ONLY answer about meetings/events
2. Use ONLY the calendar_tool
3. Return ONLY calendar information
4. For non-calendar questions, respond: "I only handle calendar queries"
5. After completing your task, ALWAYS transfer back to supervisor
"""
retriever_prompt = """You are a document retrieval assistant. Follow these rules STRICTLY:
1. ONLY answer about blogs/documents/llm/prompts
2. Use ONLY the retriever_tool
3. Return ONLY blog information
4. For non-blog, non-documents, non-llm, non-prompt questions, respond: "I only handle blog queries"
5. After completing your task, ALWAYS transfer back to supervisor
"""

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
        # Get the last human message
        messages = state.get("messages", [])
        last_human_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
        )
        
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={
                "messages": messages + [tool_message],
                "active_agent": agent_name,
                "task_description": last_human_msg.content if last_human_msg else "",
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

agents = [weather_agent, calendar_agent, retriever_agent]

# Supervisor agent using standard LangGraph supervisor
supervisor = create_supervisor(
    agents,
    model=llm,
    prompt="""
    You are a supervisor agent managing `weather_agent`, `calendar_agent`, and `retriever_agent`.
    
    Responsibilities:
    1. Analyze the user's query and split into distinct sub-questions by '?'
    2. For each sub-question:
        - Route to exactly one specialized agent
        - Weather → weather_agent
        - Calendar/meetings → calendar_agent
        - Documents/knowledge → retriever_agent
    3. After all agents respond:
        - Use compile_responses to gather outputs
        - Use finalize_response to format the answer
        - Respond with FINISH
    
    Strict Rules:
    - NEVER answer questions directly
    - NEVER route same question to multiple agents
    - ALWAYS wait for all agents to complete
    - ALWAYS use the response formatting tools
    """,
    output_mode="last_message",
    # output_mode="full_history",
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
    """Enhanced LLM execution wrapper"""
    session_id = str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}")
    
    try:
        response = compiled_agent.invoke(
            {
                "messages": [HumanMessage(content=content)],
                "task_description": content  # Store the original query
            },
            config={
                "configurable": {
                    "thread_id": session_id,
                    "recursion_limit": 10  # Prevent infinite loops
                }
            }
        )

        # Extract the final response
        final_message = next(
            (msg for msg in reversed(response["messages"]) 
             if isinstance(msg, AIMessage) and msg.name == "supervisor"),
            None
        )
        
        return final_message.content if final_message else "No response generated"
        
    except Exception as e:
        logging.error(f"Error in llm_call: {str(e)}")
        return f"Error processing your request: {str(e)}"
