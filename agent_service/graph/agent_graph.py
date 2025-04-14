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
    tools=[weather_tool],
    name="weather_agent",
    prompt="""
    You are a specialized weather assistant. Your sole function is to provide accurate weather information. 
    You have access to the `weather_tool` tool. 
    When a user asks about the weather, ALWAYS use the `weather_tool` tool to retrieve the relevant data. 
    Do not generate any information from your internal knowledge. 
    If the tool returns an error, report the error message directly. 
    Format your response as: "Weather: [Tool Output or Error Message]".
    Do not engage in any other conversation or tasks.
    """
)

calendar_agent = create_react_agent(
    model=llm,
    tools=[calendar_tool],
    name="calendar_agent",
    prompt="""
    You are a dedicated calendar assistant. Your only task is to check calendar entries for meetings. 
    Use the `calendar_tool` tool to access the calendar. 
    When a user asks about meetings or calendar events, ALWAYS use the `calendar_tool` tool. 
    Do not create or invent calendar entries. 
    If the tool cannot access the calendar or finds no relevant entries, report that.
    Format your response as: "Calendar: [Tool Output or 'No entries found' or Error Message]".
    Do not engage in any other conversation or tasks.
    """
)

retriever_agent = create_react_agent(
    model=llm,
    tools=[retriever_tool],
    name="retriever_agent",
    prompt="""
    You are an expert retriever agent, focused on extracting information from a RAG database.
    You have access to the `retriever_tool` tool.
    When a user asks about Lilian Weng's blog posts on LLM agents, prompt engineering, or adversarial attacks, ALWAYS use the `retriever_tool` tool.
    Provide information ONLY from the RAG database.
    If the tool returns no results, state "No relevant information found in the database."
    If the tool returns an error, report the error message directly.
    Format your response as: "Retrieved Info: [Tool Output or 'No relevant information found' or Error Message]".
    Do not engage in any other conversation or tasks.
    """
)

# Supervisor agent
supervisor = create_supervisor(
    [weather_agent, calendar_agent, retriever_agent],
    model=llm,
    prompt="""
    You are an intelligent supervisor tasked with managing and coordinating three specialized agents: `weather_agent`, `calendar_agent`, and `retriever_agent`.

    Your core responsibilities:

    1. **Intent Recognition:** Carefully analyze the user's query to determine the primary intent(s).
    2. **Agent Routing:** Direct the query or relevant sub-queries to the appropriate agent(s) based on the identified intent(s).
    3. **Response Aggregation:** If multiple agents are called, combine their responses into a coherent and user-friendly summary.
    4. **Error Handling:** Gracefully handle any errors reported by the agents.

    Agent-Specific Guidelines:

    - **`weather_agent`:** Use this agent for queries related to weather (temperature, rain, forecast, location). Expect a response formatted as "Weather: [Tool Output or Error Message]".
    - **`calendar_agent`:** Use this agent for queries concerning calendar events, meetings, or schedules. Expect a response formatted as "Calendar: [Tool Output or 'No entries found' or Error Message]".
    - **`retriever_agent`:** Use this agent for queries about Lilian Weng's blog posts on LLM agents, prompt engineering, and adversarial attacks. Expect a response formatted as "Retrieved Info: [Tool Output or 'No relevant information found' or Error Message]".

    Workflow:

    1. **Single Intent:** If the query relates to a single agent's domain, call that agent directly.
    2. **Multiple Intents:** If the query contains multiple distinct intents, call each relevant agent sequentially, providing each agent with the specific sub-query it needs to address. Then, synthesize the responses into a single, comprehensive answer.
    3. **Unknown Intent:** If the query does not fall within the domains of the available agents, respond with "I cannot assist with that request."

    Important Rules:

    - **Always call the appropriate agent(s) to fulfill the user's request.** Do not fabricate information.
    - **Clearly label each agent's response** in the final output to avoid confusion.
    - **Handle agent errors gracefully** and inform the user if an agent fails to provide a response.
    - **Prioritize clarity and conciseness** in your responses.
    - **Maintain a conversational tone** that is helpful and informative.
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