# agent_service/graph/agent_graph.py
import logging
import uuid
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph_supervisor import create_supervisor
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from tools.weather_tool import WeatherTool
from tools.calendar_tool import CalendarTool
from tools.retriever_tool import RetrieverTool

logging.basicConfig(level=logging.INFO)

model = "llama3.1:8b"

llm = ChatOllama(
    model=model,
    base_url="http://ollama_server:11434",
    temperature=0.0,
)

weather_agent = create_react_agent(
    model=llm,
    tools=[WeatherTool.weather_tool],
    name="weather_agent",
    prompt="You are a helpful weather assistant with access to one tool: `weather_tool`. Call this tool to find the weather of a location. Do not engage in any other conversation or tasks."
)

calendar_agent = create_react_agent(
    model=llm,
    tools=[CalendarTool.calendar_tool],
    name="calendar_agent",
    prompt="You are a helpful calendar assistant with access to one tool: `calendar_tool`. Call this tool to check the calendar for meetings. Do not engage in any other conversation or tasks."
)

retriever_agent = create_react_agent(
    model=llm,
    tools=[RetrieverTool.retriever_tool],
    name="retriever_agent",
    prompt=(
        "You are a helpful RAG-based retriever agent with access to one tool: `retriever_tool`. "
        "Call this tool to search and return information about Lilian Weng's blog posts on LLM agents, "
        "prompt engineering, and adversarial attacks on LLMs."
        "Do not engage in any other conversation or tasks."
    )
)

# Supervisor agent
supervisor = create_supervisor(
    [weather_agent, calendar_agent],
    model=llm,
    prompt = (
        "You are a supervisor managing three specialized agents: `weather_agent`, `calendar_agent` and `retriever_agent`.\n"
        "\n"
        "Your responsibilities:\n"
        "1. If the user's query includes weather-related information (e.g., temperature, rain, forecast, location), use `weather_agent`.\n"
        "2. If the user's query includes calendar-related information (e.g., meetings, events, schedules), use `calendar_agent`.\n"
        "3. If the user's query includes information about blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs, use `retriever_agent`.\n"
        "4. If the query includes **all**, you MUST call all agents one after another and combine their responses clearly.\n"
        "\n"
        "Guidelines:\n"
        "- When using `weather_agent`, check the `status` field in the response:\n"
        "  - If `report`, return the weather report.\n"
        "  - If `error_message`, inform the user of the error.\n"
        "- When using `calendar_agent`, check the `status` field in the response:\n"
        "  - If `report`, return the meeting details.\n"
        "  - If `error_message`, inform the user of the error.\n"
        "- When using `retriever_agent`, check the `status` field in the response:\n"
        "  - If `report`, return the blog post details.\n"
        "  - If `error_message`, inform the user of the error.\n"
        "\n"
        "Do NOT ignore parts of a multi-intent query. Always answer each intent by calling the appropriate agent.\n"
        "If the query doesn't relate to weather or calendar, state clearly that you cannot help with that.\n"
    ),
    # output_mode="full_history",
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
