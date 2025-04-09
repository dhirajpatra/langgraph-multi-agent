# agent_service/graph/agent_graph.py
import logging
from langchain_ollama import ChatOllama
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState
from tools.weather_tool import WeatherTool
from tools.calendar_tool import CalendarTool

logging.basicConfig(level=logging.INFO)

# Initialize LLM
llm = ChatOllama(
    model="llama3.1:8b",
    base_url="http://ollama_server:11434",
    temperature=0,
    format="json",
    timeout=300.0,
)

# Define agents
weather_agent = create_react_agent(
    model=llm,
    tools=[WeatherTool.weather_tool],
    name="weather_agent",
    prompt="You are a helpful assistant. You can check the weather of a location."
)

calendar_agent = create_react_agent(
    model=llm,
    tools=[CalendarTool.calendar_tool],
    name="calendar_agent",
    prompt="You are a helpful assistant. You can check the calendar for meetings."
)

# Supervisor agent
supervisor = create_supervisor(
    [weather_agent, calendar_agent],
    model=llm,
    prompt=(
        "You are a team supervisor managing a weather assistant and a calendar assistant.\n"
        "For weather-related queries (e.g., temperature, forecast, location weather), use weather_agent.\n"
        "For calendar or meeting-related queries (e.g., upcoming events, meetings on a date), use calendar_agent.\n"
    ),
    output_mode="full_history",
)

# Compile the graph
compiled_agent = supervisor.compile()

# Nodes
def llm_call(content: str) -> str:
    """LLM decides whether to call a tool or not"""
    try:
        response = compiled_agent.invoke(
            {"messages": [{"role": "user", "content": content}]}
        )
        if not response.get("messages"):
            raise ValueError("Empty response from agent")
        return response
    except Exception as e:
        logging.error(f"Error in llm_call: {str(e)}")
        raise
