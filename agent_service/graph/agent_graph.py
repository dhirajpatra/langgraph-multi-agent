# agent_service/graph/agent_graph.py
import logging
from langchain_ollama import ChatOllama
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState
from tools.weather_tool import WeatherTool
from tools.calendar_tool import CalendarTool

logging.basicConfig(level=logging.INFO)
# model = "qwen2.5:7b-instruct"
model = "llama3.1:8b"

# Initialize LLM
llm = ChatOllama(
    model=model,
    base_url="http://ollama_server:11434",
    temperature=0.0,
)

# Define agents
weather_agent = create_react_agent(
    model=llm,
    tools=[WeatherTool.weather_tool],
    name="weather_agent",
    prompt="You are a helpful assistant with access to one tool: weather_agent. Call this tool to find the weather of a location. Do not engage in any other conversation or tasks."
)

calendar_agent = create_react_agent(
    model=llm,
    tools=[CalendarTool.calendar_tool],
    name="calendar_agent",
    prompt="You are a helpful assistant with access to one tool: calendar_agent. Call this tool to check the calendar for meetings. Do not engage in any other conversation or tasks."
)

# Supervisor agent
supervisor = create_supervisor(
    [weather_agent, calendar_agent],
    model=llm,
    prompt = (
        "You are a supervisor managing two specialized agents: `weather_agent` and `calendar_agent`.\n"
        "\n"
        "Your responsibilities:\n"
        "1. If the user's query includes weather-related information (e.g., temperature, rain, forecast, location), use `weather_agent`.\n"
        "2. If the user's query includes calendar-related information (e.g., meetings, events, schedules), use `calendar_agent`.\n"
        "3. If the query includes **both**, you MUST call both agents one after another and combine their responses clearly.\n"
        "\n"
        "Guidelines:\n"
        "- When using `weather_agent`, check the `status` field in the response:\n"
        "  - If `report`, return the weather report.\n"
        "  - If `error_message`, inform the user of the error.\n"
        "- When using `calendar_agent`, check the `status` field in the response:\n"
        "  - If `report`, return the meeting details.\n"
        "  - If `error_message`, inform the user of the error.\n"
        "\n"
        "Do NOT ignore parts of a multi-intent query. Always answer each intent by calling the appropriate agent.\n"
        "If the query doesn't relate to weather or calendar, state clearly that you cannot help with that.\n"
    ),
    output_mode="full_history",
    # output_mode="last_message",
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

        for m in response["messages"]:
            logging.info(m.pretty_print())
        return response
    except Exception as e:
        logging.error(f"Error in llm_call: {str(e)}")
        raise
