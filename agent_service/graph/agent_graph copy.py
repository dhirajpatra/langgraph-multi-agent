import logging
from datetime import date
from typing import Sequence, Annotated, TypedDict
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, add_messages, StateGraph
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from tools.weather_tool import WeatherTool
from tools.calendar_tool import CalendarTool


logging.basicConfig(level=logging.INFO)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class AgentGraph:
    def __init__(self, llm: Runnable):
        self.llm = llm
        self._build_graph()

    def _build_graph(self):
        # Agents with names (required for supervisor)
        weather_agent = create_react_agent(
            model=self.llm,
            tools=[WeatherTool.weather_tool],
            name="weather_agent",
            prompt="You are a helpful assistant. You can check the weather of a location."
        )
        calendar_agent = create_react_agent(
            model=self.llm,
            tools=[CalendarTool.calendar_tool],
            name="calendar_agent",
            prompt="You are a helpful assistant. You can check the calendar for meetings."
        )

        # Supervisor with routing logic
        supervisor_prompt = (
            "You are a team supervisor managing a weather assistant and a calendar assistant.\n"
            "For weather-related queries (e.g., temperature, forecast, location weather), use weather_agent.\n"
            "For calendar or meeting-related queries (e.g., upcoming events, meetings on a date), use calendar_agent.\n"
        )

        supervisor = create_supervisor(
            [weather_agent, calendar_agent],
            model=self.llm,
            prompt=supervisor_prompt,
            # output_mode="last_message"
            output_mode="full_history",
        )
        self.workflow = supervisor.compile()

    def run(self, user_input: str):
        result = self.workflow.invoke({
            "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
        })
        # Print the result
        for m in result["messages"]:
            logging.info(">>", m.pretty_print())
        return result

