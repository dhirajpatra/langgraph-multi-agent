# agent_service/graph/agent_graph.py
import logging
import json
from datetime import date
from typing import Sequence, Annotated, TypedDict

from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable
from langchain_core.messages import (
    SystemMessage, AIMessage, HumanMessage, BaseMessage
)

from tools.weather_tool import WeatherTool
from tools.calendar_tool import CalendarTool

today = date.today().isoformat()
logging.basicConfig(level=logging.INFO)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class AgentGraph:
    def __init__(self, llm: Runnable):
        self.system_prompt = f"You are a helpful assistant. Today's date is {today}."
        self.bound_llm = llm.bind_tools(tools=[WeatherTool.weather_tool, CalendarTool.calendar_tool])
        self.workflow = self._build_graph()

    def _call_model(self, state: AgentState):
        messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
        try:
            response = self.bound_llm.invoke(messages)
            logging.info(f"Response: {response}")
            
            # Simulate tool logic from JSON if present
            try:
                parsed = json.loads(response.content.strip())
                # Check for simulated tool trigger
                if "weather_tool" in parsed and parsed["weather_tool"]:
                    location = parsed.get("location", "unknown")
                    unit = parsed.get("unit", "celsius")
                    weather = WeatherTool.weather_tool(location, unit)
                    return {"messages": [AIMessage(content=weather)]}
                elif "calendar_tool" in parsed and parsed["calendar_tool"]:
                    date = parsed.get("date", "")
                    meetings = CalendarTool.calendar_tool(date)
                    return {"messages": [AIMessage(content=meetings)]}
            except Exception as tool_parse_error:
                logging.warning(f"Tool parsing error: {tool_parse_error}")

            return {"messages": [response]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"Error: {str(e)}")]}


    def _should_continue(self, state: AgentState):
        last_msg = state["messages"][-1]
        return "continue" if getattr(last_msg, "tool_calls", None) else "end"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._call_model)
        graph.set_entry_point("agent")
        graph.set_finish_point("agent")
        return graph.compile()

    def run(self, user_input: str):
        return self.workflow.invoke({"messages": [HumanMessage(content=user_input)]})
