# agent_service/graph/agent_graph.py

import json
import logging
from datetime import date
from typing import Sequence, Annotated, TypedDict

from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    BaseMessage
)
from tools.weather_tool import WeatherTool
from tools.calendar_tool import CalendarTool

today = date.today().isoformat()
logging.basicConfig(level=logging.INFO)

# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define the graph class
class AgentGraph:
    def __init__(self, llm: Runnable):
        self.system_prompt = f"You are such a nice helpful bot. Today's date is {today}."
        self.bound_llm = llm.bind_tools(tools=[WeatherTool.weather_tool, CalendarTool.calendar_tool])
        self.workflow = self._build_graph()

    @staticmethod
    def get_tools_schema():
        return [
            {
                "name": "weather_tool",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "calendar_tool",
                "description": "Check your calendar for meetings on a specific date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
                    },
                    "required": ["date"]
                }
            }
        ]

    def _call_model(self, state: AgentState):
        system_message = SystemMessage(content=self.system_prompt)
        full_messages = [system_message] + state["messages"]
        response = self.bound_llm.invoke(full_messages)
        print(f"************************* Response: {response}***********************")
        
        # Handle tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Return the tool calls as content
            return {"messages": [AIMessage(content="", tool_calls=response.tool_calls)]}
        
        # Handle regular content
        content = response.content
        if content is None:
            content = ""
        elif isinstance(content, dict):
            content = content.get("value", json.dumps(content))
        elif not isinstance(content, str):
            content = str(content)
        
        return {"messages": [AIMessage(content=content)]}
    
    def _should_continue(self, state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "continue"
        return "end"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._call_model)
        graph.add_node("tools", ToolNode([WeatherTool.weather_tool, CalendarTool.calendar_tool]))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._should_continue, {"continue": "tools", "end": END})
        graph.add_edge("tools", "agent")
        return graph.compile()

    def run(self, user_input: str):
        return self.workflow.invoke({"messages": [HumanMessage(content=user_input)]})
