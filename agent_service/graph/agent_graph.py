# agent_service/graph/agent_graph.py
import logging
import json
from datetime import date
from typing import Sequence, Annotated, TypedDict

from langgraph.graph import StateGraph, END, add_messages, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable
from langchain_core.messages import (
    SystemMessage, AIMessage, HumanMessage, BaseMessage
)
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

from tools.weather_tool import WeatherTool
from tools.calendar_tool import CalendarTool

logging.basicConfig(level=logging.INFO)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class AgentGraph:
    def __init__(self, llm: Runnable):
        self._build_graph(llm)
        self.bound_llm = llm
        today = date.today().isoformat()
        self.system_prompt = (
            f"You are a helpful assistant. You can check the weather and calendar for meeting."
            "You can also ask me to continue or end the conversation."
        )

    def run(self, user_input: str):
        return self.workflow.invoke({"messages": [HumanMessage(content=user_input)]})

    def _call_model(self, state: AgentState):
        today = date.today().isoformat()
        logging.info(f"*****************  Today's date: {today}")
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
                    meetings = CalendarTool.calendar_tool()
                    return {"messages": [AIMessage(content=meetings)]}
            except Exception as tool_parse_error:
                logging.warning(f"Tool parsing error: {tool_parse_error}")

            return {"messages": [response]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"Error: {str(e)}")]}


    def _should_continue(self, state: AgentState):
        last_msg = state["messages"][-1]
        return "continue" if getattr(last_msg, "tool_calls", None) else "end"

    def _build_graph(self, llm: Runnable):
        tools = [WeatherTool.weather_tool, CalendarTool.calendar_tool]
        agent_node = create_react_agent(
            llm,
            tools
        )
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.set_entry_point("agent")
        workflow.set_finish_point("agent")
        self.workflow = workflow.compile()

    def run(self, user_input: str):
        return self.workflow.invoke({"messages": [HumanMessage(content=user_input)]})
