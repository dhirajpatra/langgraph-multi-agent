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

# Update the agent creation functions to ensure proper message handling
def create_agent(model, tools, name, prompt):
    # Create the prompt template with proper placeholder
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"{prompt}\nYou are the {name}."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")  # For tool interaction history
    ])
    
    # Create the agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        name=name,
        prompt=prompt_template,
    )

    # Sync version
    def sync_agent(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]

        agent_scratchpad = [msg for msg in messages if getattr(msg, "tool_call_id", None)]

        inputs = {
            "input": last_message.content,
            "agent_scratchpad": agent_scratchpad
        }

        response = agent.invoke(inputs)

        output = {"messages": [response]}

        if hasattr(response, "tool_calls") and response.tool_calls:
            output["tool_calls"] = response.tool_calls

        return output
    
    # Wrap the agent to handle LangGraph state
    async def async_agent(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]

        # Extract tool call messages for agent_scratchpad
        agent_scratchpad = [msg for msg in messages if getattr(msg, "tool_call_id", None)]

        inputs = {
            "input": last_message.content,
            "agent_scratchpad": agent_scratchpad
        }

        response = await agent.ainvoke(inputs)

        output = {"messages": [response]}

        if hasattr(response, "tool_calls") and response.tool_calls:
            output["tool_calls"] = response.tool_calls

        return output
    
    # Return both versions
    return {
        "sync": sync_agent,
        "async": async_agent
    }

weather_agent = create_agent(
    model=llm,
    tools=[WeatherTool.weather_tool],
    name="weather_agent",
    prompt="You are a helpful assistant with access to one tool: weather_agent. Call this tool to find the weather of a location. Do not engage in any other conversation or tasks."
)

calendar_agent = create_agent(
    model=llm,
    tools=[CalendarTool.calendar_tool],
    name="calendar_agent",
    prompt="You are a helpful assistant with access to one tool: calendar_agent. Call this tool to check the calendar for meetings. Do not engage in any other conversation or tasks."
)

retriever_agent = create_agent(
    model=llm,
    tools=[RetrieverTool.retriever_tool],
    name="retriever_agent",
    prompt=(
        "You are a helpful RAG-based retriever agent with access to one tool: retriever_agent. "
        "Call this tool to search and return information about Lilian Weng's blog posts on LLM agents, "
        "prompt engineering, and adversarial attacks on LLMs."
    )
)

# Define the routing agent
def route_agent(state: MessagesState) -> dict:
    logging.info(f"Routing agent invoked. {state}")
    if state.get("messages") and len(state["messages"]) > 0:
        last_message = state["messages"][-1]
    else:
        logging.warning("No messages found in state.")
        return {"error": "No messages available"}

    if isinstance(last_message, HumanMessage):
        last_user_input = last_message.content.lower()
    else:
        last_user_input = str(last_message).lower()
    
    branches = []
    if any(x in last_user_input for x in ["weather", "temperature", "forecast", "rain", "sunny"]):
        branches.append("weather")
    if any(x in last_user_input for x in ["calendar", "meeting", "schedule", "event"]):
        branches.append("calendar")
    if "lilian weng" in last_user_input or "blog" in last_user_input:
        branches.append("retriever")

    logging.info(f"User input: {last_user_input}")
    logging.info(f"Selected branches: {branches}")

    if not branches:
        return {"next": "end"}

    return {"next": branches[0], "next_agents": branches[1:] if len(branches) > 1 else []}

def conditional_step(state: MessagesState) -> dict:
    if "next_agents" in state and state["next_agents"]:
        next_agent = state["next_agents"][0]
        return {"next": next_agent, "next_agents": state["next_agents"][1:]}
    return {"next": "end"}

# Graph Definition
graph = StateGraph(MessagesState)

# Add all nodes first
graph.add_node("weather", weather_agent["async"])
graph.add_node("calendar", calendar_agent["async"])
graph.add_node("retriever", retriever_agent["async"])
graph.add_node("router", route_agent)
graph.add_node("end", lambda state: state)  # Add end node properly

# Set entry point
graph.set_entry_point("router")

# Add conditional edges from router
graph.add_conditional_edges(
    "router",
    lambda state: state.get("next", "end"),
    {
        "weather": "weather",
        "calendar": "calendar",
        "retriever": "retriever",
        "end": "end"  # Now points to our properly defined end node
    }
)

# Add conditional edges from each agent
for node in ["weather", "calendar", "retriever"]:
    graph.add_conditional_edges(
        node,
        lambda state: state.get("next", "end"),
        {"end": "end"}  # Points to our end node
    )

# Add final edge from end to END
graph.add_edge("end", END)

# Compile
checkpointer = InMemorySaver()
store = InMemoryStore()
compiled_agent = graph.compile(
    checkpointer=checkpointer,
    store=store,
)

# Node Call
async def async_llm_call(content: str) -> str:
    """LLM decides whether to call a tool or not"""
    session_id = str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}")
    try:
        response = await compiled_agent.ainvoke(
            {
                "input": content,
                "agent_scratchpad": "",  # Initialize this if your prompt expects it
            },
            config={"configurable": {"thread_id": session_id}}
        )
        logging.info(f"Response: {response}")
        outputs = []
        if hasattr(response, "content"):
            outputs.append(response.content)
        elif hasattr(response, "output"):
            outputs.append(str(response.output))

        return "\n".join(outputs) if outputs else "No response generated"
    except Exception as e:
        logging.error(f"Error in llm_call: {str(e)}")
        raise

def llm_call(content: str) -> str:
    import asyncio
    return asyncio.run(async_llm_call(content))

