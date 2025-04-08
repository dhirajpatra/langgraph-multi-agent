import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from graph.agent_graph import AgentGraph
from langchain_core.messages import AIMessage

# Load env variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Initialize Ollama LLM
llm = OllamaFunctions(
    model="deepseek-r1:1.5b",
    base_url="http://ollama_server:11434",
    temperature=0,
    format="json"
)

# Initialize agent
agent = AgentGraph(llm)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class InputMessage(BaseModel):
    text: str

class OutputMessage(BaseModel):
    reply: str

# Chat endpoint
@app.post("/chat", response_model=OutputMessage)
def chat(input_msg: InputMessage):
    try:
        result = agent.run(input_msg.text)
        msg = result["messages"][-1]
        print(f"************************* Response: {msg}***********************")
        
        # Handle tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_responses = []
            for tool_call in msg.tool_calls:
                tool_responses.append(
                    f"Tool call: {tool_call['name']} with args {tool_call['args']}"
                )
            return {"reply": "\n".join(tool_responses)}
        
        # Handle regular content
        if isinstance(msg.content, str):
            reply = msg.content
        elif isinstance(msg.content, dict):
            reply = msg.content.get("value", json.dumps(msg.content))
        elif isinstance(msg.content, list):
            reply = "\n".join(str(item) for item in msg.content)
        else:
            reply = str(msg.content)

        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Other endpoints
@app.get("/")
def health():
    return {"status": "agent running"}

@app.get("/info")
def info():
    return {"status": "ok"}

@app.post("/runs/batch")
def run_batch():
    return {"status": "OK", "message": "Batch endpoint placeholder"}

# Run with: python app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
