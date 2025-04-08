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
    # model="deepseek-r1:1.5b",
    model="llama3.2:1b",
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
        msg = agent.run(input_msg.text)["messages"][-1]
        logging.info(f"Response: {msg}")

        if getattr(msg, "tool_calls", None):
            return {"reply": "\n".join(
                f"Tool call: {t['name']} with args {t['args']}" for t in msg.tool_calls
            )}

        return {"reply": str(msg.content)}
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
