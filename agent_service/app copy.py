import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from graph.agent_graph import AgentGraph
from langchain_core.messages import AIMessage

# Load env variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Initialize Ollama LLM
llm = ChatOllama(
    # model="deepseek-r1:1.5b",
    model="llama3.1:8b",
    base_url="http://ollama_server:11434",
    temperature=0,
    format="json",
    streaming=True, # Ollama does not support streaming yet
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
        messages = result.get("messages", [])

        if not messages:
            logging.warning("No messages returned from agent.")
            raise HTTPException(status_code=500, detail="Agent did not return any response.")

        msg = messages[-1]
        logging.info(f"Response content: {msg.content}")
        logging.debug(f"Full message object: {msg}")

        if getattr(msg, "tool_calls", None):
            return {"reply": "\n".join(
                f"Tool call: {t['name']} with args {t['args']}" for t in msg.tool_calls
            )}

        return {"reply": str(msg.content) if msg.content else "No content returned."}

    except ValueError as ve:
        if "No data received from Ollama stream" in str(ve):
            logging.error("Ollama failed to return streaming data.")
            raise HTTPException(status_code=502, detail="Ollama stream returned no data.")
        logging.exception("ValueError during chat.")
        raise HTTPException(status_code=500, detail=str(ve))

    except Exception as e:
        logging.exception("Chat processing failed.")
        raise HTTPException(status_code=500, detail="Unexpected server error.")

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
