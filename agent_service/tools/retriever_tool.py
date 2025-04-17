# agent_service/tools/retriever_tool.py

import logging
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
from tools.retriever import retriever  # Assumes retriever is defined and imported correctly

logging.basicConfig(level=logging.INFO)


class RetrieverToolArgs(BaseModel):
    query: str = Field(description="The search query to look for in the blog posts.")


# Global retriever tool instance
retriever_tool_instance = create_retriever_tool(
    retriever,
    name="retriever_tool_instance",
    description=(
        "Search and return information about Lilian Weng's blog posts "
        "on LLM agents, prompt engineering, and adversarial attacks on LLMs."
    ),
)


@tool(args_schema=RetrieverToolArgs, description="Search blog posts based on user query.")
def retriever_tool(query: str, tool_call_id: str | None = None) -> dict:
    """
    Tool to search blog posts based on user query.
    """
    logging.info(f"[retriever_tool] Received query: {query}")
    try:
        result = retriever_tool_instance.invoke(query)
        logging.info(f"[retriever_tool] Result: {result}")
        return {"status": "success", "report": f"Retrieved from Lilian Blog: {result}"}
    except Exception as e:
        logging.error(f"[retriever_tool] Error: {e}")
        return {"status": "error", "report": f"Error retrieving blog posts: {str(e)}"}
