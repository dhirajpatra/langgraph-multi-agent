# File: agent_service/tools/retriever_tool.py
import logging
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from tools.retriever import retriever

logging.basicConfig(level=logging.INFO)

# Create retriever tool instance globally
retriever_tool_instance = create_retriever_tool(
    retriever,
    name="retrieve_blog_posts",
    description=(
        "Search and return information about Lilian Weng blog posts "
        "on LLM agents, prompt engineering, and adversarial attacks on LLMs."
    ),
)

@tool("retrieve_blog_posts")
def retriever_tool(query: str) -> dict:
    """
    Tool to search blog posts based on user query.
    """
    logging.info(f"[RetrieverTool] Query received: {query}")
    try:
        result = retriever_tool_instance.invoke(query)
        logging.info(f"[RetrieverTool] Retrieved result: {result}")
        return {
            "status": "success",
            "report": f"Retrieved from Lilian Blog: {result}"
        }
    except Exception as e:
        logging.error(f"[RetrieverTool] Error occurred: {e}")
        return {
            "status": "error",
            "report": f"Error occurred while retrieving blog posts: {str(e)}"
        }
