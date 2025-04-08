# LangGraph Application Setup

LangGraph Ollama multi Agent application locally, and how to obtain an API key for LangSmith.

## Prerequisites

* Python 3.11 or higher
* pip (Python package installer)
* A Google Cloud Platform (GCP) account (for Gemini API access)
* A LangSmith account (for tracing and debugging)

## Installation

1.  **Clone the Repository (if applicable):**

    ```bash
    git clone <your_repository_url>
    cd <your_application_directory>
    ```

2.  **Create a Virtual Environment (recommended):**

    ```bash
    python3.11 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    Ensure your `requirements.txt` file includes the necessary packages, such as:

    ```
    langchain
    langgraph
    langchain-google-genai
    langsmith
    python-dotenv
    ```

4.  **Create an `.env` file:**

    Create a file named `.env` in the root directory of your project. This file will store your API keys and other sensitive information.

    ```
    GOOGLE_API_KEY=<your_google_api_key>
    LANGCHAIN_API_KEY=<your_langsmith_api_key>
    LANGCHAIN_TRACING_V2="true"
    LANGCHAIN_PROJECT="Your_LangGraph_Project_Name"
    ```

## Obtaining API Keys

### 1. Google Gemini API Key

1.  **Go to Google AI Studio:**
    * [Google AI Studio](https://makersuite.google.com/)
2.  Follow the instructions to create a project and obtain an API key.
3.  Alternatively, you can obtain a Google Cloud API key from the google cloud console.
    * [Google Cloud Platform](https://cloud.google.com/)
4.  Enable the Gemini API for your project.
5.  Create API credentials.
6.  Add the key to your `.env` file as `GOOGLE_API_KEY`.

### 2. LangSmith API Key

1.  **Sign up for LangSmith:**
    * [LangSmith](https://smith.langchain.com/)
2.  **Obtain your API key:**
    * After creating an account, navigate to your settings to find your API key.
3.  **Set Environment Variables:**
    * Add the API key to your `.env` file as `LANGCHAIN_API_KEY`.
    * Also add `LANGCHAIN_TRACING_V2="true"` to your .env file.
    * Set `LANGCHAIN_PROJECT` to your desired project name.

## Running the Application

1.  **Navigate to your application directory:**

    ```bash
    cd <your_application_directory>
    ```

2.  **Run the application using the `langgraph` CLI:**

    ```bash
    langgraph run --config langgraph.json

    or 

    langgraph dev
    ```

    * Ensure that your `langgraph.json` file is correctly configured to point to your LangGraph definition.

## LangGraph and LangSmith Local Development Notes.

* Using Langsmith and setting `LANGCHAIN_TRACING_V2="true"` in your `.env` file will enable tracing of your LangGraph runs on your local machine. This is very useful for debugging.
* You can review your LangGraph runs in the LangSmith UI.
* Make sure that your `.env` file is not committed to version control, as it contains sensitive information. Add `.env` to your `.gitignore` file.
* If you encounter any issues, refer to the official LangGraph and LangChain documentation.
    * [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
    * [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
    * [LangSmith Documentation](https://docs.smith.langchain.com/)


* 🐳 Running with Docker Compose
You can also run the application using Docker Compose, which will spin up both the LangGraph agent and Ollama server.

* Start the services
`docker-compose up --build`

* This will launch:

🚀 agent_service: Your LangGraph agent on http://localhost:5000

🧠 ollama_server: The local model server running on http://localhost:11434

Use this for full isolation and easy multi-service orchestration.