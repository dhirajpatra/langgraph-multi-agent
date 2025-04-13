# File: agent_service/tools/retriever.py
import os
import logging
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

logging.basicConfig(level=logging.INFO)

persist_dir = "./chroma_db"
collection_name = "rag-chroma"
model = "llama3.1:8b"

embeddings = OllamaEmbeddings(model=model)
vectorstore = None

# Ensure the directory exists
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)
    logging.info(f"Created directory {persist_dir}.")
else:
    logging.info(f"Directory {persist_dir} found.")

# Try to load existing vectorstore
if os.listdir(persist_dir):
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        logging.info("Loaded existing vectorstore.")
    except Exception as e:
        logging.error(f"Error loading vectorstore: {e}")
else:
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Create without embedding_function to avoid duplicate argument error
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    vectorstore.persist()
    logging.info("Created and persisted new vectorstore.")

    # Reload with embedding function
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

retriever = vectorstore.as_retriever()
retriever.search_kwargs["k"] = 3
retriever.search_kwargs["search_type"] = "similarity"
retriever.search_kwargs["search_kwargs"] = {
    "nprobe": 10,
    "distance_metric": "cosine",
    "ef_search": 512,
    "ef_construction": 512,
    "m": 16,
    "num_partitions": 1,
    "num_subvectors": 8,
    "num_neighbors": 10,
    "num_candidates": 100,
    "num_threads": 4,
    "num_results": 10,
    "num_probes": 10,
}
