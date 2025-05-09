# Use a more specific and official base image
FROM python:3.11-slim

# Install CA certificates and system dependencies in one layer to keep the image size smaller
RUN apt-get update && apt-get install -y \
    ca-certificates \
    build-essential \
    libpq-dev \
    wget \
    gcc \
    g++ \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create and set working directory
WORKDIR /app

# Copy and install dependencies in a single step
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create the .streamlit directory and add the config.toml file
RUN mkdir -p /root/.streamlit
COPY config.toml /root/.streamlit/config.toml

# Copy the rest of the application code
COPY . .

# Expose the desired ports
EXPOSE 8501
# EXPOSE 11434

# Define the entrypoint for the Streamlit app
ENTRYPOINT ["sh", "-c", "streamlit run chat/app.py --server.port=8501 --server.address=0.0.0.0"]