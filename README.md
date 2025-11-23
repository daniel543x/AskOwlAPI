# AskOwl

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
![Status: WIP](https://img.shields.io/badge/Status-WIP-yellow.svg)

**AskOwl** is a self-hosted, LLM-powered search engine designed to give you full control over your information discovery process.

> **Note:** This project is currently in its early development phase. Functionality is subject to change.


## Why AskOwl?

Existing AI tools often offer limited personalization options. AskOwl was born from the need to create a search engine that adapts to you, not the other way around. Want programming answers based on your preferred tech stack? Or maybe you're researching saltwater aquariums and want to filter out everything else? With AskOwl, you can define these contexts and receive precisely tailored results.

## Key Features

*   **Self-hosted:** Full control over your data and infrastructure. Nothing leaves your network.
*   **Three Search Modes:** Choose the level of autonomy that best suits your needs—from a simple query to a fully autonomous research task.
*   **Flexible AI Providers:** We support multiple providers, including Ollama (local models), Groq, OpenRouter, Cohere, and others, with a focus on open-source models.
*   **Easy Setup:** Get AskOwl running in just a few steps using Podman or Docker.

## Quick Start

Prerequisites:
*   [Podman](https://podman.io/) or [Docker](https://www.docker.com/)
*   `podman-compose` or `docker-compose`

### Step 1: Clone the repository
```bash
git clone https://github.com/your-username/askowl.git
cd askowl
```

### Step 2: Configure environment variables
Copy the template file and fill in the required variables.
```bash
mv .env.template .env
```
Open the `.env` file in a text editor and set the API keys for your chosen cloud providers (e.g., `GROQ_API_KEY`). Required variables are marked with the `# Required` comment.

### Step 3: Run with Compose
```bash
podman-compose up -d
# or if you're using Docker:
# docker-compose up -d
```

### Step 4: Use AskOwl
By default, the API will be available at `http://localhost:8000`. You can test it using a tool like `curl` or Postman by sending requests to the appropriate endpoints.

## Search Modes

AskOwl offers three distinct modes of operation, powered by Large Language Models (LLMs).

### 1. Search
This is a direct, enhanced search mode. You provide a query, and AskOwl:
1.  Performs a standard web search (via SearXNG by default).
2.  Passes the results to an LLM.
3.  You receive a synthesized, coherent answer based on the information found.

### 2. Research
For when you need a deep understanding of a topic. This mode automates the research process:
1.  Based on your query, the LLM generates a series of sub-questions.
2.  It then uses the **Search** mode for each of these sub-questions.
3.  It gathers all the answers and compiles them into a comprehensive report on your topic.

### 3. Autonomous
The most advanced mode, giving the AI full freedom to act. Simply state your goal, and the LLM:
1.  Decides for itself how to best solve the task.
2.  Can modify the original query, formulate new ones, and use available tools repeatedly.
3.  Acts like an agent that plans and executes the steps needed to achieve the objective.
*(Requires a model with "tool calling" capabilities).*

## Configuration

The main configuration file is `.env`.

*   **LLM Provider:** By default, AskOwl tries to connect to a local **Ollama** instance. If that fails, you must configure a cloud provider by setting the appropriate API key (e.g., `OPENROUTER_API_KEY`). You can change the provider by modifying the `AI_PROVIDER` variable.
*   **Search Engine:** The default search engine is **SearXNG**. Future versions will allow an administrator to define multiple search engines.

## For Developers

### Local Development Environment
To run the project for development purposes:
1.  Clone the repository.
2.  Create a Python virtual environment: `python -m venv venv` and activate it.
3.  Install the `uv` package manager: `pip install uv`.
4.  Install dependencies: `uv pip install -r requirements.txt`.
5.  Run the application from the `src` directory: `python src/main.py`.

### Tests
The project uses `pytest` for testing. To run them, execute:
```bash
pytest
```

### Contributing
Guidelines for reporting bugs and suggesting features will be added in the future.

## Tech Stack

*   **Backend:** Python, FastAPI
*   **Frontend:** Planned in Vue
*   **Containerization:** Podman/Docker

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Copyright

Copyright 2025 The AskOwl Contributors
```
