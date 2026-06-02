# Memanto + LangGraph Integrations

This directory contains several out-of-the-box examples demonstrating how to integrate Memanto's persistent memory capabilities into LangGraph agents. 

All examples share the core Memanto tools defined in core/memanto_tools.py.

## Directory Setup

`ash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Memanto and OpenAI API keys
`

## Running the Examples

Because the examples share the core tools module, you should run them as Python modules from this root directory:

`ash
python -m basic_integration.demo
python -m cross_session_recall.main
python -m custom_memory_saver.run_demo
python -m research_pipeline.run_full_pipeline
`

## Available Examples

* **asic_integration/**: A minimal, drop-in example of using Memanto tools within a simple LangGraph agent.
* **cross_session_recall/**: An agent designed to remember facts and preferences across different conversation sessions.
* **esearch_pipeline/**: A multi-agent setup where one agent researches and saves facts to Memanto, and another synthesizes them.
* **custom_memory_saver/**: An advanced implementation showing how to integrate Memanto directly at the LangGraph CheckpointSaver level.
