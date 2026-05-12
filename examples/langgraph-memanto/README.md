# Memanto + LangGraph: Long-Term Memory Integration

This example demonstrates how to use **Memanto** as a long-term memory layer for **LangGraph** agents. It showcases a **Hybrid Memory Architecture** where LangGraph handles short-term conversation state and Memanto provides a persistent, cross-session brain.

## 🎥 Demo

![Cross-Session Recall Demo](demo/cross_session_recall_demo.gif)

> **Session 1** stores research findings via `memanto_remember` → **Session 2** (fresh thread, empty LangGraph state) retrieves them via `memanto_recall`. The agent remembers across sessions!

## 🧠 Why Memanto for LangGraph?

LangGraph's built-in persistence (Checkpointers) is excellent for maintaining state within a thread. However, agents often need to:
- **Share knowledge** across disjointed threads.
- **Recall facts** from weeks or months ago without bloating the current context window.
- **Synthesize insights** from a massive history of interactions via RAG.

Memanto solves this by providing a semantic memory layer that sits outside the standard graph state.

## 🚀 Key Features

- **Hybrid Memory**: Uses LangGraph's `MemorySaver` for thread-local state and Memanto for global agent persistence.
- **Structured Memories**: Stores facts, decisions, and preferences with specific semantic types.
- **Cross-Session Recall**: The agent "remembers" information in a new thread even after the short-term state is cleared.
- **Grounded Answers**: Uses Memanto's RAG capabilities to answer questions based on its entire life history.

## 🛠️ Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ../../
   ```

2. **Configure Environment**:
   Create a `.env` file in this directory with your API keys:
   ```env
   MOORCHEH_API_KEY=your_memanto_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## 🏃 Running the Example

The `main.py` script demonstrates the transition between sessions:

1. **Session 1 (`thread_id="session-1"`)**: The agent researches and stores benefits of Memanto.
2. **Session 2 (`thread_id="session-2"`)**: A completely new graph instance successfully retrieves those benefits from its persistent brain.

```bash
python main.py
```

## 📂 Project Structure

- `memanto_langgraph/tools.py`: Wraps Memanto SDK into LangChain tools with strict typing.
- `graph.py`: Defines the LangGraph state machine with integrated checkpointer.
- `main.py`: Orchestrates the multi-session demonstration.
- `demo/`: Demo GIF and animation source files.
