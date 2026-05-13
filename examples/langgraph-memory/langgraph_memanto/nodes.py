"""
LangGraph node definitions for the Research + Writer pipeline.

Each node is a function (state_in, state_out) that gets compiled into
the LangGraph pipeline via langgraph_memanto.graph.build_graph().
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv

from langgraph_memanto.memory_tools import (
    memanto_remember as _memanto_remember,
    memanto_recall,
    memanto_answer,
)

load_dotenv()

MOORCHEH_API_KEY = os.getenv("MOORCHEH_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm():
    """Build an OpenRouter-backed chat model."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-3.5-haiku",
        temperature=0.7,
    )


# ---------------------------------------------------------------------------
# LangChain Tool wrappers (enables actual tool calling by the LLM)
# ---------------------------------------------------------------------------

class MemantoRememberTool:
    """Wrapper to make memanto_remember bindable as a LangChain tool."""
    name = "memanto_remember"
    description = (
        "Store a structured memory in Memanto. "
        "Use this to save key findings, facts, observations from your research. "
        "Required args: memory_type, title, content, confidence, tags."
    )

    def invoke(self, tool_input: str | Dict) -> str:
        if isinstance(tool_input, str):
            data = json.loads(tool_input)
        else:
            data = tool_input
        state = {}
        result = _memanto_remember(
            state=state,
            api_key=MOORCHEH_API_KEY,
            memory_type=data.get("memory_type", "observation"),
            title=data.get("title", "")[:100],
            content=data.get("content", "")[:500],
            confidence=float(data.get("confidence", 0.85)),
            tags=data.get("tags", []),
        )
        msg = result.get("messages", [{}])[0].get("content", "")
        return msg


# Singleton tool instance
_memanto_tool = MemantoRememberTool()


# ---------------------------------------------------------------------------
# Research Agent nodes
# ---------------------------------------------------------------------------

def research_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The Research Agent analyzes a topic and stores key findings as Memanto memories.

    Uses tool calling: the LLM invokes memanto_remember as a real bound tool,
    demonstrating actual cross-session memory persistence.
    """
    topic = state.get("research_topic", "")
    agent_id = state.get("memanto_agent_id", "langgraph-default")
    llm = _get_llm()

    # Bind memanto_remember as an actual tool the LLM can call
    tools = [_memanto_tool]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = (
        f"You are a Senior Market Research Analyst specialized in '{topic}'.\n"
        f"Your job is to research this topic thoroughly and store every significant "
        f"finding as a structured Memanto memory using the memanto_remember tool.\n\n"
        f"Research approach:\n"
        f"1. Think about the key aspects of '{topic}'\n"
        f"2. Use the memanto_remember tool to store at least 3 findings\n"
        f"3. Each memory should be atomic: one fact/observation per memory\n\n"
        f"Memory types: fact, observation, decision, learning, event\n"
        f"Confidence: 1.0 for verified facts, 0.7-0.9 for observations\n"
        f"Tags: relevant keywords like ['AI', 'market', 'trends']\n\n"
        f"After storing memories, summarize what you found in 2-3 sentences."
    )

    messages = [{"role": "system", "content": system_prompt}]
    response = llm_with_tools.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    # Execute any tool calls the model made
    tool_calls = getattr(response, "tool_calls", []) or []
    tool_results = []
    findings = []
    for tc in tool_calls:
        if tc.get("name") == "memanto_remember":
            args = tc.get("args", {})
            result = _memanto_tool.invoke(args)
            tool_results.append({"role": "tool", "content": result, "name": "memanto_remember"})
            title = args.get("title", "unknown")
            findings.append(title)

    all_messages = [{"role": "assistant", "content": content}]
    all_messages.extend(tool_results)

    return {
        "messages": all_messages,
        "findings": findings,
    }


def writer_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The Writer Agent retrieves stored memories and writes an executive briefing.

    It calls memanto_recall to fetch all research memories, then uses the LLM
    to synthesize them into a written briefing.
    """
    topic = state.get("research_topic", "")
    agent_id = state.get("memanto_agent_id", "langgraph-default")
    llm = _get_llm()

    # First, recall memories
    recall_result = memanto_recall(
        state={**state, "memanto_agent_id": agent_id},
        api_key=MOORCHEH_API_KEY,
        query=f"research findings about {topic}",
        limit=10,
    )

    # Then answer a synthesis question
    answer_result = memanto_answer(
        state={**state, "memanto_agent_id": agent_id},
        api_key=MOORCHEH_API_KEY,
        question=f"Provide a detailed summary of all research findings about {topic}",
    )

    recall_content = recall_result.get("messages", [{}])[0].get("content", "")
    answer_content = answer_result.get("messages", [{}])[0].get("content", "")

    synthesis_prompt = (
        f"You are a Technical Briefing Writer.\n"
        f"Topic: {topic}\n\n"
        f"Retrieved memories:\n{recall_content}\n\n"
        f"RAG Answer:\n{answer_content}\n\n"
        f"Write a clear, data-driven executive briefing on '{topic}' "
        f"using ONLY the information from the retrieved memories above. "
        f"Do not fabricate data. Cite sources."
    )

    response = llm.invoke(synthesis_prompt)
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "messages": [
            {"role": "assistant", "content": content},
        ],
    }


def should_continue(state: Dict[str, Any]) -> Literal["research", "writer", "end"]:
    """
    Routing logic: research → writer → end.
    """
    messages = state.get("messages", [])
    if not messages:
        return "research"
    last = messages[-1]
    role = last.get("role", "")
    if role == "assistant":
        return "writer"
    return "end"
