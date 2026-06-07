"""
Dynamic Preference Shift Benchmark — Memanto vs Mem0
=====================================================

Runs the "Alex" scenario through both memory libraries, measuring:
  - Token usage (tiktoken cl100k_base)
  - Retrieval latency (p95 across all operations)
  - Accuracy (Groq llama-3.3-70b judge, 8 probe questions × 3 runs each)

Usage
-----
    cp .env.example .env   # fill in your API keys
    pip install -r requirements.txt
    python benchmark.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

USER_ID = "alex-benchmark-v1"
PROBE_RUNS = 3

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SESSIONS = [
    {"session": 1, "messages": [
        "I've been really into hip-hop lately, especially J. Cole and Kendrick.",
        "For diet I'm doing strict keto — no carbs at all.",
        "I work out every morning, usually before 7am.",
    ]},
    {"session": 2, "messages": [
        "Just discovered Knxwledge, loving the lo-fi hip-hop vibes.",
        "Still on keto, lost 5kg so far.",
    ]},
    {"session": 3, "messages": [
        "Hip-hop is life. Added Freddie Gibbs to my rotation.",
        "Meal prepping keto bowls for the week.",
    ]},
    {"session": 4, "messages": [
        "Actually I've been getting into jazz lately. Miles Davis, Coltrane.",
        "I went fully vegan last week. No more keto.",
    ]},
    {"session": 5, "messages": [
        "Jazz just hits different. Bill Evans is incredible.",
        "Vegan cooking is harder than I thought but worth it.",
    ]},
    {"session": 6, "messages": [
        "Okay jazz was a phase. Now it's metal — Metallica, Tool.",
        "Still vegan but found out I have an egg allergy.",
    ]},
    {"session": 7, "messages": [
        "Tool's Fear Inoculum is a masterpiece.",
        "Egg allergy is serious, even trace amounts affect me.",
    ]},
]

PROBE_QUESTIONS = [
    # Current state questions
    {
        "query": "What music does Alex like?",
        "expected": "metal (Tool, Metallica)",
        "wrong_answers": ["hip-hop", "jazz"],
    },
    {
        "query": "What are Alex's dietary preferences or restrictions?",
        "expected": "vegan, egg allergy",
        "wrong_answers": ["keto", "no restrictions"],
    },
    {
        "query": "What time does Alex work out?",
        "expected": "morning before 7am",
        "wrong_answers": [],
    },
    # Specificity questions
    {
        "query": "What specific artists or bands does Alex listen to right now?",
        "expected": "Tool, Metallica",
        "wrong_answers": ["J. Cole", "Kendrick", "Knxwledge", "Freddie Gibbs", "Miles Davis", "Bill Evans"],
    },
    {
        "query": "Does Alex have any food allergies?",
        "expected": "yes, egg allergy",
        "wrong_answers": ["no", "none", "no allergies"],
    },
    # Safety-critical rephrasing
    {
        "query": "Is it safe to serve Alex a dish that contains eggs?",
        "expected": "no, Alex has a severe egg allergy",
        "wrong_answers": ["yes", "should be fine", "no allergy"],
    },
    # Historical recall questions
    {
        "query": "What diet did Alex follow before going vegan?",
        "expected": "keto diet",
        "wrong_answers": ["always vegan", "vegetarian", "no specific diet"],
    },
    {
        "query": "Has Alex's music taste changed over time? What genres did they go through?",
        "expected": "yes — hip-hop, then jazz, then metal",
        "wrong_answers": ["no change", "always liked metal"],
    },
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text)) if text else 0


def summarise(records: list[dict]) -> dict:
    ingest = [r for r in records if r["operation"] == "ingest"]
    recall = [r for r in records if r["operation"] == "recall"]
    all_latencies = [r["latency_ms"] for r in records]
    return {
        "total_tokens_ingested": sum(r["tokens_in"] for r in ingest),
        "total_tokens_retrieved": sum(r["tokens_out"] for r in recall),
        "p95_latency_ms": round(float(np.percentile(all_latencies, 95)), 1) if all_latencies else 0.0,
    }


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge_response(query: str, expected: str, wrong_answers: list[str], response: str) -> dict:
    if not response.strip():
        return {"score": 0, "reason": "Empty response from memory system"}

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""You are an evaluation judge for AI memory systems.

Query: {query}
Expected (current preference): {expected}
Stale/wrong answers to penalize: {', '.join(wrong_answers) if wrong_answers else 'none'}
Memory system response: {response}

Does the response correctly reflect the CURRENT/LATEST preference?
Score 1 if it returns the current state. Score 0 if it returns stale info or nothing useful.
Respond ONLY with valid JSON: {{"score": 0 or 1, "reason": "one sentence"}}"""

    try:
        chat = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        raw = chat.choices[0].message.content.strip()
        # strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as exc:
        logger.error("Judge error: %s", exc)
        return {"score": 0, "reason": f"Judge call failed: {exc}"}


def compute_accuracy(probe_scores: list[dict]) -> float:
    if not probe_scores:
        return 0.0
    return round(sum(p["score"] for p in probe_scores) / len(probe_scores), 4)


# ---------------------------------------------------------------------------
# Memanto helpers
# ---------------------------------------------------------------------------

def _memanto_client():
    from memanto.cli.client.sdk_client import SdkClient
    key = os.getenv("MOORCHEH_API_KEY")
    if not key:
        raise EnvironmentError("MOORCHEH_API_KEY is not set")
    return SdkClient(api_key=key)


def memanto_clear(client, agent_id: str):
    try:
        if hasattr(client, "clear"):
            client.clear(agent_id=agent_id)
        elif hasattr(client, "delete_all"):
            client.delete_all(agent_id=agent_id)
        else:
            logger.warning("Memanto: no clear method found — clear via dashboard if needed.")
    except Exception as exc:
        logger.warning("Memanto clear failed (non-fatal): %s", exc)


def memanto_ingest(client, agent_id: str, message: str, session: int) -> dict:
    tokens_in = count_tokens(message)
    t0 = time.perf_counter()
    try:
        result = client.remember(
            agent_id=agent_id,
            memory_type="preference",
            title=f"Session {session} preference",
            content=message,
        )
        response_text = str(result) if result else ""
    except Exception as exc:
        logger.error("Memanto ingest error (session %d): %s", session, exc)
        response_text = ""
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "library": "memanto",
        "session": session,
        "operation": "ingest",
        "tokens_in": tokens_in,
        "tokens_out": count_tokens(response_text),
        "latency_ms": round(latency_ms, 2),
    }


def memanto_recall(client, agent_id: str, query: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    try:
        result = client.recall(agent_id=agent_id, query=query, limit=3)
        # result is a dict — extract memories list
        if isinstance(result, dict):
            memories = result.get("memories") or result.get("results") or []
            if isinstance(memories, list):
                response_text = " | ".join(
                    m.get("content", str(m)) for m in memories
                )
            else:
                response_text = str(result)
        else:
            response_text = str(result) if result else ""
    except Exception as exc:
        logger.error("Memanto recall error: %s", exc)
        response_text = ""
    latency_ms = (time.perf_counter() - t0) * 1000
    return response_text, round(latency_ms, 2)


# ---------------------------------------------------------------------------
# Mem0 helpers
# ---------------------------------------------------------------------------

def _mem0_client():
    from mem0 import MemoryClient
    key = os.getenv("MEM0_API_KEY")
    if not key:
        raise EnvironmentError("MEM0_API_KEY is not set")
    return MemoryClient(api_key=key)


def mem0_clear(client, user_id: str):
    try:
        if hasattr(client, "delete_all"):
            client.delete_all(filters={"user_id": user_id})
        else:
            logger.warning("Mem0: no delete_all method — clear via dashboard if needed.")
    except Exception as exc:
        logger.warning("Mem0 clear failed (non-fatal): %s", exc)


def mem0_ingest(client, user_id: str, message: str, session: int) -> dict:
    tokens_in = count_tokens(message)
    messages = [{"role": "user", "content": message}]
    t0 = time.perf_counter()
    try:
        result = client.add(messages, user_id=user_id)
        response_text = str(result) if result else ""
    except Exception as exc:
        logger.error("Mem0 ingest error (session %d): %s", session, exc)
        response_text = ""
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "library": "mem0",
        "session": session,
        "operation": "ingest",
        "tokens_in": tokens_in,
        "tokens_out": count_tokens(response_text),
        "latency_ms": round(latency_ms, 2),
    }


def mem0_recall(client, user_id: str, query: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    try:
        results = client.search(query, filters={"user_id": user_id})
        if isinstance(results, list):
            response_text = " | ".join(
                r.get("memory", str(r)) for r in results
            )
        elif isinstance(results, dict):
            items = results.get("results", [])
            response_text = " | ".join(
                r.get("memory", str(r)) for r in items
            )
        else:
            response_text = str(results) if results else ""
    except Exception as exc:
        logger.error("Mem0 recall error: %s", exc)
        response_text = ""
    latency_ms = (time.perf_counter() - t0) * 1000
    return response_text, round(latency_ms, 2)


# ---------------------------------------------------------------------------
# Per-library runners
# ---------------------------------------------------------------------------

def run_memanto() -> dict:
    logger.info("=" * 60)
    logger.info("MEMANTO — starting benchmark")
    logger.info("=" * 60)

    try:
        client = _memanto_client()
    except EnvironmentError as exc:
        logger.error("%s — skipping Memanto", exc)
        return _empty_result("memanto")

    agent_id = USER_ID

    # Create agent if not exists, then activate
    try:
        client.create_agent(agent_id)
        logger.info("Memanto: created agent %s", agent_id)
    except Exception:
        logger.info("Memanto: agent already exists, skipping create")
    client.activate_agent(agent_id)

    memanto_clear(client, agent_id)

    raw_results: list[dict] = []
    probe_scores: list[dict] = []

    for session_data in SESSIONS:
        s = session_data["session"]
        for msg in session_data["messages"]:
            logger.info("Memanto ingest session=%d  msg=%r", s, msg[:60])
            record = memanto_ingest(client, agent_id, msg, s)
            raw_results.append(record)
            logger.info("  tokens_in=%d  latency=%.1fms", record["tokens_in"], record["latency_ms"])

    logger.info("Waiting 20 seconds for Memanto to index memories...")
    time.sleep(20)

    for probe in PROBE_QUESTIONS:
        run_texts: list[str] = []

        for run_idx in range(PROBE_RUNS):
            response_text, latency_ms = memanto_recall(client, agent_id, probe["query"])
            run_texts.append(response_text)
            raw_results.append({
                "library": "memanto",
                "session": 8,
                "operation": "recall",
                "tokens_in": count_tokens(probe["query"]),
                "tokens_out": count_tokens(response_text),
                "latency_ms": latency_ms,
            })
            logger.info("Memanto recall run=%d  query=%r  latency=%.1fms", run_idx + 1, probe["query"], latency_ms)

        best_response = run_texts[-1]
        logger.info("Memanto response: %s", best_response[:300])

        judge_result = judge_response(
            query=probe["query"],
            expected=probe["expected"],
            wrong_answers=probe["wrong_answers"],
            response=best_response,
        )
        probe_scores.append({"query": probe["query"], "expected": probe["expected"], "response": best_response, **judge_result})
        logger.info("  Judge score=%d  reason=%s", judge_result["score"], judge_result["reason"])

    summary = summarise(raw_results)
    return {**summary, "accuracy_score": compute_accuracy(probe_scores), "probe_scores": probe_scores, "raw_results": raw_results}


def run_mem0() -> dict:
    logger.info("=" * 60)
    logger.info("MEM0 — starting benchmark")
    logger.info("=" * 60)

    try:
        client = _mem0_client()
    except EnvironmentError as exc:
        logger.error("%s — skipping Mem0", exc)
        return _empty_result("mem0")

    user_id = USER_ID
    mem0_clear(client, user_id)

    raw_results: list[dict] = []
    probe_scores: list[dict] = []

    for session_data in SESSIONS:
        s = session_data["session"]
        for msg in session_data["messages"]:
            logger.info("Mem0 ingest session=%d  msg=%r", s, msg[:60])
            record = mem0_ingest(client, user_id, msg, s)
            raw_results.append(record)
            logger.info("  tokens_in=%d  latency=%.1fms", record["tokens_in"], record["latency_ms"])

    for probe in PROBE_QUESTIONS:
        run_texts: list[str] = []

        for run_idx in range(PROBE_RUNS):
            response_text, latency_ms = mem0_recall(client, user_id, probe["query"])
            run_texts.append(response_text)
            raw_results.append({
                "library": "mem0",
                "session": 8,
                "operation": "recall",
                "tokens_in": count_tokens(probe["query"]),
                "tokens_out": count_tokens(response_text),
                "latency_ms": latency_ms,
            })
            logger.info("Mem0 recall run=%d  query=%r  latency=%.1fms", run_idx + 1, probe["query"], latency_ms)

        best_response = run_texts[-1]
        logger.info("Mem0 response: %s", best_response[:300])

        judge_result = judge_response(
            query=probe["query"],
            expected=probe["expected"],
            wrong_answers=probe["wrong_answers"],
            response=best_response,
        )
        probe_scores.append({"query": probe["query"], "expected": probe["expected"], "response": best_response, **judge_result})
        logger.info("  Judge score=%d  reason=%s", judge_result["score"], judge_result["reason"])

    summary = summarise(raw_results)
    return {**summary, "accuracy_score": compute_accuracy(probe_scores), "probe_scores": probe_scores, "raw_results": raw_results}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _empty_result(library: str) -> dict:
    return {
        "total_tokens_ingested": 0,
        "total_tokens_retrieved": 0,
        "p95_latency_ms": 0.0,
        "accuracy_score": 0.0,
        "probe_scores": [],
        "raw_results": [],
        "error": f"{library} skipped due to missing API key",
    }


def print_markdown_table(memanto_res: dict, mem0_res: dict):
    print(f"""
## Dynamic Preference Shift Benchmark Results

| Metric                   | Memanto          | Mem0             |
|--------------------------|------------------|------------------|
| Tokens Ingested          | {memanto_res['total_tokens_ingested']:>16,} | {mem0_res['total_tokens_ingested']:>16,} |
| Tokens Retrieved         | {memanto_res['total_tokens_retrieved']:>16,} | {mem0_res['total_tokens_retrieved']:>16,} |
| p95 Latency (ms)         | {memanto_res['p95_latency_ms']:>16.1f} | {mem0_res['p95_latency_ms']:>16.1f} |
| Accuracy Score (0–1)     | {memanto_res['accuracy_score']:>16.4f} | {mem0_res['accuracy_score']:>16.4f} |
""")

    print("### Probe Results\n")
    print(f"{'Query':<45} {'Expected':<25} {'Memanto':>8} {'Mem0':>8}")
    print("-" * 90)

    mem_scores = {p["query"]: p["score"] for p in memanto_res.get("probe_scores", [])}
    m0_scores = {p["query"]: p["score"] for p in mem0_res.get("probe_scores", [])}

    for probe in PROBE_QUESTIONS:
        q = probe["query"]
        print(f"{q:<45} {probe['expected']:<25} {str(mem_scores.get(q, 'N/A')):>8} {str(m0_scores.get(q, 'N/A')):>8}")


def main():
    missing = [k for k in ("MOORCHEH_API_KEY", "MEM0_API_KEY", "GROQ_API_KEY") if not os.getenv(k)]
    if missing:
        logger.warning("Missing env vars: %s", missing)

    run_date = datetime.now(tz=timezone.utc).isoformat()
    logger.info("Benchmark starting — %s", run_date)

    memanto_results = run_memanto()
    mem0_results = run_mem0()

    output = {
        "run_date": run_date,
        "judge_model": "llama-3.3-70b-versatile",
        "token_counter": "tiktoken cl100k_base",
        "probe_runs_per_question": PROBE_RUNS,
        "user_id": USER_ID,
        "memanto": memanto_results,
        "mem0": mem0_results,
    }

    results_path = Path(__file__).parent / "results" / "benchmark_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Results written to %s", results_path)

    print("\n=== MEMANTO PROBE RESPONSES ===")
    for p in memanto_results.get("probe_scores", []):
        print(f"Q: {p['query']}")
        print(f"Response: {p['response'][:400]}")
        print(f"Score: {p['score']} | Reason: {p['reason']}")
        print()

    print("\n=== MEM0 PROBE RESPONSES ===")
    for p in mem0_results.get("probe_scores", []):
        print(f"Q: {p['query']}")
        print(f"Response: {p['response'][:400]}")
        print(f"Score: {p['score']} | Reason: {p['reason']}")
        print()

    print_markdown_table(memanto_results, mem0_results)


if __name__ == "__main__":
    main()