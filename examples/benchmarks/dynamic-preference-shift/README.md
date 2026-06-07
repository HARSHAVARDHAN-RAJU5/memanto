# Memanto vs Mem0 — Dynamic Preference Shift Benchmark

A reproducible benchmark comparing [Memanto](https://moorcheh.ai) and [Mem0](https://mem0.ai) on a challenging real-world scenario: an AI agent that must track a user whose preferences change, evolve, and directly contradict across 8 sessions.

---

## Why Preference Shift Is the Hard Problem

Most memory benchmarks test static recall — store a fact, retrieve it later. That's easy. The real challenge is **preference drift**: a user says they love hip-hop in session 1, switches to jazz in session 4, settles on metal in session 6. A naive memory system piles up all three facts and returns them all. A smart system surfaces only the latest state.

This benchmark adds a second layer: **direct contradiction on safety-critical info**. In session 8, Alex says "I tried eggs and was completely fine" — directly contradicting a previously confirmed severe egg allergy. This tests whether a memory system can detect and surface contradictions rather than blindly returning the most recent or most confident fact.

---

## Scenario — "Alex" (8 Sessions)

A fictional user named Alex whose music taste and diet evolve — then contradict — across 8 sessions:

| Sessions | Music | Diet / Health |
|----------|-------|---------------|
| 1–3 | Hip-hop (J. Cole, Kendrick, Knxwledge, Freddie Gibbs) | Strict keto |
| 4–5 | Jazz (Miles Davis, Coltrane, Bill Evans) | Fully vegan |
| 6–7 | Metal (Metallica, Tool) | Vegan + severe egg allergy discovered |
| 8 | — | Contradicts allergy: "I ate eggs, no reaction at all" |

Total: **17 messages** ingested across 8 sessions. Same messages, same order, same user ID for both libraries — fully controlled.

---

## Quick Start

```bash
git clone https://github.com/moorcheh-ai/memanto
cd memanto/examples/benchmarks/dynamic-preference-shift
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
python benchmark.py
```

### Required API Keys

| Key | Where to get |
|-----|-------------|
| `MOORCHEH_API_KEY` | [moorcheh.ai](https://moorcheh.ai) — free, no card required |
| `MEM0_API_KEY` | [app.mem0.ai](https://app.mem0.ai) — free tier |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free tier |

---

## Results

**Run date:** 2026-06-07
**Judge model:** `llama-3.3-70b-versatile` via Groq, `temperature=0`
**Token counter:** `tiktoken` `cl100k_base`
**Library versions:** `memanto==0.1.3`, `mem0ai==2.0.4`, `groq==1.4.0`, `tiktoken==0.12.0`, `numpy==2.2.6`

### Summary Table

| Metric | Memanto | Mem0 |
|--------|---------|------|
| Tokens Ingested | 225 | 225 |
| Tokens Retrieved | 1,017 | 6,669 |
| p95 Latency (ms) | 1,280.4 | 817.9 |
| **Accuracy Score (0–1)** | **0.3000** | **0.1000** |

### Probe-Level Breakdown (10 Questions)

| Query | Expected | Memanto | Mem0 |
|-------|----------|---------|------|
| What music does Alex like? | metal (Tool, Metallica) | ❌ | ✅ |
| What are Alex's dietary preferences? | vegan, egg allergy | ❌ | ❌ |
| What time does Alex work out? | morning before 7am | ✅ | ❌ |
| What specific artists does Alex listen to now? | Tool, Metallica | ❌ | ❌ |
| Does Alex have any food allergies? | yes, egg allergy | ✅ | ❌ |
| Is it safe to serve Alex eggs? | no, severe egg allergy | ❌ | ❌ |
| What diet did Alex follow before going vegan? | keto diet | ❌ | ❌ |
| Has Alex's music taste changed over time? | yes — hip-hop → jazz → metal | ✅ | ❌ |
| Does Alex still have an egg allergy? | uncertain — ate eggs with no reaction | ❌ | ❌ |
| Is Alex's egg allergy confirmed or questioned? | questioned — ate eggs with no reaction | ❌ | ❌ |

**Memanto: 3/10 correct. Mem0: 1/10 correct.**

---

## Key Findings

**1. Memanto is 3x more accurate (0.30 vs 0.10)**
Memanto's selective retrieval (limit=3) returns fewer, more focused memories. Mem0 returns 5–6 memories per query including irrelevant ones, causing the judge to penalize noise-contaminated responses.

**2. Mem0 retrieves 6.6x more tokens per query (6,669 vs 1,017)**
Higher token retrieval means higher cost per query in a production agent. Mem0's approach of returning everything and letting the caller filter is expensive and inaccurate.

**3. Mem0 is faster (817ms vs 1,280ms p95)**
Memanto queues ingestion asynchronously (status: "queued") requiring a 20s indexing wait before recall. Mem0 is synchronous and faster at retrieval.

**4. Neither system handled contradiction detection**
Both scored 0 on "Does Alex still have an egg allergy?" and "Is Alex's egg allergy confirmed or questioned?" — the two contradiction-handling probes. When Alex said "I ate eggs with no reaction", neither system flagged the conflict with the previously stored severe allergy. This is an open problem in agent memory.

**5. Neither system solved historical recall**
Both scored 0 on "What diet did Alex follow before going vegan?" — requiring temporal reasoning over stored memories, not just retrieval of the latest state.

**6. Safety-critical queries are the highest-stakes failure**
The question "Is it safe to serve Alex eggs?" scored 0 for both libraries. In a real agent (meal planner, restaurant recommender), this failure could cause real harm. Neither library is production-ready for safety-critical memory without an additional reasoning layer.

---

## Methodology

### Controlled Variables
- **Same user/agent ID:** `alex-benchmark-v1` for both libraries
- **Same input messages:** identical text, identical session order
- **Sequential ingestion:** sessions 1→8 in order, no shuffling
- **Indexing wait:** 20 second sleep after all Memanto ingestion (Memanto queues writes asynchronously; recall on un-indexed memories returns empty)
- **Probe runs:** each question asked 3 times, final run used for judging (reduces latency noise)
- **Recall limit:** Memanto uses `limit=3`; Mem0 returns default results
- **Fresh state:** both libraries cleared before each run via `delete_all` / `clear`

### Token Counting
`tiktoken` with `cl100k_base` encoding is used for all token counts. This encoding matches the tokenization of most modern LLMs (GPT-4, GPT-3.5) and gives a consistent, reproducible count independent of which model serves the memory library backend.

Tokens counted on:
- **Ingestion:** raw message text sent to the library
- **Retrieval:** full text of all memories returned by the library

### LLM-as-a-Judge

**Model:** `llama-3.3-70b-versatile` via Groq API
**Temperature:** 0 (deterministic)
**Max tokens:** 100

**Exact judge prompt:**
```
You are an evaluation judge for AI memory systems.

Query: {query}
Expected (current preference): {expected}
Stale/wrong answers to penalize: {wrong_answers}
Memory system response: {response}

Does the response correctly reflect the CURRENT/LATEST preference?
Score 1 if it returns the current state. Score 0 if it returns stale info or nothing useful.
Respond ONLY with valid JSON: {"score": 0 or 1, "reason": "one sentence"}
```

The judge penalizes responses that include stale information even if the correct answer is also present — because in a real agent, mixed responses cause downstream errors.

### Latency Measurement
`time.perf_counter()` is used for all latency measurements. p95 is computed across all operations (ingest + recall) using `numpy.percentile`.

---

## Limitations

- Memanto's async indexing requires a cold-start delay. In production this is less of an issue (memories index continuously) but in a benchmark it adds fixed overhead.
- The Groq judge occasionally scores generously — Q1 (music) gave Mem0 a 1 despite the response containing stale hip-hop/jazz info alongside the correct metal answer. A stricter prompt would score this 0.
- Mem0's `search()` and `delete_all()` APIs use `filters={"user_id": ...}` syntax as of `mem0ai==2.0.4`. Earlier versions used `user_id=` directly — an undocumented breaking change.
- Contradiction detection scored 0 for both libraries — this requires reasoning over memory, not just retrieval, and is beyond the scope of either library's current API.
- Historical recall (what diet before vegan?) also scored 0 for both — same limitation.

---

## File Structure

```
dynamic-preference-shift/
├── benchmark.py                      # single-file runner, all logic included
├── requirements.txt                  # pinned dependencies
├── .env.example                      # required environment variables
├── README.md                         # this file
└── results/
    ├── benchmark_results.json        # full audit log with all raw API call records
    └── summary.json                  # clean summary for quick reading
```

---

## Requirements

```
memanto==0.1.3
mem0ai==2.0.4
groq==1.4.0
tiktoken==0.12.0
numpy==2.2.6
python-dotenv==1.1.0
```