import json

data = json.load(open("benchmark_results.json"))

summary = {
    "run_date": data["run_date"],
    "judge_model": data["judge_model"],
    "token_counter": data["token_counter"],
    "probe_runs_per_question": data["probe_runs_per_question"],
    "memanto": {
        "total_tokens_ingested": data["memanto"]["total_tokens_ingested"],
        "total_tokens_retrieved": data["memanto"]["total_tokens_retrieved"],
        "p95_latency_ms": data["memanto"]["p95_latency_ms"],
        "accuracy_score": data["memanto"]["accuracy_score"],
        "probe_scores": [
            {"query": p["query"], "expected": p["expected"], "score": p["score"], "reason": p["reason"]}
            for p in data["memanto"]["probe_scores"]
        ],
    },
    "mem0": {
        "total_tokens_ingested": data["mem0"]["total_tokens_ingested"],
        "total_tokens_retrieved": data["mem0"]["total_tokens_retrieved"],
        "p95_latency_ms": data["mem0"]["p95_latency_ms"],
        "accuracy_score": data["mem0"]["accuracy_score"],
        "probe_scores": [
            {"query": p["query"], "expected": p["expected"], "score": p["score"], "reason": p["reason"]}
            for p in data["mem0"]["probe_scores"]
        ],
    }
}

json.dump(summary, open("summary.json", "w"), indent=2)
print("done")