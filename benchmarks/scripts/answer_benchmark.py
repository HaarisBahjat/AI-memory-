import os
import sys
import json
import yaml
import asyncio
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(BASE_DIR)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from app.core.database import AsyncSessionLocal
from benchmarks.scripts.llm_judge import evaluate_answer
import app.services.retrieval_engine
from app.services.embedding_service import get_openai_client, settings

async def generate_rag_answer(query: str, system_prompt_context: str) -> str:
    client = get_openai_client()
    chat_payload = dict(
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt_context},
            {"role": "user", "content": query},
        ],
    )
    chat_response = await client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        **chat_payload,
    )
    return chat_response.choices[0].message.content

async def run_end_to_end_eval(config_name: str, sample_size: int = 5):
    print(f"\n========================================")
    print(f"Running Answer Benchmark for config: {config_name}")
    print(f"========================================")
    
    # Find the most recent run that contains the retrieval file
    results_dir = os.path.join(BASE_DIR, "results")
    runs = sorted([d for d in os.listdir(results_dir) if d.startswith("run_")], reverse=True)
    
    retrieval_file = None
    for run in runs:
        candidate_file = os.path.join(results_dir, run, f"retrieval_{config_name}_per_query.json")
        if os.path.exists(candidate_file):
            retrieval_file = candidate_file
            break
            
    if not retrieval_file:
        print(f"Retrieval file for config {config_name} not found in any run.")
        return
        
    with open(retrieval_file, "r") as f:
        retrievals = json.load(f)
        
    data_dir = os.path.join(BASE_DIR, "dataset", "synthetic")
    with open(os.path.join(data_dir, "queries.json")) as f:
        all_queries = json.load(f)["queries"]
        
    with open(os.path.join(data_dir, "memories.json")) as f:
        all_memories = json.load(f)["memories"]
    mem_map = {m["id"]: m["content"] for m in all_memories}

    # Filter to human authored for gold standard
    gold_queries = [q for q in all_queries if q.get("is_human_authored", False)]
    if not gold_queries:
        # Fallback to random sample
        gold_queries = all_queries[:sample_size]
    
    gold_queries = gold_queries[:sample_size]
    
    metrics = {
        "total_evaluated": 0,
        "is_correct": 0,
        "is_supported": 0,
        # Abstention categorization
        "correct_abstention": 0,       # NO_ANSWER query -> system correctly refused
        "incorrect_abstention": 0,     # ANSWER query -> system wrongly refused
        "hallucinated_memory": 0,      # system answered but context was empty / irrelevant
        "false_answer": 0,             # system answered NO_ANSWER query with wrong answer
        "per_query": []
    }
    
    for gq in gold_queries:
        ret = next((r for r in retrievals if r["query_id"] == gq["id"]), None)
        if not ret: continue
        
        # Assemble context string
        context_memories = []
        for rid in ret["retrieved_memory_ids"]:
            if rid in mem_map:
                context_memories.append({"text": mem_map[rid], "category": "fact", "adjusted_score": 1.0, "reinforcement_count": 1})
        
        system_prompt = app.services.retrieval_engine.assemble_system_prompt([], [], context_memories)
        
        print(f"\nQuery: {gq['query']}")
        print(f"Context IDs: {ret['retrieved_memory_ids']}")
        
        # 1. Generate Answer
        gen_answer = await generate_rag_answer(gq['query'], system_prompt)
        print(f"Generated: {gen_answer}")
        
        context_str = "\n".join([m["text"] for m in context_memories])
        
        # 2. Judge Answer
        judge_score = await evaluate_answer(
            query=gq['query'],
            ground_truth=gq.get('ground_truth_answer', ''),
            generated_answer=gen_answer,
            retrieved_context=context_str,
            category=gq['category']
        )
        
        expected_abstain = gq.get("expected_behavior") == "ABSTAIN"
        generated_lower = gen_answer.lower()
        # Heuristic: check if model said it doesn't know
        model_abstained = any(phrase in generated_lower for phrase in [
            "i don't know", "i do not know", "no information", "cannot find",
            "not available", "no record", "i have no", "not mentioned", "no context"
        ])
        
        # Categorize abstention behavior
        if expected_abstain and model_abstained:
            abstention_label = "CORRECT_ABSTENTION"
            metrics["correct_abstention"] += 1
        elif expected_abstain and not model_abstained:
            abstention_label = "FALSE_ANSWER"  # Model hallucinated an answer for unknown info
            metrics["false_answer"] += 1
        elif not expected_abstain and model_abstained:
            abstention_label = "INCORRECT_ABSTENTION"  # Model refused when it should answer
            metrics["incorrect_abstention"] += 1
        else:
            abstention_label = "ANSWERED"  # Normal case
        
        # Hallucination check: answered but with empty context
        if not model_abstained and not context_memories:
            metrics["hallucinated_memory"] += 1
        
        print(f"Judge: Correct={judge_score.is_correct}, Supported={judge_score.is_supported}, Abstention={abstention_label}")
        print(f"Reasoning: {judge_score.reasoning}")
        
        metrics["total_evaluated"] += 1
        metrics["is_correct"] += int(judge_score.is_correct)
        metrics["is_supported"] += int(judge_score.is_supported)
        
        metrics["per_query"].append({
            "query_id": gq["id"],
            "category": gq.get("category"),
            "expected_behavior": gq.get("expected_behavior"),
            "model_abstained": model_abstained,
            "abstention_label": abstention_label,
            "is_correct": judge_score.is_correct,
            "is_supported": judge_score.is_supported,
            "reasoning": judge_score.reasoning
        })
        
    if metrics["total_evaluated"] == 0:
        print("No queries evaluated.")
        return
        
    n = metrics["total_evaluated"]
    final_metrics = {
        "strict_accuracy": metrics["is_correct"] / n,
        "hallucination_rate": 1.0 - (metrics["is_supported"] / n),
        "abstention_precision": metrics["correct_abstention"] / max(metrics["correct_abstention"] + metrics["false_answer"], 1),
        "abstention_recall": metrics["correct_abstention"] / max([q for q in gold_queries if q.get("expected_behavior")=="ABSTAIN"].__len__(), 1),
        "false_answer_rate": metrics["false_answer"] / n,
        "hallucinated_memory_rate": metrics["hallucinated_memory"] / n,
        "correct_abstentions": metrics["correct_abstention"],
        "false_answers": metrics["false_answer"],
        "incorrect_abstentions": metrics["incorrect_abstention"],
        "hallucinated_memories": metrics["hallucinated_memory"],
    }
    
    print("\n--- FINAL ANSWER METRICS ---")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")
        
    run_id = f"run_{int(datetime.now().timestamp())}"
    res_dir = os.path.join(BASE_DIR, "results", run_id)
    os.makedirs(res_dir, exist_ok=True)
    
    output = {**final_metrics, "per_query": metrics["per_query"]}
    with open(os.path.join(res_dir, f"answer_metrics_{config_name}.json"), "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="full_system", help="Config to evaluate")
    args = parser.parse_args()
    
    asyncio.run(run_end_to_end_eval(args.config))
