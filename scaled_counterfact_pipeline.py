"""
Scaled Knowledge Editing Pipeline (n = 500) for CounterFact
1. Connects to Hugging Face Hub to load 'azhx/counterfact' and 'gpt2'.
2. Performs stratified sampling across 500 eligible records.
3. Computes log-probabilities for target, paraphrases, and neighbourhood prompts.
4. Outputs structured CSV data for evaluation and statistical testing.

Run:
    pip install torch transformers datasets accelerate
    python scaled_counterfact_pipeline.py --n 500 --seed 42 --model gpt2 --out scaled_baseline_results.csv
"""

import argparse
import csv
import random
import platform
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import datasets as datasets_lib

REPO = "azhx/counterfact"

def get_revision():
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(REPO).sha
    except Exception as err:
        return f"unknown ({err})"

def is_eligible(rec):
    """Inclusion / exclusion rules (C1-C5, E1-E5)."""
    rw = rec["requested_rewrite"]
    new = (rw["target_new"]["str"] or "").strip()
    true = (rw["target_true"]["str"] or "").strip()
    subject = (rw["subject"] or "").strip()
    template = rw["prompt"] or ""
    if not new or not true or not subject or "{}" not in template:
        return "E2 missing fields"
    if new.casefold() == true.casefold():
        return "E3 identical answers"
    if not rec["paraphrase_prompts"] or not rec["neighborhood_prompts"]:
        return "E2 no paraphrase/neighbourhood prompts"
    return None

def mean_logprob(model, tok, prompt, target, device):
    """Average log-probability of target tokens given prompt."""
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    target_ids = tok(" " + target.strip(), return_tensors="pt").input_ids.to(device)
    ids = torch.cat([prompt_ids, target_ids], dim=1)
    with torch.no_grad():
        logits = model(ids).logits
    logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
    start = prompt_ids.shape[1] - 1
    total = 0.0
    for i in range(target_ids.shape[1]):
        total += logprobs[start + i, target_ids[0, i]].item()
    return total / target_ids.shape[1]

def prefers(model, tok, prompt, first, second, device):
    return mean_logprob(model, tok, prompt, first, device) > mean_logprob(model, tok, prompt, second, device)

def greedy_answer(model, tok, prompt, device, max_new_tokens=5):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:]).strip()

def sample_cases(records, n, seed):
    """Stratified sampling across relation types using round-robin selection."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for rec in records:
        groups[rec["requested_rewrite"]["relation_id"]].append(rec)
    for key in groups:
        rng.shuffle(groups[key])
    keys = sorted(groups)
    chosen = []
    while len(chosen) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(chosen) < n:
                chosen.append(groups[k].pop())
    return chosen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--out", default="scaled_baseline_results.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== SCALED KNOWLEDGE EDITING PIPELINE ===")
    print(f"Python: {platform.python_version()} | OS: {platform.platform()}")
    print(f"torch: {torch.__version__} | Device: {device}")
    print(f"Dataset: {REPO} | Revision: {get_revision()}")
    print(f"Model: {args.model} | Target Sample Size (n): {args.n}")

    # Load dataset from Hugging Face
    data = load_dataset(REPO, split="train")
    print(f"Total records in train split: {len(data)}")

    eligible, excluded = [], defaultdict(int)
    for rec in data:
        reason = is_eligible(rec)
        if reason:
            excluded[reason] += 1
        else:
            eligible.append(rec)
    print(f"Eligible records: {len(eligible)} | Excluded breakdown: {dict(excluded)}")

    cases = sample_cases(eligible, args.n, args.seed)
    print(f"Successfully sampled {len(cases)} cases across relation types.")

    # Load Tokenizer and Model from Hugging Face Hub
    print(f"Loading model '{args.model}' from Hugging Face...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    rows = []
    for k, rec in enumerate(cases, 1):
        rw = rec["requested_rewrite"]
        new, true = rw["target_new"]["str"], rw["target_true"]["str"]
        prompt = rw["prompt"].format(rw["subject"])

        target_ok = prefers(model, tok, prompt, new, true, device)
        para = [prefers(model, tok, p, new, true, device) for p in rec["paraphrase_prompts"]]
        neigh = [prefers(model, tok, p, true, new, device) for p in rec["neighborhood_prompts"]]
        generated = greedy_answer(model, tok, prompt, device)

        rows.append({
            "case_id": rec["case_id"],
            "relation_id": rw["relation_id"],
            "subject": rw["subject"],
            "prompt": prompt,
            "target_new": new,
            "target_true": true,
            "target_new_preferred": int(target_ok),
            "paraphrase_new_preferred": round(sum(para) / len(para), 3) if para else 0.0,
            "neighbourhood_true_preferred": round(sum(neigh) / len(neigh), 3) if neigh else 0.0,
            "greedy_answer": generated,
            "greedy_equals_true": int(generated.casefold().startswith(true.casefold())),
        })
        if k % 50 == 0 or k == len(cases):
            print(f"  Processed {k}/{len(cases)} cases...")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print("\n=== SCALED BASELINE SUMMARY (n = 500) ===")
    print(f"Total Cases Evaluated: {n}")
    print(f"Target Success Rate (New preferred over True): {sum(r['target_new_preferred'] for r in rows) / n:.3f}")
    print(f"Paraphrase Generalisation Score:              {sum(r['paraphrase_new_preferred'] for r in rows) / n:.3f}")
    print(f"Neighbourhood Locality Score:                 {sum(r['neighbourhood_true_preferred'] for r in rows) / n:.3f}")
    print(f"Greedy Match with True Object:                {sum(r['greedy_equals_true'] for r in rows) / n:.3f}")
    print(f"Results successfully saved to: {args.out}")

if __name__ == "__main__":
    main()