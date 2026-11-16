"""
Condition C2: ROME edit on GPT-2, evaluated with EXACTLY the same scoring code as the
unedited baseline (scaled_counterfact_pipeline.py).

For every case (in the same order as the baseline CSV):
    1. start from the original GPT-2 weights
    2. apply one ROME edit (new object for the subject)
    3. score target / paraphrase / neighbourhood prompts + greedy answer
    4. restore the original weights (edits never accumulate)

Run (from the folder that contains EasyEdit/, scaled_counterfact_pipeline.py and the baseline CSV):
    python run_rome.py --limit 5                 # smoke test, ~1-2 min on GPU
    python run_rome.py                           # full run, 500 cases (resumable)
"""
import argparse
import csv
import os
import sys
import time
import types
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                          # scaled_counterfact_pipeline.py
sys.path.insert(0, os.path.join(HERE, "EasyEdit"))  # easyeditor package

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same functions, same scoring rule as the baseline -> fair comparison.
from scaled_counterfact_pipeline import REPO, greedy_answer, prefers


# --------------------------------------------------------------------------------------
# Importing easyeditor pulls in many packages that are only needed for multimodal /
# trainer methods (timm, cv2, higher, ...). ROME does not use them, so if one of these
# OPTIONAL packages is missing we replace it with an empty stub instead of installing it.
# Anything else that is missing is a real error and is reported.
# --------------------------------------------------------------------------------------
OPTIONAL = {"higher", "timm", "cv2", "av", "iopath", "fairscale", "qwen_vl_utils",
            "zhipuai", "openai", "bitsandbytes", "gpustat", "rouge", "peft",
            "sentence_transformers", "decord", "torchvision", "torchaudio"}


import importlib.machinery

class _Stub(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
        self.__spec__ = importlib.machinery.ModuleSpec(name, None)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None})


def import_rome():
    stubbed = []
    for _ in range(60):
        try:
            from easyeditor.models.rome import ROMEHyperParams, apply_rome_to_model
            if stubbed:
                print(f"[info] stubbed optional packages not needed for ROME: {sorted(set(stubbed))}")
            return ROMEHyperParams, apply_rome_to_model
        except ModuleNotFoundError as err:
            top = (err.name or "").split(".")[0]
            if top in OPTIONAL:
                for name in {err.name, top}:
                    sys.modules[name] = _Stub(name)
                stubbed.append(top)
                continue
            raise SystemExit(f"Missing required package '{err.name}'. Install it with:  pip install {top}")
    raise SystemExit("Could not import easyeditor.models.rome")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="scaled_baseline_results.csv")
    ap.add_argument("--out", default="rome_results.csv")
    ap.add_argument("--errors", default="rome_errors.csv")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--hparams", default=os.path.join(HERE, "hparams_gpt2_rome.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases (smoke test)")
    args = ap.parse_args()

    ROMEHyperParams, apply_rome_to_model = import_rome()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | device: {device}")
    torch.manual_seed(42)

    base = pd.read_csv(args.baseline)
    order = base["case_id"].tolist()
    base_target = dict(zip(base["case_id"], base["target_new_preferred"]))
    if args.limit:
        order = order[: args.limit]

    print(f"Loading {REPO} ...")
    data = load_dataset(REPO, split="train")
    wanted = set(order)
    by_id = {ex["case_id"]: ex for ex in data if ex["case_id"] in wanted}
    missing = [c for c in order if c not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} baseline case_ids not found in dataset, e.g. {missing[:5]}")

    hparams = ROMEHyperParams.from_hparams(args.hparams)
    hparams.device = 0 if device == "cuda" else "cpu"
    print(f"ROME hyper-parameters: layers={hparams.layers}, v_loss_layer={hparams.v_loss_layer}, "
          f"v_num_grad_steps={hparams.v_num_grad_steps}, mom2_adjustment={hparams.mom2_adjustment}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()
    params = dict(model.named_parameters())

    # resume support: skip cases that are already in the output file
    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out)["case_id"].tolist())
        print(f"Resuming: {len(done)} cases already in {args.out}")

    fields = ["case_id", "relation_id", "subject", "prompt", "target_new", "target_true",
              "target_new_preferred", "paraphrase_new_preferred", "neighbourhood_true_preferred",
              "greedy_answer", "greedy_equals_true", "edit_seconds"]
    new_file = not os.path.exists(args.out)
    f_out = open(args.out, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fields)
    if new_file:
        writer.writeheader()
    f_err = open(args.errors, "a", encoding="utf-8", newline="")
    err_writer = csv.writer(f_err)

    n_err, n_pre_mismatch, t_start = 0, 0, time.time()
    for k, cid in enumerate(order, 1):
        if cid in done:
            continue
        rec = by_id[cid]
        rw = rec["requested_rewrite"]
        new, true = rw["target_new"]["str"], rw["target_true"]["str"]
        prompt = rw["prompt"].format(rw["subject"])

        # sanity check: the unedited model must reproduce the baseline CSV
        if int(prefers(model, tok, prompt, new, true, device)) != int(base_target[cid]):
            n_pre_mismatch += 1

        request = [{"prompt": rw["prompt"], "subject": rw["subject"],
                    "target_new": new, "target_true": true}]
        orig = {}
        try:
            t0 = time.time()
            _, orig = apply_rome_to_model(model, tok, request, hparams,
                                          copy=False, return_orig_weights=True)
            edit_s = time.time() - t0

            target_ok = prefers(model, tok, prompt, new, true, device)
            para = [prefers(model, tok, p, new, true, device) for p in rec["paraphrase_prompts"]]
            neigh = [prefers(model, tok, p, true, new, device) for p in rec["neighborhood_prompts"]]
            generated = greedy_answer(model, tok, prompt, device)

            writer.writerow({
                "case_id": cid,
                "relation_id": rw["relation_id"],
                "subject": rw["subject"],
                "prompt": prompt,
                "target_new": new,
                "target_true": true,
                "target_new_preferred": int(target_ok),
                "paraphrase_new_preferred": round(sum(para) / len(para), 3),
                "neighbourhood_true_preferred": round(sum(neigh) / len(neigh), 3),
                "greedy_answer": generated,
                "greedy_equals_true": int(generated.casefold().startswith(true.casefold())),
                "edit_seconds": round(edit_s, 2),
            })
            f_out.flush()
        except Exception as err:  # keep going, but never hide the failure
            n_err += 1
            err_writer.writerow([cid, repr(err)])
            f_err.flush()
            print(f"  [error] case {cid}: {err!r}")
        finally:
            # restore original GPT-2 weights, also after a failed edit
            with torch.no_grad():
                for name, w in orig.items():
                    params[name][...] = w
            if device == "cuda":
                torch.cuda.empty_cache()

        if k % 10 == 0 or k == len(order):
            el = time.time() - t_start
            print(f"  {k}/{len(order)} cases | {el/60:.1f} min elapsed | errors: {n_err}")

    f_out.close()
    f_err.close()
    final = pd.read_csv(args.out)
    print("\n=== ROME SUMMARY ===")
    print(f"Cases written: {len(final)} / {len(order)} | failed edits this session: {n_err}")
    print(f"Pre-edit check vs baseline CSV: {n_pre_mismatch} mismatches (should be 0 or very few)")
    print(f"Total edit time (s): {final['edit_seconds'].sum():.0f}")
    print(f"Target success (new preferred):  {final['target_new_preferred'].mean():.3f}")
    print(f"Paraphrase (new preferred):      {final['paraphrase_new_preferred'].mean():.3f}")
    print(f"Neighbourhood (true preferred):  {final['neighbourhood_true_preferred'].mean():.3f}")
    print(f"Greedy answer starts with true:  {final['greedy_equals_true'].mean():.3f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()