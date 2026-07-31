#!/usr/bin/env python
# coding: utf-8
"""
Verify that the fused adapter (`adapters/cocktail_lora`) reproduces the original
two-adapter pipeline (SFT -> thinker, then GRPO -> full model).

Two checks, in one process so both models see identical inputs:

  1. Weights — assemble the model both ways and diff every parameter.
  2. Generation — run the 10 Cocktail-DialogGen example turns through each
     assembled model and compare the emitted action token and response text.

Only one model is resident on the GPU at a time; the first assembly's weights
are snapshotted to CPU before it is freed.
"""
import os
import gc
import json
import argparse

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from peft import PeftModel

from inference import (
    BASE_MODEL, SFT_ADAPTER, GRPO_ADAPTER, PROCESSOR_DIR, EXAMPLES_DIR,
    run_turn,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MERGED_ADAPTER = os.environ.get("CT_MERGED_ADAPTER", os.path.join(HERE, "adapters", "cocktail_lora"))


def load_base(device, dtype):
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, device_map=device
    )
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    return model


def load_two_stage(device, dtype):
    """The original recipe: SFT into the thinker, then GRPO into the full model."""
    model = load_base(device, dtype)
    model.thinker = PeftModel.from_pretrained(model.thinker, SFT_ADAPTER).merge_and_unload()
    model = PeftModel.from_pretrained(model, GRPO_ADAPTER).merge_and_unload()
    return model.eval()


def load_merged(device, dtype):
    """The fused recipe: one adapter, one call (rooted at the thinker)."""
    model = load_base(device, dtype)
    model.thinker = PeftModel.from_pretrained(model.thinker, MERGED_ADAPTER).merge_and_unload()
    return model.eval()


def snapshot(model):
    return {k: v.detach().to("cpu", torch.float32) for k, v in model.state_dict().items()}


def run_examples(model, processor, examples, examples_dir):
    preds = []
    for e in examples:
        audio_path = os.path.join(examples_dir, e["input_audio"])
        action, text, raw = run_turn(model, processor, audio_path, e["metadata_prompt"])
        preds.append({"id": e["id"], "action": action, "text": text, "raw": raw,
                      "oracle_action": e["oracle_action"]})
        print(f"    {e['id']:42s} -> {action:12s} {text[:60]!r}", flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--examples", default=EXAMPLES_DIR)
    ap.add_argument("--skip-weights", action="store_true",
                    help="only compare generations (halves peak CPU RAM)")
    args = ap.parse_args()

    dtype = torch.bfloat16
    examples = json.load(open(os.path.join(args.examples, "examples.json")))
    processor = Qwen2_5OmniProcessor.from_pretrained(PROCESSOR_DIR)

    print("=" * 78)
    print("[1/2] Assembling the ORIGINAL two-adapter model (SFT -> thinker, GRPO -> full)")
    print("=" * 78, flush=True)
    model = load_two_stage(args.device, dtype)
    ref_weights = None if args.skip_weights else snapshot(model)
    print("  generating:", flush=True)
    ref_preds = run_examples(model, processor, examples, args.examples)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print()
    print("=" * 78)
    print("[2/2] Assembling the FUSED model (single adapter)")
    print("=" * 78, flush=True)
    model = load_merged(args.device, dtype)

    weight_report = None
    if ref_weights is not None:
        new_weights = snapshot(model)
        missing = sorted(set(ref_weights) ^ set(new_weights))
        worst, worst_key, n_exact = 0.0, None, 0
        for k in ref_weights:
            if k not in new_weights:
                continue
            a, b = ref_weights[k], new_weights[k]
            d = (a - b).abs().max().item()
            n_exact += int(d == 0.0)
            if d > worst:
                worst, worst_key = d, k
        weight_report = {
            "n_params": len(ref_weights),
            "key_mismatch": missing,
            "bitwise_identical": n_exact,
            "max_abs_diff": worst,
            "max_abs_diff_at": worst_key,
        }
        del new_weights
        gc.collect()

    print("  generating:", flush=True)
    new_preds = run_examples(model, processor, examples, args.examples)

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)

    ok = True
    if weight_report is not None:
        wr = weight_report
        print(f"Weights : {wr['n_params']} tensors compared, "
              f"{wr['bitwise_identical']} bitwise identical")
        print(f"          max |two-stage - fused| = {wr['max_abs_diff']:.3e}"
              f"  (at {wr['max_abs_diff_at']})")
        if wr["key_mismatch"]:
            print(f"          !! key mismatch: {wr['key_mismatch'][:5]}")
            ok = False
        # The two paths are equal in exact arithmetic but round to bf16 a
        # different number of times, so the LoRA-touched matrices may differ in
        # their last bit. bf16 carries 8 mantissa bits, so one ULP is ~2^-8 of
        # the magnitude; weights here are O(1) at most. Anything past 1e-2 is a
        # wrong delta, not rounding.
        if wr["max_abs_diff"] > 1e-2:
            ok = False
    else:
        print("Weights : skipped")

    n_same_action = n_same_text = 0
    for r, n in zip(ref_preds, new_preds):
        same_a = r["action"] == n["action"]
        same_t = r["text"] == n["text"]
        n_same_action += same_a
        n_same_text += same_t
        # The turn action is the model's decision and the metric the paper
        # reports; it must match exactly. Response *wording* is a ~30-step
        # greedy rollout, where a last-bit logit difference anywhere can pick a
        # different token and diverge from there — see control_numerics.py,
        # which shows the unfused model drifting just as much against itself
        # when only the attention kernel changes. So wording is reported, not
        # required.
        flag = "OK  " if same_a and same_t else ("WORD" if same_a else "DIFF")
        print(f"[{flag}] {r['id']}")
        if not same_a:
            ok = False
        if not (same_a and same_t):
            print(f"        two-stage: {r['action']:12s} {r['text']!r}")
            print(f"        fused    : {n['action']:12s} {n['text']!r}")

    n = len(examples)
    print(f"\nAction match : {n_same_action}/{n}   <- must be {n}/{n}")
    print(f"Text match   : {n_same_text}/{n}   <- informational (greedy rollout drift)")
    print(f"Action accuracy vs oracle — two-stage: "
          f"{sum(p['action'] == p['oracle_action'] for p in ref_preds)}/{n}, "
          f"fused: {sum(p['action'] == p['oracle_action'] for p in new_preds)}/{n}")

    out = os.path.join(HERE, "outputs", "merge_verification.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"weights": weight_report, "two_stage": ref_preds, "fused": new_preds},
              open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\nReport written to {out}")
    print("\n" + ("PASS — same weights to within bf16 rounding, same turn actions."
                  if ok else "FAIL — the fused adapter is not the same model, see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
