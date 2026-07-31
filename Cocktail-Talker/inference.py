#!/usr/bin/env python
# coding: utf-8
"""
Cocktail-Talker — turn-action + response inference on the Cocktail-DialogGen examples.

Cocktail-Talker is Qwen2.5-Omni-7B adapted with two LoRA modules:
  1) an SFT adapter (applied to the *thinker*), which also learns the three new
     turn-action tokens  <|respond|> / <|listen|> / <|ignore|> , and
  2) a GRPO adapter (applied to the *full* model) trained on top of the SFT model.
By default this script loads them pre-fused as the single adapter
`adapters/cocktail_lora` (see `merge_lora.py`); `--two-adapters` applies the two
original adapters in training order instead. Both give identical outputs.

At each dialog turn the model receives the speakers' metadata (text) and the
noisy multi-party audio mixture heard so far, and emits ONE action token:
  <|respond|> <response text>   -> the agent speaks
  <|listen|>                    -> the agent stays silent (attending)
  <|ignore|>                    -> the agent stays silent (irrelevant sound)

This script produces the model's decision (action) and, for <|respond|>, the
response *text*. Rendering that text to speech (Qwen3-TTS, Chelsie voice) is a
separate stage — see `synthesize_tts.py`. Predictions are written to
`outputs/predictions.json` for that stage to consume.

The base Qwen2.5-Omni-7B weights are downloaded from the Hugging Face Hub; this
repo ships only the two small LoRA adapters + the tokenizer/processor.
"""
import os
import json
import argparse

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Configuration (override via CLI flags or environment variables)
# ----------------------------------------------------------------------------
BASE_MODEL = os.environ.get("CT_BASE_MODEL", "Qwen/Qwen2.5-Omni-7B")
SFT_ADAPTER = os.environ.get("CT_SFT_ADAPTER", os.path.join(HERE, "adapters", "sft_lora"))
GRPO_ADAPTER = os.environ.get("CT_GRPO_ADAPTER", os.path.join(HERE, "adapters", "grpo_lora"))
MERGED_ADAPTER = os.environ.get("CT_MERGED_ADAPTER", os.path.join(HERE, "adapters", "cocktail_lora"))
PROCESSOR_DIR = os.environ.get("CT_PROCESSOR", os.path.join(HERE, "processor"))
EXAMPLES_DIR = os.environ.get(
    "CT_EXAMPLES", os.path.normpath(os.path.join(HERE, "..", "Cocktail-DialogGen", "examples"))
)
OUTPUT_DIR = os.environ.get("CT_OUTPUT", os.path.join(HERE, "outputs"))

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def load_cocktail_talker(device="cuda:0", dtype=torch.bfloat16, fused=True):
    """Load stock Qwen2.5-Omni-7B and apply the Cocktail-Talker LoRA weights.

    Two equivalent routes to the same checkpoint:

    `fused=True` (default) applies `adapters/cocktail_lora`, the single rank-256
    adapter produced by `merge_lora.py` that carries the sum of the two deltas.

    `fused=False` reproduces the original training order: the SFT adapter into
    the *thinker* (its native training scope, where `embed_tokens`/`lm_head`
    unambiguously refer to the language model and the three action-token
    embeddings are learned), then the GRPO adapter into the full model.

    The talker (speech decoder) is disabled: Cocktail-Talker only *decides and
    writes text*; speech is rendered separately by Qwen3-TTS.
    """
    print(f"Loading base model: {BASE_MODEL}", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, device_map=device
    )
    if fused:
        print(f"Applying fused SFT+GRPO adapter: {MERGED_ADAPTER}", flush=True)
        model.thinker = PeftModel.from_pretrained(
            model.thinker, MERGED_ADAPTER
        ).merge_and_unload()
    else:
        print(f"Applying SFT adapter (thinker):  {SFT_ADAPTER}", flush=True)
        model.thinker = PeftModel.from_pretrained(model.thinker, SFT_ADAPTER).merge_and_unload()
        print(f"Applying GRPO adapter (full):    {GRPO_ADAPTER}", flush=True)
        model = PeftModel.from_pretrained(model, GRPO_ADAPTER).merge_and_unload()
    if hasattr(model, "disable_talker"):
        model.disable_talker()  # text-only; frees the speech decoder
    model.eval()

    processor = Qwen2_5OmniProcessor.from_pretrained(PROCESSOR_DIR)
    print("Cocktail-Talker ready.\n", flush=True)
    return model, processor


# ----------------------------------------------------------------------------
# A single dialog turn
# ----------------------------------------------------------------------------
@torch.inference_mode()
def run_turn(model, processor, audio_path, metadata_prompt, max_new_tokens=512):
    """Run one turn: given the prefix audio + metadata prompt, return
    (action_token, response_text, raw_output)."""
    device = next(model.parameters()).device
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": metadata_prompt},
        ]},
    ]
    prompt_text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=prompt_text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
    ).to(device)
    for k, v in list(inputs.items()):
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs[k] = v.to(model.dtype)

    with torch.autocast(device_type=device.type, dtype=model.dtype):
        text_ids = model.generate(
            **inputs,
            return_audio=False,
            thinker_max_new_tokens=max_new_tokens,
            thinker_do_sample=False,
        )

    gen_ids = text_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(
        gen_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=True
    )
    for marker in ["<|im_end|>", "<|endoftext|>", "</audio>", "Human:"]:
        p = raw.find(marker)
        if p != -1:
            raw = raw[:p]
    raw = raw.strip()

    parts = raw.split(None, 1)
    action = parts[0] if parts else ""
    response_text = parts[1].strip() if len(parts) > 1 else ""
    return action, response_text, raw


# ----------------------------------------------------------------------------
# Run all examples
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run Cocktail-Talker on the example turns.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--examples", default=EXAMPLES_DIR)
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--two-adapters", action="store_true",
                    help="apply the separate SFT + GRPO adapters instead of the fused one")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    examples = json.load(open(os.path.join(args.examples, "examples.json")))
    model, processor = load_cocktail_talker(device=args.device, fused=not args.two_adapters)

    predictions = []
    n_correct = 0
    for e in examples:
        audio_path = os.path.join(args.examples, e["input_audio"])
        action, text, _ = run_turn(model, processor, audio_path, e["metadata_prompt"])

        correct = (action == e["oracle_action"])
        n_correct += correct
        predictions.append({
            "id": e["id"],
            "pred_action": action,
            "pred_text": text,
            "oracle_action": e["oracle_action"],
            "oracle_text": e["oracle_text"],
            "correct_action": bool(correct),
        })
        print(f"[{'OK ' if correct else 'XX '}] {e['id']}")
        print(f"        oracle : {e['oracle_action']:12s} {(e['oracle_text'] or '')[:70]!r}")
        print(f"        pred   : {action:12s} {text[:70]!r}")

    out_json = os.path.join(args.output, "predictions.json")
    json.dump(predictions, open(out_json, "w"), indent=2, ensure_ascii=False)
    print(f"\nAction accuracy vs oracle: {n_correct}/{len(examples)}")
    print(f"Predictions written to    : {out_json}")
    print("Render <|respond|> replies to speech with:  python synthesize_tts.py")


if __name__ == "__main__":
    main()
