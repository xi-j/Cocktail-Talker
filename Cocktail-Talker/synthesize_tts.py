#!/usr/bin/env python
# coding: utf-8
"""
Stage 2 — render Cocktail-Talker's <|respond|> text to speech with Qwen3-TTS.

Cocktail-Talker decides the turn action and writes the response *text*
(`inference.py` -> outputs/predictions.json). This script speaks the text of the
<|respond|> turns using Qwen3-TTS with the agent's fixed "Chelsie" voice — the
same TTS engine and voice used for the dataset's ground-truth agent audio.

This runs in a SEPARATE environment from `inference.py` (Qwen3-TTS needs a newer
transformers). See requirements-tts.txt:

    conda create -n cocktail-tts python=3.12 -y && conda activate cocktail-tts
    pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements-tts.txt
    python synthesize_tts.py
"""
import os
import json
import argparse

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

HERE = os.path.dirname(os.path.abspath(__file__))

TTS_MODEL = os.environ.get("CT_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
REF_AUDIO = os.environ.get("CT_TTS_REF_AUDIO", os.path.join(HERE, "assets", "Chelsie.wav"))
REF_TEXT = os.environ.get("CT_TTS_REF_TEXT",
    "Hi, nice to meet you. I am glad to serve as the agent's voice for this "
    "project. Please let me know how I can help.")
PREDICTIONS = os.environ.get("CT_PREDICTIONS", os.path.join(HERE, "outputs", "predictions.json"))
OUTPUT_DIR = os.environ.get("CT_OUTPUT", os.path.join(HERE, "outputs"))


def main():
    ap = argparse.ArgumentParser(description="Render <|respond|> texts to Chelsie speech via Qwen3-TTS.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--predictions", default=PREDICTIONS)
    ap.add_argument("--output", default=OUTPUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    predictions = json.load(open(args.predictions))
    respond = [p for p in predictions if p["pred_action"] == "<|respond|>" and p["pred_text"].strip()]
    print(f"{len(respond)} <|respond|> turns to synthesize.")

    print(f"Loading {TTS_MODEL} ...", flush=True)
    model = Qwen3TTSModel.from_pretrained(TTS_MODEL, device_map=args.device, dtype=torch.bfloat16)
    voice = model.create_voice_clone_prompt(
        ref_audio=REF_AUDIO, ref_text=REF_TEXT, x_vector_only_mode=False
    )
    print("Chelsie voice-clone prompt ready.\n", flush=True)

    for p in respond:
        torch.manual_seed(0)
        wavs, sr = model.generate_voice_clone(
            text=[p["pred_text"]], language=["English"], voice_clone_prompt=voice
        )
        wav = wavs[0]
        out_wav = os.path.join(args.output, f"{p['id']}.wav")
        sf.write(out_wav, wav, sr)
        print(f"[{p['id']}]  {len(wav)/sr:5.1f}s  {p['pred_text'][:60]!r}")

    print(f"\nSpeech WAVs -> {args.output}")


if __name__ == "__main__":
    main()
