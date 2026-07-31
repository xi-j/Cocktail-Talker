# Cocktail-Talker

**Multi-speaker spoken dialog modeling in noisy social environments.**

Cocktail-Talker is a speech LLM built on [`Qwen2.5-Omni-7B`](https://huggingface.co/Qwen/Qwen2.5-Omni-7B).
In a noisy, multi-party conversation it decides, at each turn, **whether and how to
participate** by emitting one of three *turn-action tokens*:

| Action token | Behavior |
|---|---|
| `<\|respond\|> <response text>` | The agent speaks (text → rendered to speech). |
| `<\|listen\|>` | The agent stays **silent**, attending to a conversation directed at others. |
| `<\|ignore\|>` | The agent stays **silent**, disregarding irrelevant speech / background noise. |

This directory releases the Cocktail-Talker checkpoint from the paper as a single LoRA
adapter on top of stock Qwen2.5-Omni-7B, with a runnable inference script and notebook.

The example conversations it runs on come from our data pipeline,
[**Cocktail-DialogGen**](../Cocktail-DialogGen) — see `../Cocktail-DialogGen/examples`.

---

## Two stages

Cocktail-Talker **decides the turn action and writes the response text**. The agent's
**speech** is then rendered from that text by **Qwen3-TTS** with a cloned "Chelsie" voice. 
The two stages use different environments (Qwen3-TTS needs a newer `transformers`), so they are separate:

> **Why Qwen3-TTS instead of Qwen2.5-Omni's own talker?** Finetuning the thinker leaves its
> hidden states unrecognizable to the frozen talker, so if you want speech output, use
> Qwen3-TTS instead — at ~1.9 B params it is only about **40% larger** than the talker it
> replaces (~1.35 B).

| Stage | Script | Env | Output |
|---|---|---|---|
| 1. Turn action + response text | `inference.py` / `inference.ipynb` | `requirements.txt` | `outputs/predictions.json` |
| 2. Speech synthesis (optional) | `synthesize_tts.py` | `requirements-tts.txt` | `outputs/<id>.wav` |

---

## Contents

```
Cocktail-Talker/
├── inference.py            # Stage 1: turn action + response text  -> outputs/predictions.json
├── inference.ipynb         # identical content as a notebook
├── synthesize_tts.py       # Stage 2: Qwen3-TTS speech for <|respond|> turns (Chelsie voice)
├── requirements.txt        # Stage 1 deps (Qwen2.5-Omni)
├── requirements-tts.txt    # Stage 2 deps (Qwen3-TTS)
├── processor/              # tokenizer + feature-extractor config (incl. the 3 action tokens)
├── adapters/cocktail_lora/ # the released Cocktail-Talker LoRA adapter
├── assets/Chelsie.wav      # reference clip for the agent's TTS voice
└── outputs/                # predictions.json and synthesized speech land here
```

The **base Qwen2.5-Omni-7B weights are *not* included** — they are downloaded from the
Hugging Face Hub on first run (~16 GB). Only the ~2.4 GB LoRA adapter is shipped here.

---

## Setup & run — Stage 1 (turn action + text)

Tested on Python 3.12, CUDA 12.4, NVIDIA A40/L40 (≈20 GB VRAM with the talker disabled).

```bash
conda create -n cocktail-talker python=3.12 -y
conda activate cocktail-talker

# PyTorch (CUDA 12.4 build)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# everything else (Qwen2.5-Omni transformers is pinned to a specific commit)
pip install -r requirements.txt

# run the 10 examples
python inference.py --device cuda:0
```

Expected (10/10 correct turn actions; predictions saved to `outputs/predictions.json`):

```
[OK ] 01_WASHING_assisting_4spk_respond
        oracle : <|respond|>  'Yes, that table is fine. Make sure they are separated ...'
        pred   : <|respond|>  'Please place them directly into the mesh hampers. ...'
...
[OK ] 08_MEETING_casual_4spk_ignore
        oracle : <|ignore|>   ''
        pred   : <|ignore|>   ''
...
Action accuracy vs oracle: 10/10
```

`inference.ipynb` has the same logic.

## Setup & run — Stage 2 (speech, optional)

```bash
conda create -n cocktail-tts python=3.12 -y
conda activate cocktail-tts
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-tts.txt

python synthesize_tts.py     # reads outputs/predictions.json -> outputs/<id>.wav
```

---

## Configuration

Defaults can be overridden via environment variables (or CLI flags):

| Variable | Default | Meaning |
|---|---|---|
| `CT_BASE_MODEL` | `Qwen/Qwen2.5-Omni-7B` | base weights (HF id or local path) |
| `CT_MERGED_ADAPTER` | `adapters/cocktail_lora` | Cocktail-Talker LoRA adapter |
| `CT_PROCESSOR` | `processor` | tokenizer / feature-extractor |
| `CT_EXAMPLES` | `../Cocktail-DialogGen/examples` | input turns |
| `CT_OUTPUT` | `outputs` | predictions + WAVs |
| `CT_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | TTS model (stage 2) |
| `CT_TTS_REF_AUDIO` | `assets/Chelsie.wav` | agent voice reference (stage 2) |
| `HF_HOME` | — | Hugging Face cache location |

---

## Input / output format

Each turn, the model receives a **text prompt** (speakers' metadata) and an **audio prompt**
(the full noisy mixture of the conversation up to and including the current input utterance):

```
Speakers' Metadata: {'voice_profiles': {'agent': {'gender': ..., 'name': ..., 'role': ...},
                                         'speaker_b': {...}, ...}, 'mode': 'assisting' | 'casual'}
<audio> Generate the next response of the agent speaker.
```

It outputs one action token, followed by response text only for `<|respond|>`.
See `../Cocktail-DialogGen/examples/examples.json` for the 10 worked examples.
