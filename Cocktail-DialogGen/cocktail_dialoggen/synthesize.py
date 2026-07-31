"""Qwen3-TTS stage of Cocktail-DialogGen.

Each speaker's utterances are synthesized with a consistent voice:
  * enrollment given  -> clone the provided speaker.wav (Base model).
  * no enrollment     -> design the speaker's FIRST utterance from its gender+style
                         (VoiceDesign model), then clone that designed audio for the
                         speaker's remaining utterances (Base model), keeping identity
                         consistent.
The agent is always the first speaker ("agent"); enrollments[0] is the agent's voice.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from .prompts import other_speaker_keys

BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICEDESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def _too_long(text: str, wav, sr: float, max_spw: float = 1.5, min_dur: float = 5.0) -> bool:
    n_words = max(1, len(text.strip().split()))
    return (len(wav) / sr) > max(min_dur, n_words * max_spw)


def _clone_batch(base_model, texts, voice_prompt, patience: int = 5):
    """Clone a batch of texts with one voice, retrying if any utterance is too long."""
    import torch

    last = None
    for _ in range(patience):
        wavs, sr = base_model.generate_voice_clone(
            text=texts, language=["English"] * len(texts), voice_clone_prompt=voice_prompt
        )
        last = (wavs, sr)
        if not any(_too_long(t, w, sr) for t, w in zip(texts, wavs)):
            return wavs, sr
    return last


def _voice_instruct(profile: dict) -> str:
    gender = (profile.get("gender") or "").strip() or "neutral"
    style = (profile.get("style") or "").strip()
    base = f"A {gender} speaker."
    return f"{base} {style}" if style else base


def _speaker_texts(dialog: List[dict], speaker: str) -> List[tuple[int, str]]:
    """Ordered (turn_index, text) for a given speaker across the dialog."""
    out = []
    for i, turn in enumerate(dialog):
        if speaker == "agent":
            t = turn.get("agent_text")
        else:
            t = turn.get("input_text") if turn.get("input_speaker") == speaker else None
        if t not in (None, "<SILENCE>"):
            out.append((i, t))
    return out


def load_tts_models(device: str = "cuda:0", need_voice_design: bool = False):
    """Load the Base clone model (always) and, if needed, the VoiceDesign model."""
    import torch
    from qwen_tts import Qwen3TTSModel

    base = Qwen3TTSModel.from_pretrained(BASE_MODEL, device_map=device, dtype=torch.bfloat16)
    design = None
    if need_voice_design:
        design = Qwen3TTSModel.from_pretrained(VOICEDESIGN_MODEL, device_map=device, dtype=torch.bfloat16)
    return base, design


def synthesize_conversation(
    conversation: dict,
    output_folder: str,
    *,
    base_model,
    design_model=None,
    enrollments: Optional[List[str]] = None,
    enrollment_texts: Optional[List[str]] = None,
) -> str:
    """Synthesize all utterances of one conversation into output_folder/sources/."""
    import torch

    os.makedirs(os.path.join(output_folder, "sources"), exist_ok=True)
    with open(os.path.join(output_folder, "dialog_logs.json"), "w") as f:
        json.dump(conversation, f, indent=4, ensure_ascii=False)

    voice_profiles = conversation["setting"]["voice_profiles"]
    n_speakers = len(voice_profiles)
    speaker_order = ["agent"] + other_speaker_keys(n_speakers)
    dialog = conversation["dialog"]

    # Map each speaker -> a Base clone prompt (built from enrollment or from a designed first utterance)
    generation_logs = []
    for idx, speaker in enumerate(speaker_order):
        if speaker not in voice_profiles:
            continue
        turn_texts = _speaker_texts(dialog, speaker)
        if not turn_texts:
            continue
        turns = [t for t, _ in turn_texts]
        texts = [t for _, t in turn_texts]

        speaker_dir = os.path.join(output_folder, "sources", speaker)
        os.makedirs(speaker_dir, exist_ok=True)

        if enrollments is not None:
            # Clone directly from the provided enrollment wav.
            ref_audio = enrollments[idx]
            ref_text = enrollment_texts[idx] if enrollment_texts else None
            prompt = base_model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text,
                x_vector_only_mode=ref_text is None,
            )
            wavs, sr = _clone_batch(base_model, texts, prompt)
        else:
            # Design the FIRST utterance from gender+style, then clone it for the rest.
            instruct = _voice_instruct(voice_profiles[speaker])
            torch.manual_seed(0)
            d_wavs, sr = design_model.generate_voice_design(
                text=[texts[0]], instruct=[instruct], language=["English"]
            )
            designed = d_wavs[0]
            wavs = [designed]
            if len(texts) > 1:
                prompt = base_model.create_voice_clone_prompt(
                    ref_audio=(designed, sr), ref_text=texts[0], x_vector_only_mode=False
                )
                rest, sr = _clone_batch(base_model, texts[1:], prompt)
                wavs = [designed] + list(rest)

        for turn, wav, text in zip(turns, wavs, texts):
            sf.write(os.path.join(speaker_dir, f"{turn}.wav"), wav, sr)
            generation_logs.append({
                "speaker": speaker, "turn": turn, "text": text,
                "duration_sec": round(len(wav) / sr, 3),
                "designed": enrollments is None and turn == turns[0],
            })

    with open(os.path.join(output_folder, "generation_logs.json"), "w") as f:
        json.dump(generation_logs, f, indent=4, ensure_ascii=False)
    return output_folder
