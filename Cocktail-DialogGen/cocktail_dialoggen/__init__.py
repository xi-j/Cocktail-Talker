"""Cocktail-DialogGen — simulate noisy multi-speaker dialogs from an environment recording.

Top-level entry point: `generate_dialogs(...)`. See README for the full description.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import List, Optional

from .backgrounds import LABELS, list_backgrounds, resolve_environment_audio  # stdlib-only

__all__ = [
    "generate_dialogs", "caption_environment", "compose_dialogs",
    "LABELS", "list_backgrounds", "resolve_environment_audio",
]


def __getattr__(name):
    # Lazy re-export so importing the package doesn't require every stage's deps
    # (e.g. compose needs google-genai; assemble/synthesize need librosa + qwen_tts).
    if name in ("caption_environment", "compose_dialogs"):
        from . import compose
        return getattr(compose, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

DEFAULT_CONFIG = {"n_speakers": 3, "vibe": "assisting", "names": "anonymous"}


def _slug(text: str, maxlen: int = 60) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip()).strip("-")
    return text[:maxlen] or "dialog"


def _conversation_id(conversation: dict, index: int) -> str:
    setting = conversation.get("setting", {})
    activity = setting.get("activity", f"dialog-{index}")
    names = [p.get("name", "") for p in setting.get("voice_profiles", {}).values()]
    stem = _slug(activity) + "-" + "-".join(_slug(n, 20) for n in names if n)
    suffix = hashlib.md5(json.dumps(conversation, sort_keys=True).encode()).hexdigest()[:8]
    return f"{stem}-{suffix}"


def generate_dialogs(
    environment_audio: str,
    noise_description: Optional[str] = None,
    config: Optional[dict] = None,
    speaker_enrollments: Optional[List[str]] = None,
    n_dialogs: int = 10,
    gemini_model: str = "gemini-3.1-pro-preview",
    gemini_thinking_level: str = "medium",
    gemini_api_key: Optional[str] = None,
    *,
    agent_gender: str = "female",
    other_genders: Optional[List[str]] = None,
    enrollment_texts: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda:0",
    seed: int = 0,
) -> str:
    """Generate noisy multi-speaker dialogs for an acoustic environment.

    Args:
        environment_audio: path to an environment recording (e.g. "cafeteria.wav"),
            also used as the background noise for the mixes.
        noise_description: one-sentence description of the environment. If None, Gemini
            captions `environment_audio`.
        config: {"n_speakers": 3|4, "vibe": "assisting"|"casual", "names": "named"|"anonymous"}.
        speaker_enrollments: ordered list of speaker .wav files; must match n_speakers if given.
            enrollments[0] is the agent voice (e.g. Chelsie). If None, each speaker's voice is
            designed (VoiceDesign) from its gender+style and cloned for consistency.
        n_dialogs: number of dialogs to compose in this call (default 10).
        gemini_model / gemini_thinking_level / gemini_api_key: Gemini configuration. The key
            falls back to $GEMINI_API_KEY and is never hardcoded.
        agent_gender: the agent's gender, prompted explicitly to Gemini.
        other_genders: optional list of genders for the non-agent speakers (len == n_speakers-1).
        enrollment_texts: optional transcripts for the enrollments (improves clone fidelity).
        output_dir: where to write the dialogs. Defaults to runs/<env>-<vibe>-<names>-<n>spk.
        device: CUDA device for Qwen3-TTS.
        seed: base random seed.

    Returns:
        The output directory containing one subfolder per generated dialog.
    """
    from .compose import caption_environment, compose_dialogs
    from .assemble import assemble_dialog
    from .backgrounds import resolve_environment_audio

    # environment_audio may be a file path or one of the predefined labels (e.g. "CAFETER").
    environment_audio = resolve_environment_audio(environment_audio, seed=seed)

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    n_speakers = int(cfg["n_speakers"])
    vibe = str(cfg["vibe"])
    names_policy = str(cfg["names"])

    if speaker_enrollments is not None and len(speaker_enrollments) != n_speakers:
        raise ValueError(
            f"speaker_enrollments has {len(speaker_enrollments)} entries but n_speakers={n_speakers}."
        )
    if other_genders is not None and len(other_genders) != n_speakers - 1:
        raise ValueError(
            f"other_genders has {len(other_genders)} entries but n_speakers-1={n_speakers - 1}."
        )

    # 1) Environment description (caption if not provided).
    if noise_description is None:
        print("[compose] captioning environment recording with Gemini ...", flush=True)
        noise_description = caption_environment(
            environment_audio, api_key=gemini_api_key, model=gemini_model, thinking_level="low"
        )
    print(f"[compose] environment: {noise_description}", flush=True)

    # 2) Compose dialogs with Gemini.
    print(f"[compose] composing {n_dialogs} dialog(s) ...", flush=True)
    conversations = compose_dialogs(
        environment_description=noise_description,
        n_speakers=n_speakers, vibe=vibe, names_policy=names_policy,
        agent_gender=agent_gender, other_genders=other_genders, n_dialogs=n_dialogs,
        api_key=gemini_api_key, model=gemini_model, thinking_level=gemini_thinking_level,
    )
    print(f"[compose] received {len(conversations)} conversation(s).", flush=True)

    if output_dir is None:
        output_dir = os.path.join(
            "runs", f"{_slug(noise_description, 24)}-{vibe}-{names_policy}-{n_speakers}spk"
        )
    os.makedirs(output_dir, exist_ok=True)

    # 3) Load Qwen3-TTS models (VoiceDesign only needed when there are no enrollments).
    from .synthesize import load_tts_models, synthesize_conversation
    need_design = speaker_enrollments is None
    print(f"[tts] loading Qwen3-TTS models (voice_design={need_design}) ...", flush=True)
    base_model, design_model = load_tts_models(device=device, need_voice_design=need_design)

    # 4) Synthesize + assemble each dialog.
    for i, conv in enumerate(conversations):
        conv_id = _conversation_id(conv, i)
        folder = os.path.join(output_dir, conv_id)
        os.makedirs(folder, exist_ok=True)
        print(f"[{i + 1}/{len(conversations)}] {conv_id}", flush=True)

        synthesize_conversation(
            conv, folder,
            base_model=base_model, design_model=design_model,
            enrollments=speaker_enrollments, enrollment_texts=enrollment_texts,
        )
        assemble_dialog(folder, environment_audio, seed=seed + i)

    # 5) Write a manifest of everything produced.
    manifest = {
        "environment_audio": os.path.abspath(environment_audio),
        "environment_description": noise_description,
        "config": cfg,
        "agent_gender": agent_gender,
        "other_genders": other_genders,
        "enrolled": speaker_enrollments is not None,
        "n_dialogs": len(conversations),
        "dialogs": sorted(os.listdir(output_dir)),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[done] wrote {len(conversations)} dialog(s) -> {output_dir}", flush=True)
    return output_dir
