"""Gemini stage of Cocktail-DialogGen: caption the environment (optional) and
compose multi-speaker dialog logs.

The API key is taken from the `api_key` argument or the GEMINI_API_KEY environment
variable. It is never hardcoded.
"""
from __future__ import annotations

import json
import mimetypes
import os
from typing import List, Optional

from .prompts import build_composer_prompts, caption_prompt


def _make_client(api_key: Optional[str]):
    from google import genai  # lazy import so TTS-only use doesn't need google-genai

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "No Gemini API key provided. Pass gemini_api_key=... or set GEMINI_API_KEY."
        )
    return genai.Client(api_key=key)


def _thinking_config(thinking_level: str):
    from google.genai import types

    # Newer SDKs use thinking_level ('low'|'medium'|'high'); tolerate older ones.
    try:
        return types.ThinkingConfig(thinking_level=thinking_level)
    except (TypeError, ValueError):
        return None


def caption_environment(
    audio_path: str,
    *,
    api_key: Optional[str] = None,
    model: str = "gemini-3.1-pro-preview",
    thinking_level: str = "low",
) -> str:
    """Describe an environment recording in one sentence with Gemini (multimodal)."""
    from google.genai import types

    client = _make_client(api_key)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    mime = mimetypes.guess_type(audio_path)[0] or "audio/wav"

    cfg_kwargs = {}
    tc = _thinking_config(thinking_level)
    if tc is not None:
        cfg_kwargs["thinking_config"] = tc

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type=mime),
            caption_prompt(),
        ],
        config=types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None,
    )
    return (response.text or "").strip().strip('"')


def _clean_json_string(s: str) -> str:
    """Strip Markdown code fences and surrounding whitespace."""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _parse_conversations(text: str) -> List[dict]:
    """Parse a Gemini response into a list of {'setting', 'dialog'} conversations."""
    data = json.loads(_clean_json_string(text))
    if isinstance(data, dict):
        # tolerate a single conversation or a wrapper like {"conversations": [...]}
        if "setting" in data and "dialog" in data:
            return [data]
        for v in data.values():
            if isinstance(v, list):
                return v
        raise ValueError("Unexpected dict response shape from Gemini.")
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of conversations, got {type(data)}")
    return data


def compose_dialogs(
    *,
    environment_description: str,
    n_speakers: int,
    vibe: str,
    names_policy: str,
    agent_gender: str = "female",
    other_genders: Optional[List[str]] = None,
    n_dialogs: int = 10,
    api_key: Optional[str] = None,
    model: str = "gemini-3.1-pro-preview",
    thinking_level: str = "medium",
) -> List[dict]:
    """Ask Gemini to compose `n_dialogs` conversations; return parsed conversation dicts."""
    from google.genai import types

    client = _make_client(api_key)
    system_prompt, user_prompt = build_composer_prompts(
        n_speakers=n_speakers,
        environment_description=environment_description,
        vibe=vibe,
        names_policy=names_policy,
        agent_gender=agent_gender,
        other_genders=other_genders,
        n_dialogs=n_dialogs,
    )

    cfg_kwargs = {"system_instruction": system_prompt}
    tc = _thinking_config(thinking_level)
    if tc is not None:
        cfg_kwargs["thinking_config"] = tc

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(**cfg_kwargs),
    )
    conversations = _parse_conversations(response.text)

    # Post-process: enforce anonymity constraint would go here if names_policy == "anonymous".
    return conversations
