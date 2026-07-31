"""Access the bundled background recordings (10 per predefined environment label).

Users can pass either a path to their own environment recording, or one of the
predefined labels below (a recording is then picked from backgrounds/<LABEL>/).
"""
from __future__ import annotations

import os
import random
from typing import List, Optional

# The 18 predefined environment labels bundled under ../backgrounds/<LABEL>/
LABELS = [
    "BUS", "CAFE", "CAFETER", "CAR", "FIELD", "HALLWAY", "KITCHEN", "LIVING",
    "MEETING", "METRO", "OFFICE", "PARK", "RESTO", "RIVER", "SQUARE", "STATION",
    "TRAFFIC", "WASHING",
]

_BG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backgrounds")


def background_root() -> str:
    return os.environ.get("COCKTAIL_BACKGROUNDS", _BG_ROOT)


def list_backgrounds(label: str) -> List[str]:
    """List the bundled background recordings for a predefined label."""
    d = os.path.join(background_root(), label.upper())
    if not os.path.isdir(d):
        raise ValueError(
            f"Unknown environment label {label!r}. Known labels: {LABELS}. "
            f"(Or pass a path to your own recording.)"
        )
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".wav"))


def resolve_environment_audio(environment_audio: str, seed: Optional[int] = None) -> str:
    """Return a usable recording path from either a file path or a predefined label."""
    if os.path.isfile(environment_audio):
        return environment_audio
    if environment_audio.upper() in LABELS:
        files = list_backgrounds(environment_audio)
        if not files:
            raise FileNotFoundError(f"No background recordings bundled for label {environment_audio!r}.")
        rng = random.Random(seed)
        return rng.choice(files)
    raise FileNotFoundError(
        f"environment_audio {environment_audio!r} is neither an existing file nor a known label {LABELS}."
    )
