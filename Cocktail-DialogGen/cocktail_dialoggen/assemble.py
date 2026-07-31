"""Assembler stage of Cocktail-DialogGen.

Aligns per-utterance sources in time (with the dialog's pauses), mixes the speakers
with the agent, and adds the user's environment recording as background at several
SNR levels. Produces, per mix level:
  noisy_dialog.wav, clean_dialog.wav, background.wav, clean_agent.wav, noisy_other.wav,
  and per-turn prefix clips noisy_dialog_turn_<i>.wav (the audio the agent has heard up
  to and including turn i's input utterance).
Also writes dialog_logs_timed.json with per-turn timing.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

DEFAULT_BUCKETS = {  # name -> (snr_low_db, snr_high_db); mix_clean has no noise
    "mix_3db": (0.0, 3.0),
    "mix_6db": (3.0, 6.0),
    "mix_9db": (6.0, 9.0),
    "mix_12db": (9.0, 12.0),
}


def _normalize(x: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 1e-9:
        return x.astype(np.float32, copy=False)
    return (x * (10 ** (target_db / 20) / rms)).astype(np.float32, copy=False)


def _looped_noise(noise: np.ndarray, length: int) -> Tuple[np.ndarray, int, int]:
    if len(noise) >= length:
        start = random.randint(0, len(noise) - length)
        return noise[start:start + length], start, start + length
    reps = int(np.ceil(length / max(1, len(noise))))
    return np.tile(noise, reps)[:length], 0, length


def _build_streams(dialog, sources_dir, sr, agent_db, speaker_dbs, tts_delay):
    """Concatenate aligned agent/speaker streams; record per-turn input_end (samples)."""
    agent_chunks: List[np.ndarray] = []
    spk_chunks: List[np.ndarray] = []
    active: List[np.ndarray] = []
    timed: List[dict] = []
    cursor = 0

    def add(chunk_a, chunk_s):
        nonlocal cursor
        agent_chunks.append(chunk_a)
        spk_chunks.append(chunk_s)
        cursor += len(chunk_a)

    for i, turn in enumerate(dialog):
        turn_start = cursor
        pause = float(turn.get("pause_before_input", 0) or 0)
        if pause > 0:
            sil = np.zeros(int(pause * sr), dtype=np.float32)
            add(sil, sil)

        input_start = cursor
        speaker = turn.get("input_speaker")
        if speaker and speaker != "noise":
            fp = os.path.join(sources_dir, speaker, f"{i}.wav")
            if os.path.exists(fp):
                y, _ = librosa.load(fp, sr=sr)
                y = _normalize(y, speaker_dbs[speaker])
                add(np.zeros_like(y), y)
                active.append(y)
        input_end = cursor

        timed.append({
            "turn_start_sec": round(turn_start / sr, 2),
            "input_start_sec": round(input_start / sr, 2),
            "input_end_sec": round(input_end / sr, 2),
            "input_end_sample": input_end,
        })

        if turn.get("agent_text") not in (None, "<SILENCE>"):
            ap = os.path.join(sources_dir, "agent", f"{i}.wav")
            if os.path.exists(ap):
                pad = np.zeros(int(tts_delay * sr), dtype=np.float32)
                add(pad, pad)
                ya, _ = librosa.load(ap, sr=sr)
                ya = _normalize(ya, agent_db)
                add(ya, np.zeros_like(ya))
                active.append(ya)

    if not agent_chunks:
        raise RuntimeError("No audio found for this dialog.")
    agent_stream = np.concatenate(agent_chunks)
    spk_stream = np.concatenate(spk_chunks)
    clean = agent_stream + spk_stream
    active_seq = np.concatenate(active) if active else np.zeros_like(clean)
    return agent_stream, spk_stream, clean, active_seq, timed


def _write_mix(mix_dir, clean, background, agent_stream, spk_stream, timed, sr, mix_config):
    os.makedirs(mix_dir, exist_ok=True)
    noisy = clean + background
    noisy_other = spk_stream + background

    peak = float(np.max(np.abs(noisy))) if len(noisy) else 0.0
    scale = 0.99 / peak if peak > 1.0 else 1.0

    outs = {
        "background.wav": background * scale,
        "clean_dialog.wav": clean * scale,
        "noisy_dialog.wav": noisy * scale,
        "clean_agent.wav": agent_stream * scale,
        "noisy_other.wav": noisy_other * scale,
    }
    for name, data in outs.items():
        sf.write(os.path.join(mix_dir, name), data.astype(np.float32), sr)

    # per-turn prefix clips of the noisy mix (up to each turn's input end)
    noisy_scaled = noisy * scale
    for i, t in enumerate(timed):
        end = int(t["input_end_sample"])
        sf.write(os.path.join(mix_dir, f"noisy_dialog_turn_{i}.wav"),
                 noisy_scaled[:end].astype(np.float32), sr)

    with open(os.path.join(mix_dir, "mix_config.json"), "w") as f:
        json.dump(mix_config, f, indent=2)


def assemble_dialog(
    dialog_folder: str,
    environment_audio: str,
    *,
    buckets: Optional[Dict[str, Tuple[float, float]]] = None,
    include_clean: bool = True,
    target_sr: int = 24000,
    agent_target_db: float = -20.0,
    speaker_offset: Tuple[float, float] = (-5.0, 5.0),
    tts_delay: float = 0.2,
    seed: Optional[int] = None,
) -> dict:
    """Assemble one dialog folder (with dialog_logs.json + sources/) into noisy mixes."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    buckets = DEFAULT_BUCKETS if buckets is None else buckets

    with open(os.path.join(dialog_folder, "dialog_logs.json")) as f:
        dialog_logs = json.load(f)
    dialog = dialog_logs["dialog"]
    sources_dir = os.path.join(dialog_folder, "sources")

    # per-speaker levels (agent fixed, others offset)
    speakers = {t.get("input_speaker") for t in dialog if t.get("input_speaker") not in (None, "noise")}
    speaker_dbs = {s: agent_target_db + random.uniform(*speaker_offset) for s in speakers}

    agent_stream, spk_stream, clean, active_seq, timed = _build_streams(
        dialog, sources_dir, target_sr, agent_target_db, speaker_dbs, tts_delay
    )

    # write timed dialog log (drop the internal sample field)
    timed_dialog = json.loads(json.dumps(dialog_logs))
    for turn, t in zip(timed_dialog["dialog"], timed):
        turn["turn_start_sec"] = t["turn_start_sec"]
        turn["input_start_sec"] = t["input_start_sec"]
        turn["input_end_sec"] = t["input_end_sec"]
    with open(os.path.join(dialog_folder, "dialog_logs_timed.json"), "w") as f:
        json.dump(timed_dialog, f, indent=4, ensure_ascii=False)

    noise_raw, _ = librosa.load(environment_audio, sr=target_sr)
    speech_rms = float(np.sqrt(np.mean(active_seq ** 2)))

    mixes = {}
    if include_clean:
        mixes["mix_clean"] = None
    for name, rng in buckets.items():
        mixes[name] = rng

    for name, rng in mixes.items():
        mix_dir = os.path.join(dialog_folder, name)
        if rng is None:  # clean, no background
            background = np.zeros_like(clean)
            cfg = {"snr": None, "noise_file": None, "total_duration": round(len(clean) / target_sr, 2)}
        else:
            snr_db = random.uniform(*rng)
            noise_chunk, s0, s1 = _looped_noise(noise_raw, len(clean))
            noise_chunk = _normalize(noise_chunk)
            cur_rms = float(np.sqrt(np.mean(noise_chunk ** 2)))
            target_noise_rms = speech_rms / (10 ** (snr_db / 20))
            gain = (target_noise_rms / cur_rms) if cur_rms > 1e-9 else 0.0
            background = (noise_chunk * gain).astype(np.float32)
            cfg = {
                "snr": round(float(snr_db), 2),
                "noise_file": os.path.basename(environment_audio),
                "noise_start_sec": round(s0 / target_sr, 2),
                "total_duration": round(len(clean) / target_sr, 2),
            }
        _write_mix(mix_dir, clean, background, agent_stream, spk_stream, timed, target_sr, cfg)

    return {"folder": dialog_folder, "n_turns": len(dialog), "mixes": list(mixes)}
