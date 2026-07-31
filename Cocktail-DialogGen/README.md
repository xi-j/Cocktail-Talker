# Cocktail-DialogGen

**An LLM + TTS pipeline that simulates realistic, noisy, multi-speaker conversations** for
training and evaluating [Cocktail-Talker](../Cocktail-Talker). Given an acoustic environment,
it composes multi-party dialogs (with per-turn agent actions), synthesizes each speaker's
speech, and mixes everything into noisy audio at several SNR levels.

## Top-level interface

```python
from cocktail_dialoggen import generate_dialogs

generate_dialogs(
    environment_audio = "cafeteria.wav",       # (1) an environment recording (also the background);
                                                #     OR a predefined label like "CAFETER" (see backgrounds/)
    noise_description = None,                   # (2) 1-sentence desc; if None, Gemini captions the audio
    config = {"n_speakers": 3,                  # (3) 3 or 4
              "vibe": "assisting",              #     "assisting" | "casual"
              "names": "anonymous"},            #     "named" | "anonymous"
    speaker_enrollments = None,                 # (4) ordered wavs; [0]=agent (e.g. Chelsie).
                                                #     if None -> design each voice, then clone it
    n_dialogs = 10,                             # (5) dialogs to generate this call
    gemini_model = "gemini-3-pro-preview",      # (6)
    gemini_thinking_level = "medium",           # (7)
    gemini_api_key = None,                      # (8) falls back to $GEMINI_API_KEY (never hardcoded)
    agent_gender = "female",                    #     agent gender, prompted explicitly to Gemini
    other_genders = None,                       #     optional genders for the other speakers
    output_dir = None,
)
```

CLI equivalent:

```bash
export GEMINI_API_KEY=...
python run.py --environment-audio cafeteria.wav --n-speakers 3 \
    --vibe assisting --names anonymous --n-dialogs 5
```

## Pipeline

1. **Describe** — if `noise_description` is None, Gemini captions `environment_audio` in one sentence.
2. **Compose** (`compose.py`) — Gemini composes `n_dialogs` conversations for that environment and
   config. Each turn is labelled with one of three agent actions: **`<RESPOND>` / `<LISTEN>` / `<IGNORE>`**
   (the prompt still requires at least one addressee *switch*, which is expressed as a `<RESPOND>`).
3. **Synthesize** (`synthesize.py`) — Qwen3-TTS renders every utterance, keeping each speaker's
   voice consistent:
   - **enrollments given** → clone the provided `speaker.wav` (agent = `enrollments[0]`).
   - **no enrollments** → *design* each speaker's **first** utterance from its gender + style
     ([Qwen3-TTS-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)),
     then *clone* that designed audio (Qwen3-TTS Base) for the speaker's remaining utterances.
4. **Assemble** (`assemble.py`) — aligns the utterances in time with the dialog's pauses, mixes the
   speakers with the agent, and adds `environment_audio` as background at several SNRs.

## Output layout

```
<output_dir>/
├── manifest.json
└── <activity>-<names>-<hash>/
    ├── dialog_logs.json         # composed dialog (setting + per-turn action/text/pause)
    ├── dialog_logs_timed.json   # + per-turn timing (turn_start_sec / input_start_sec / input_end_sec)
    ├── generation_logs.json     # per-utterance TTS log
    ├── sources/<speaker>/<turn>.wav   # per-utterance audio
    └── mix_clean|mix_3db|mix_6db|mix_9db|mix_12db/
        ├── noisy_dialog.wav            # full noisy mixture
        ├── clean_dialog.wav, background.wav, clean_agent.wav, noisy_other.wav
        └── noisy_dialog_turn_<i>.wav   # prefix the agent has heard up to turn i (Cocktail-Talker input)
```

The `noisy_dialog_turn_<i>.wav` clips are exactly the input format consumed by
[Cocktail-Talker](../Cocktail-Talker); see `examples/` for 10 such ready-made samples.

## Predefined environments

`environment_audio` accepts either a path to your own recording or one of **18 predefined
labels**, each with 10 bundled background recordings under [`backgrounds/`](backgrounds/):

```
BUS CAFE CAFETER CAR FIELD HALLWAY KITCHEN LIVING MEETING
METRO OFFICE PARK RESTO RIVER SQUARE STATION TRAFFIC WASHING
```

`generate_dialogs(environment_audio="CAFETER", ...)` picks a bundled CAFETER recording. Add your
own clips under `backgrounds/<LABEL>/` or point `$COCKTAIL_BACKGROUNDS` elsewhere. **Note:** the
bundled clips are derived from Freesound — see [`backgrounds/README.md`](backgrounds/README.md) for
the licensing caveat.

## Setup

Tested on Python 3.12, CUDA 12.4. One env runs the whole pipeline (Gemini API client +
Qwen3-TTS). The no-enrollment path loads two 1.9 B TTS models (Base + VoiceDesign), ~8 GB VRAM.

```bash
conda create -n cocktail-dialoggen python=3.12 -y && conda activate cocktail-dialoggen
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Notes

- **Agent voice.** With enrollments, `enrollments[0]` is the agent (the paper uses Qwen2.5-Omni's
  "Chelsie" — see `../Cocktail-Talker/assets/Chelsie.wav`). Without enrollments, the agent voice is
  designed from `agent_gender` + the agent's Gemini-assigned style.
- **API key.** Provide it via `gemini_api_key=...` or `export GEMINI_API_KEY=...`. It is never
  written to disk or embedded in code.
- **Actions.** The agent's turn actions are `<RESPOND>`, `<LISTEN>`, `<IGNORE>`. For `<LISTEN>` and
  `<IGNORE>`, `agent_text` is `null` (the agent stays silent).
