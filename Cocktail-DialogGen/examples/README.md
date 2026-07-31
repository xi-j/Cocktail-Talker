# Cocktail-DialogGen — example conversations

Ten example turns produced by **Cocktail-DialogGen** (the same samples shown on the demo
page), used to test **[Cocktail-Talker](../../Cocktail-Talker)**. They span both
`test_seen`/`test_unseen` environments, 3/4 speakers, `assisting`/`casual` modes, and all
three turn actions: **5× respond, 4× listen, 1× ignore**.

## Layout

```
examples/
├── examples.json                         # index: one record per example (see below)
└── <NN>_<ENV>_<mode>_<n>spk_<action>/
    ├── input.wav     # the noisy multi-party mixture the model hears (prefix up to the current turn)
    ├── dialog.json   # the full generated dialog log: speaker metadata, per-turn text / action / timing
    └── sample.json   # this turn's prompt + oracle answer + condition tags
```

## `examples.json` / `sample.json` fields

| Field | Meaning |
|---|---|
| `id` | example directory name |
| `input_audio` | path to `input.wav` (relative to `examples/`) |
| `metadata_prompt` | the text prompt given to the model (speakers' metadata + instruction) |
| `oracle_action` | ground-truth turn action: `<\|respond\|>` / `<\|listen\|>` / `<\|ignore\|>` |
| `oracle_text` | ground-truth agent response text (empty for listen/ignore) |
| `split` | `test_seen` or `test_unseen` |
| `environment`, `mode`, `n_speakers`, `snr` | condition tags |
| `turn_index` | which turn of the dialog this input clip ends at |

The agent's voice in the dataset is Qwen2.5-Omni's "Chelsie", synthesized with Qwen3-TTS;
other speakers are sampled from LibriTTS. Background sound is mixed at the listed SNR.
