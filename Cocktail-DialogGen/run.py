#!/usr/bin/env python
"""CLI wrapper around cocktail_dialoggen.generate_dialogs().

Example (no enrollments -> voices are designed from each speaker's gender+style):
  export GEMINI_API_KEY=...
  python run.py --environment-audio cafeteria.wav --n-speakers 3 \
      --vibe assisting --names anonymous --n-dialogs 5

Example (with enrollments; first wav is the agent voice, e.g. Chelsie):
  python run.py --environment-audio park.wav \
      --enrollments Chelsie.wav spkB.wav spkC.wav --n-speakers 3
"""
import argparse

from cocktail_dialoggen import generate_dialogs


def main():
    p = argparse.ArgumentParser(description="Cocktail-DialogGen: simulate noisy multi-speaker dialogs.")
    p.add_argument("--environment-audio", required=True, help="Environment recording (also used as background).")
    p.add_argument("--noise-description", default=None, help="One-sentence description; if omitted, Gemini captions the audio.")
    p.add_argument("--n-speakers", type=int, default=3)
    p.add_argument("--vibe", choices=["assisting", "casual"], default="assisting")
    p.add_argument("--names", choices=["named", "anonymous"], default="anonymous")
    p.add_argument("--enrollments", nargs="*", default=None,
                   help="Ordered speaker wavs; first is the agent. Must match --n-speakers. If omitted, voices are designed.")
    p.add_argument("--enrollment-texts", nargs="*", default=None, help="Optional transcripts for the enrollments.")
    p.add_argument("--agent-gender", default="female")
    p.add_argument("--other-genders", nargs="*", default=None, help="Genders for non-agent speakers (len == n_speakers-1).")
    p.add_argument("--n-dialogs", type=int, default=10)
    p.add_argument("--gemini-model", default="gemini-3.1-pro-preview")
    p.add_argument("--gemini-thinking-level", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--gemini-api-key", default=None, help="Falls back to $GEMINI_API_KEY.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = generate_dialogs(
        environment_audio=args.environment_audio,
        noise_description=args.noise_description,
        config={"n_speakers": args.n_speakers, "vibe": args.vibe, "names": args.names},
        speaker_enrollments=args.enrollments,
        enrollment_texts=args.enrollment_texts,
        n_dialogs=args.n_dialogs,
        gemini_model=args.gemini_model,
        gemini_thinking_level=args.gemini_thinking_level,
        gemini_api_key=args.gemini_api_key,
        agent_gender=args.agent_gender,
        other_genders=args.other_genders,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    print(out)


if __name__ == "__main__":
    main()
