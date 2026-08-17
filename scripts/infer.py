#!/usr/bin/env python3
"""
CLI Voice Synthesis Inference Script.
Synthesizes speech from text using trained CosyVoice 3 model.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.inference.engine import CosyVoiceInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Synthesize personal voice audio from text.")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", default="outputs/test.wav", help="Output WAV file path")
    parser.add_argument("--model", default="models/base", help="Path to base model directory")
    parser.add_argument("--checkpoint", default=None, help="Path to fine-tuned checkpoint")
    parser.add_argument("--reference-audio", default=None, help="Reference audio file for prompt cloning")
    parser.add_argument("--device", default="auto", help="Device (cuda, mps, cpu, auto)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech rate (default 1.0)")
    args = parser.parse_args()

    engine = CosyVoiceInferenceEngine(
        model_dir=args.model,
        checkpoint_dir=args.checkpoint,
        device=args.device
    )
    
    res = engine.synthesize(
        text=args.text,
        output_path=args.output,
        reference_audio=args.reference_audio,
        speed=args.speed
    )
    print(f"Synthesized voice ({res['duration']}s) saved to {res['output_path']}")

if __name__ == "__main__":
    main()
