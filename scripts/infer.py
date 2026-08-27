#!/usr/bin/env python3
"""
CLI Voice Synthesis Inference Script.
Synthesizes speech from text with personal voice tuning and formant morphing.
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
    parser.add_argument("--voice", default="itsme", choices=["itsme", "prabhat", "andrew", "brian", "guy", "bangla"], help="Voice profile preset")
    parser.add_argument("--pitch", default="+4Hz", help="Pitch shift adjustment (e.g. +4Hz, +10Hz, -5Hz, +0Hz)")
    parser.add_argument("--formant-strength", type=float, default=0.5, help="Vocal tract formant morphing strength (0.0 to 1.0)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech rate multiplier (default 1.0)")
    parser.add_argument("--model", default="models/base", help="Path to base model directory")
    parser.add_argument("--checkpoint", default=None, help="Path to fine-tuned checkpoint")
    parser.add_argument("--reference-audio", default=None, help="Reference audio file for prompt cloning")
    parser.add_argument("--device", default="auto", help="Device (cuda, mps, cpu, auto)")
    args = parser.parse_args()

    engine = CosyVoiceInferenceEngine(
        model_dir=args.model,
        checkpoint_dir=args.checkpoint,
        device=args.device,
        default_voice=args.voice
    )
    
    res = engine.synthesize(
        text=args.text,
        output_path=args.output,
        reference_audio=args.reference_audio,
        voice=args.voice,
        pitch=args.pitch,
        formant_strength=args.formant_strength,
        speed=args.speed
    )
    print(f"Synthesized voice ({res['duration']}s, voice={res.get('voice')}) saved to {res['output_path']}")

if __name__ == "__main__":
    main()
