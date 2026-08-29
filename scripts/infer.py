#!/usr/bin/env bash
"true" '''\'
exec "$(dirname "$0")/../.venv/bin/python" "$0" "$@"
'''
# -*- coding: utf-8 -*-
"""
CLI Voice Synthesis Inference Script.
Synthesizes speech from text using authentic neural voice cloning or fast vocal formant synthesis.
Default Language: Bangla (bn-BD-PradeepNeural).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.inference.engine import CosyVoiceInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Synthesize personal voice audio from text. Default is Bangla.")
    parser.add_argument("--text", required=True, help="Text to synthesize (Bangla or English)")
    parser.add_argument("--output", default="outputs/bangla_output.wav", help="Output WAV file path")
    parser.add_argument("--mode", default="fast", choices=["fast", "neural", "clone"], help="Synthesis mode: 'fast' for calibrated vocal morphing, 'neural' for fine-tuned acoustic model, 'clone' for zero-shot cloning")
    parser.add_argument("--voice", default="bangla", choices=["bangla", "bangla_male", "bangla_female", "bashkar", "pradeep", "nabanita", "tanishaa", "itsme", "prabhat", "andrew", "brian", "guy"], help="Voice preset (default: bangla)")
    parser.add_argument("--pitch", default="-22Hz", help="Pitch adjustment calibrated to speaker F0 (e.g. -22Hz, -15Hz, +0Hz)")
    parser.add_argument("--formant-strength", type=float, default=0.85, help="Vocal tract formant morphing strength (0.0 to 1.0)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech rate multiplier (default 1.0)")
    parser.add_argument("--reference-audio", default=None, help="Reference audio file for cloning prompt")
    parser.add_argument("--reference-text", default=None, help="Reference transcript for cloning prompt")
    parser.add_argument("--nfe-steps", type=int, default=16, help="ODE diffusion steps for neural cloning (default 16)")
    parser.add_argument("--model", default="models/base", help="Path to base model directory")
    parser.add_argument("--device", default="auto", help="Device (cuda, mps, cpu, auto)")
    
    # Preprocess sys.argv to handle negative pitch values like --pitch -14Hz
    argv = list(sys.argv[1:])
    for i, arg in enumerate(argv):
        if arg == "--pitch" and i + 1 < len(argv) and argv[i+1].startswith("-") and not argv[i+1].startswith("--"):
            argv[i] = f"--pitch={argv[i+1]}"
            argv.pop(i+1)
            break
            
    args = parser.parse_args(argv)

    engine = CosyVoiceInferenceEngine(
        model_dir=args.model,
        device=args.device,
        default_voice=args.voice
    )
    
    res = engine.synthesize(
        text=args.text,
        output_path=args.output,
        reference_audio=args.reference_audio,
        reference_text=args.reference_text,
        mode=args.mode,
        voice=args.voice,
        pitch=args.pitch,
        formant_strength=args.formant_strength,
        speed=args.speed,
        nfe_steps=args.nfe_steps
    )
    print(f"Synthesized ({res['duration']}s, voice={res.get('voice', 'neural_clone')}) saved to: {res['output_path']}")

if __name__ == "__main__":
    main()
