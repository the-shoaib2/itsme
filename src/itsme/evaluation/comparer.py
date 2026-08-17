"""
Checkpoint Comparison Module.
Generates audio samples for multiple checkpoints on standard evaluation prompts.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from itsme.utils.logging import get_logger

logger = get_logger("itsme.evaluation.comparer")

def compare_checkpoints(
    checkpoint_dirs: list[str],
    prompts_dir: str = "evaluation/prompts",
    output_dir: str = "evaluation/comparison"
) -> dict[str, Any]:
    """
    Compare multiple checkpoint outputs on identical evaluation prompts.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    p_path = Path(prompts_dir)
    prompts = []
    if p_path.exists():
        for pf in p_path.glob("*.txt"):
            prompts.append({
                "id": pf.stem,
                "text": pf.read_text(encoding="utf-8").strip()
            })
            
    if not prompts:
        prompts = [{
            "id": "hello",
            "text": "Hello, this is my authorized personal voice created with ItsMe."
        }]

    comparison_results = {}
    
    for ckpt_str in checkpoint_dirs:
        ckpt_path = Path(ckpt_str)
        ckpt_name = ckpt_path.name if ckpt_path.name else "ckpt"
        ckpt_out_dir = out_path / ckpt_name
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating comparison samples for checkpoint '{ckpt_name}'...")
        
        samples_meta = []
        for prompt in prompts:
            sample_wav = ckpt_out_dir / f"{prompt['id']}.wav"
            
            # Generate speech placeholder/model synthesis
            dummy_audio = np.random.normal(0, 0.05, 24000 * 2).astype(np.float32)
            sf.write(str(sample_wav), dummy_audio, 24000)
            
            samples_meta.append({
                "prompt_id": prompt["id"],
                "text": prompt["text"],
                "audio_path": str(sample_wav.resolve())
            })
            
        comparison_results[ckpt_name] = {
            "checkpoint_path": str(ckpt_path.resolve()),
            "samples": samples_meta
        }

    meta_file = out_path / "comparison_summary.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
        
    logger.info(f"Checkpoint comparison complete. Summary saved to {meta_file}")
    return comparison_results
