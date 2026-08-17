"""
Dataset Audio Quality Filtering Module.
Evaluates segments and produces statistics and dataset_quality.json.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from itsme.utils.logging import get_logger

logger = get_logger("itsme.audio.filter")

def filter_and_report_segments(
    segments_dir: str = "data/segments",
    output_report_path: str = "data/reports/dataset_quality.json",
    min_duration_s: float = 2.0,
    max_duration_s: float = 15.0,
    min_rms: float = 0.005,
    max_silence_ratio: float = 0.35
) -> dict[str, Any]:
    """
    Quality filter audio segments and generate dataset quality statistics report.
    """
    seg_path = Path(segments_dir)
    wav_files = sorted(list(seg_path.glob("*.wav")))
    
    total_files = len(wav_files)
    accepted_files = []
    rejected_files = []
    
    durations = []
    
    for f in wav_files:
        try:
            audio, sr = sf.read(str(f), dtype='float32')
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
                
            dur = len(audio) / sr
            peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
            rms = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0
            
            silence_samples = np.sum(np.abs(audio) < 0.01)
            silence_ratio = float(silence_samples / len(audio)) if len(audio) > 0 else 1.0
            
            rejection_reasons = []
            if dur < min_duration_s:
                rejection_reasons.append(f"duration too short ({dur:.2f}s < {min_duration_s}s)")
            if dur > max_duration_s:
                rejection_reasons.append(f"duration too long ({dur:.2f}s > {max_duration_s}s)")
            if rms < min_rms:
                rejection_reasons.append(f"low RMS volume ({rms:.6f} < {min_rms})")
            if silence_ratio > max_silence_ratio:
                rejection_reasons.append(f"excessive silence ratio ({silence_ratio:.2f} > {max_silence_ratio})")
            if peak >= 0.999:
                rejection_reasons.append("severe clipping")
                
            if rejection_reasons:
                rejected_files.append({
                    "utt_id": f.stem,
                    "path": str(f.resolve()),
                    "duration": round(dur, 3),
                    "reasons": rejection_reasons
                })
            else:
                accepted_files.append({
                    "utt_id": f.stem,
                    "path": str(f.resolve()),
                    "duration": round(dur, 3),
                    "rms": round(rms, 6),
                    "peak": round(peak, 4)
                })
                durations.append(dur)
        except Exception as e:
            rejected_files.append({
                "utt_id": f.stem,
                "path": str(f.resolve()),
                "duration": 0.0,
                "reasons": [f"corrupted audio: {e}"]
            })

    total_dur = sum(durations)
    avg_dur = total_dur / len(durations) if durations else 0.0
    min_dur = min(durations) if durations else 0.0
    max_dur = max(durations) if durations else 0.0

    report = {
        "total_files": total_files,
        "accepted": len(accepted_files),
        "rejected": len(rejected_files),
        "total_duration_seconds": round(total_dur, 2),
        "total_duration_hours": round(total_dur / 3600.0, 3),
        "average_duration_seconds": round(avg_dur, 2),
        "minimum_duration_seconds": round(min_dur, 2),
        "maximum_duration_seconds": round(max_dur, 2),
        "accepted_segments": accepted_files,
        "rejected_segments": rejected_files
    }
    
    out_path = Path(output_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
        
    logger.info(
        f"Dataset filtering complete: {len(accepted_files)}/{total_files} segments accepted. "
        f"Report saved to {output_report_path}"
    )
    return report
