"""
Audio Quality & Integrity Validation Module.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from itsme.utils.exceptions import AudioProcessingError
from itsme.utils.logging import get_logger, log_stage_event

logger = get_logger("itsme.audio.validator")

def inspect_audio_file(file_path: str) -> dict[str, Any]:
    """
    Inspect single audio file for duration, sample rate, channels, clipping, RMS, silence ratio, quality.
    """
    path = Path(file_path)
    if not path.exists():
        raise AudioProcessingError(f"Audio file does not exist: {file_path}")
        
    try:
        try:
            info = sf.info(file_path)
            data, samplerate = sf.read(file_path, dtype='float32')
            duration = float(info.duration)
        except Exception:
            # Fallback to pydub for m4a/aac decoding via ffmpeg
            from pydub import AudioSegment
            seg = AudioSegment.from_file(file_path)
            samplerate = seg.frame_rate
            channels = seg.channels
            duration = len(seg) / 1000.0
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            max_val = float(1 << (seg.sample_width * 8 - 1))
            data = samples / max_val
            if channels > 1:
                data = data.reshape((-1, channels))

        if data.ndim == 1:
            channels = 1
            audio_data = data
        else:
            channels = data.shape[1]
            audio_data = np.mean(data, axis=1)
        
        # Peak & clipping check
        peak = float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0.0
        clipping = peak >= 0.999
        
        # RMS energy calculation
        rms = float(np.sqrt(np.mean(audio_data**2))) if len(audio_data) > 0 else 0.0
        
        # Silence ratio estimation (below -40 dBFS threshold)
        if len(audio_data) > 0:
            abs_audio = np.abs(audio_data)
            silence_samples = np.sum(abs_audio < 0.01) # ~ -40dB
            silence_ratio = float(silence_samples / len(audio_data))
        else:
            silence_ratio = 1.0
            
        # Quality assessment heuristic
        quality = "good"
        reasons = []
        if clipping:
            quality = "warning"
            reasons.append("clipping detected")
        if rms < 0.005:
            quality = "warning"
            reasons.append("extremely low volume (RMS < 0.005)")
        if silence_ratio > 0.6:
            quality = "warning"
            reasons.append("high silence ratio (> 60%)")
        if duration < 1.0:
            quality = "warning"
            reasons.append("too short (< 1s)")
        if len(reasons) >= 2 or rms == 0.0:
            quality = "poor"
            
        return {
            "file": path.name,
            "path": str(path.resolve()),
            "duration": round(duration, 3),
            "sample_rate": samplerate,
            "channels": channels,
            "peak": round(peak, 4),
            "rms": round(rms, 6),
            "clipping": clipping,
            "silence_ratio": round(silence_ratio, 3),
            "quality": quality,
            "issues": reasons
        }
    except Exception as e:
        logger.error(f"Corrupted or unreadable audio file {file_path}: {e}")
        return {
            "file": path.name,
            "path": str(path.resolve()),
            "duration": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "clipping": False,
            "rms": 0.0,
            "silence_ratio": 1.0,
            "quality": "corrupted",
            "issues": [str(e)]
        }

def validate_audio_dir(
    raw_dir: str,
    output_report_path: str = "data/reports/audio_quality.json",
    supported_exts: tuple[str, ...] = (".wav", ".flac", ".mp3", ".m4a")
) -> list[dict[str, Any]]:
    """
    Inspect all audio files in raw directory and generate quality report JSON.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise AudioProcessingError(f"Raw audio directory does not exist: {raw_dir}")
        
    audio_files = [
        p for p in raw_path.rglob("*")
        if p.suffix.lower() in supported_exts and not p.name.startswith(".")
    ]
    
    logger.info(f"Found {len(audio_files)} raw audio files in {raw_dir}")
    reports = []
    
    for f in audio_files:
        report = inspect_audio_file(str(f))
        reports.append(report)
        log_stage_event(
            logger,
            stage="audio.validation",
            status=report["quality"],
            file_id=report["file"],
            duration=report["duration"]
        )
        
    # Write report
    report_path = Path(output_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as out:
        json.dump(reports, out, indent=2)
        
    logger.info(f"Audio quality report saved to {output_report_path}")
    return reports
