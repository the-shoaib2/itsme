"""
Audio Preprocessing Pipeline Module.
Converts raw audio to mono 24 kHz WAV with safe normalization and silence trimming.
"""

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from itsme.utils.logging import get_logger, log_stage_event

logger = get_logger("itsme.audio.preprocessor")

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D numpy audio array using scipy polyphase resample."""
    if orig_sr == target_sr:
        return audio
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32)

import math


def normalize_peak(audio: np.ndarray, target_db: float = -20.0, max_peak: float = 0.95) -> np.ndarray:
    """Normalize audio level safely avoiding clipping."""
    if len(audio) == 0:
        return audio
    current_peak = np.max(np.abs(audio))
    if current_peak == 0:
        return audio
        
    target_amplitude = 10 ** (target_db / 20.0)
    scaling = target_amplitude / (np.sqrt(np.mean(audio**2)) + 1e-8)
    
    normalized = audio * scaling
    peak_after = np.max(np.abs(normalized))
    
    if peak_after > max_peak:
        normalized = (normalized / peak_after) * max_peak
        
    return normalized.astype(np.float32)

def trim_leading_trailing_silence(
    audio: np.ndarray, sr: int, threshold_db: float = -40.0, min_silence_ms: int = 100
) -> np.ndarray:
    """Trim leading and trailing silence while preserving internal speech pauses."""
    if len(audio) == 0:
        return audio
        
    threshold = 10 ** (threshold_db / 20.0)
    abs_audio = np.abs(audio)
    non_silent_indices = np.where(abs_audio >= threshold)[0]
    
    if len(non_silent_indices) == 0:
        return audio
        
    start_idx = max(0, non_silent_indices[0] - int(sr * (min_silence_ms / 1000.0)))
    end_idx = min(len(audio), non_silent_indices[-1] + int(sr * (min_silence_ms / 1000.0)))
    
    return audio[start_idx:end_idx]

def preprocess_file(
    input_path: str,
    output_dir: str = "data/processed",
    target_sr: int = 24000,
    target_db: float = -20.0
) -> dict[str, Any] | None:
    """
    Process single audio file: load, mono convert, resample to 24kHz, trim silence, normalize, save WAV.
    """
    in_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / f"{in_path.stem}.wav"
    
    try:
        try:
            data, orig_sr = sf.read(str(in_path), dtype='float32')
        except Exception:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(str(in_path))
            orig_sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            max_val = float(1 << (seg.sample_width * 8 - 1))
            data = samples / max_val
            if seg.channels > 1:
                data = data.reshape((-1, seg.channels))

        # 1. Convert to mono
        if data.ndim > 1:
            audio = np.mean(data, axis=1)
        else:
            audio = data
            
        # 2. Resample to target_sr
        audio = resample_audio(audio, orig_sr, target_sr)
        
        # 3. Trim leading and trailing silence
        audio = trim_leading_trailing_silence(audio, target_sr)
        
        # 4. Safe normalization
        audio = normalize_peak(audio, target_db=target_db)
        
        # 5. Check duration
        duration = len(audio) / target_sr
        if duration < 0.5:
            logger.warning(f"Skipping too short audio after processing ({duration:.2f}s): {in_path.name}")
            return None
            
        # Save 24kHz float32 mono WAV
        sf.write(str(out_file), audio, target_sr, subtype='PCM_16')
        
        return {
            "original_file": in_path.name,
            "processed_file": out_file.name,
            "processed_path": str(out_file.resolve()),
            "sample_rate": target_sr,
            "duration": round(duration, 3),
            "channels": 1
        }
    except Exception as e:
        logger.error(f"Failed to preprocess audio {in_path}: {e}")
        return None

def prepare_all_audio(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    target_sr: int = 24000,
    supported_exts: tuple[str, ...] = (".wav", ".flac", ".mp3", ".m4a")
) -> list[dict[str, Any]]:
    """
    Preprocess all audio files from data/raw/ to data/processed/.
    """
    raw_path = Path(raw_dir)
    files = [
        p for p in raw_path.rglob("*")
        if p.suffix.lower() in supported_exts and not p.name.startswith(".")
    ]
    
    logger.info(f"Preprocessing {len(files)} files from {raw_dir} -> {output_dir}")
    processed_records = []
    
    for f in files:
        rec = preprocess_file(str(f), output_dir=output_dir, target_sr=target_sr)
        if rec:
            processed_records.append(rec)
            log_stage_event(
                logger,
                stage="audio.preprocessing",
                status="processed",
                file_id=rec["processed_file"],
                duration=rec["duration"]
            )
            
    logger.info(f"Successfully preprocessed {len(processed_records)} files.")
    return processed_records
