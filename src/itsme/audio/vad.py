"""
Voice Activity Detection (VAD) and Speech Segmentation Module.
Applies Silero VAD or robust energy-based VAD to produce 2s-15s utterances.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from itsme.utils.logging import get_logger

logger = get_logger("itsme.audio.vad")

def energy_vad(
    audio: np.ndarray,
    sr: int,
    frame_duration_ms: int = 30,
    energy_threshold_db: float = -35.0
) -> list[tuple[int, int]]:
    """
    Fallback robust energy/spectral flux VAD segmentation.
    Returns list of (start_sample, end_sample) speech regions.
    """
    frame_size = int(sr * (frame_duration_ms / 1000.0))
    num_frames = len(audio) // frame_size
    if num_frames == 0:
        return []
        
    threshold = 10 ** (energy_threshold_db / 20.0)
    speech_frames = []
    
    for i in range(num_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        rms = np.sqrt(np.mean(frame**2))
        is_speech = rms >= threshold
        speech_frames.append(is_speech)
        
    # Find contiguous speech segments
    segments = []
    in_speech = False
    start_frame = 0
    
    for idx, active in enumerate(speech_frames):
        if active and not in_speech:
            in_speech = True
            start_frame = idx
        elif not active and in_speech:
            in_speech = False
            segments.append((start_frame * frame_size, idx * frame_size))
            
    if in_speech:
        segments.append((start_frame * frame_size, len(speech_frames) * frame_size))
        
    return segments

def segment_audio_file(
    file_path: str,
    output_dir: str = "data/segments",
    min_duration_s: float = 2.0,
    max_duration_s: float = 15.0,
    padding_s: float = 0.15,
    start_index: int = 1
) -> tuple[list[dict[str, Any]], int]:
    """
    Segment audio file into clean utterances between min_duration_s and max_duration_s.
    """
    path = Path(file_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    audio, sr = sf.read(file_path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    # Use energy VAD for fast, robust, offline segmentation
    raw_segments = energy_vad(audio, sr, energy_threshold_db=-35.0)
    
    if not raw_segments:
        # Fallback to silero if available
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
                onnx=False
            )
            (get_speech_timestamps, _, read_audio, _, _) = utils
            wav_tensor = torch.from_numpy(audio)
            speech_ts = get_speech_timestamps(
                wav_tensor,
                model,
                sampling_rate=sr,
                min_speech_duration_ms=250,
                min_silence_duration_ms=200
            )
            for ts in speech_ts:
                raw_segments.append((int(ts['start']), int(ts['end'])))
        except Exception as e:
            logger.debug(f"Silero VAD notice: {e}")
        
    if not raw_segments:
        # If no speech segments detected, use full audio if within duration range
        total_dur = len(audio) / sr
        if min_duration_s <= total_dur <= max_duration_s:
            raw_segments = [(0, len(audio))]
        else:
            return [], start_index

    # Add padding and merge close / short segments
    pad_samples = int(sr * padding_s)
    processed_segments = []
    
    current_start, current_end = raw_segments[0]
    current_start = max(0, current_start - pad_samples)
    current_end = min(len(audio), current_end + pad_samples)
    
    for s_start, s_end in raw_segments[1:]:
        s_start = max(0, s_start - pad_samples)
        s_end = min(len(audio), s_end + pad_samples)
        
        # Merge if gap is small or segment is shorter than min_duration
        duration_curr = (current_end - current_start) / sr
        gap = (s_start - current_end) / sr
        
        if gap < 0.4 or (duration_curr < min_duration_s and (s_end - current_start) / sr <= max_duration_s):
            current_end = s_end
        else:
            processed_segments.append((current_start, current_end))
            current_start, current_end = s_start, s_end
            
    processed_segments.append((current_start, current_end))

    # Save segments
    segment_records = []
    curr_idx = start_index
    
    for start_s, end_s in processed_segments:
        seg_audio = audio[start_s:end_s]
        duration = len(seg_audio) / sr
        
        # Enforce duration bounds
        if duration < min_duration_s:
            continue
            
        if duration > max_duration_s:
            # Chunk long segment into chunks of max_duration_s
            chunk_len = int(sr * max_duration_s)
            num_chunks = math.ceil(len(seg_audio) / chunk_len)
            for c in range(num_chunks):
                sub_audio = seg_audio[c * chunk_len : (c + 1) * chunk_len]
                sub_dur = len(sub_audio) / sr
                if sub_dur >= min_duration_s:
                    utt_id = f"utt_{curr_idx:06d}"
                    utt_file = out_dir / f"{utt_id}.wav"
                    sf.write(str(utt_file), sub_audio, sr, subtype='PCM_16')
                    segment_records.append({
                        "utt_id": utt_id,
                        "file": utt_file.name,
                        "path": str(utt_file.resolve()),
                        "duration": round(sub_dur, 3),
                        "source_file": path.name
                    })
                    curr_idx += 1
        else:
            utt_id = f"utt_{curr_idx:06d}"
            utt_file = out_dir / f"{utt_id}.wav"
            sf.write(str(utt_file), seg_audio, sr, subtype='PCM_16')
            segment_records.append({
                "utt_id": utt_id,
                "file": utt_file.name,
                "path": str(utt_file.resolve()),
                "duration": round(duration, 3),
                "source_file": path.name
            })
            curr_idx += 1
            
    return segment_records, curr_idx

def segment_all_audio(
    processed_dir: str = "data/processed",
    output_dir: str = "data/segments",
    min_duration_s: float = 2.0,
    max_duration_s: float = 15.0
) -> list[dict[str, Any]]:
    """
    Segment all processed WAV files in data/processed into data/segments.
    """
    proc_path = Path(processed_dir)
    wav_files = sorted(list(proc_path.glob("*.wav")))
    
    logger.info(f"Segmenting {len(wav_files)} files from {processed_dir} -> {output_dir}")
    all_records = []
    idx = 1
    
    for wav_file in wav_files:
        recs, idx = segment_audio_file(
            str(wav_file),
            output_dir=output_dir,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            start_index=idx
        )
        all_records.extend(recs)
        
    logger.info(f"Generated {len(all_records)} speech segments in {output_dir}")
    return all_records
