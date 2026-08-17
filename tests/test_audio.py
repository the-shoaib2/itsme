"""
Unit Tests for Audio Validation, Preprocessing, and VAD Segmentation.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from itsme.audio.preprocessor import preprocess_file
from itsme.audio.vad import segment_audio_file
from itsme.audio.validator import inspect_audio_file


@pytest.fixture
def sample_wav(tmp_path):
    wav_file = tmp_path / "sample.wav"
    sr = 24000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Sine wave signal
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(str(wav_file), audio, sr, subtype='PCM_16')
    return str(wav_file)

def test_inspect_audio_file(sample_wav):
    info = inspect_audio_file(sample_wav)
    assert info["duration"] == pytest.approx(3.0, rel=0.1)
    assert info["sample_rate"] == 24000
    assert info["channels"] == 1
    assert info["clipping"] is False
    assert info["quality"] == "good"

def test_preprocess_file(sample_wav, tmp_path):
    out_dir = tmp_path / "processed"
    rec = preprocess_file(sample_wav, output_dir=str(out_dir), target_sr=24000)
    assert rec is not None
    assert Path(rec["processed_path"]).exists()
    assert rec["sample_rate"] == 24000

def test_segment_audio_file(sample_wav, tmp_path):
    out_dir = tmp_path / "segments"
    recs, _next_idx = segment_audio_file(sample_wav, output_dir=str(out_dir), min_duration_s=0.5, max_duration_s=15.0)
    assert len(recs) >= 1
    assert recs[0]["duration"] >= 0.5
