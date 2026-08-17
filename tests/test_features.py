"""
Unit Tests for Speaker Embeddings and Speech Token Extraction.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from itsme.features.embeddings import extract_speaker_embeddings
from itsme.features.speech_tokens import extract_speech_tokens


@pytest.fixture
def cosyvoice_fixture(tmp_path):
    cosy_dir = tmp_path / "cosyvoice"
    cosy_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample WAV
    wav_file = tmp_path / "utt_000001.wav"
    audio = np.random.normal(0, 0.1, 24000 * 2).astype(np.float32)
    sf.write(str(wav_file), audio, 24000)
    
    # Create Kaldi files
    with open(cosy_dir / "wav.scp", "w") as f:
        f.write(f"utt_000001 {wav_file}\n")
    with open(cosy_dir / "text", "w") as f:
        f.write("utt_000001 Hello world\n")
    with open(cosy_dir / "utt2spk", "w") as f:
        f.write("utt_000001 itsme\n")
        
    return str(cosy_dir)

def test_extract_speaker_embeddings(cosyvoice_fixture):
    res = extract_speaker_embeddings(cosyvoice_dir=cosyvoice_fixture, force_recompute=True)
    assert Path(res["utt2embedding"]).exists()
    assert Path(res["spk2embedding"]).exists()
    
    embs = torch.load(res["utt2embedding"])
    assert "utt_000001" in embs

def test_extract_speech_tokens(cosyvoice_fixture):
    res_path = extract_speech_tokens(cosyvoice_dir=cosyvoice_fixture, force_recompute=True)
    assert Path(res_path).exists()
    
    toks = torch.load(res_path)
    assert "utt_000001" in toks
