"""
Unit Tests for Speaker-Conditioned Neural Acoustic Model and Training Modules.
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from itsme.training.trainer import (
    MelSpectrogramExtractor,
    SpeakerConditionedAcousticModel,
)


def test_mel_extractor_roundtrip():
    extractor = MelSpectrogramExtractor(sample_rate=24000, n_mels=80)
    audio = torch.randn(24000 * 2)  # 2 seconds
    mel = extractor.wav_to_mel(audio)
    assert mel.shape[1] == 80
    assert mel.shape[0] > 0
    
    wav = extractor.mel_to_wav(mel)
    assert isinstance(wav, np.ndarray)
    assert len(wav) > 0


def test_acoustic_model_forward():
    model = SpeakerConditionedAcousticModel(hidden_dim=64, n_mels=80, spk_dim=192, n_heads=2, num_layers=2)
    tokens = torch.randint(0, 255, (2, 20))
    token_mask = torch.ones((2, 20), dtype=torch.bool)
    spk_embs = torch.randn(2, 192)
    
    pred_mels, pred_spk = model(tokens, token_mask, spk_embs, target_len=100)
    assert pred_mels.shape == (2, 100, 80)
    assert pred_spk.shape == (2, 192)


def test_model_synthesize_mel():
    model = SpeakerConditionedAcousticModel(hidden_dim=64, n_mels=80, spk_dim=192, n_heads=2, num_layers=2)
    tokens = torch.tensor([72, 101, 108, 108, 111]) # "Hello"
    spk_emb = torch.randn(192)
    
    pred_mel = model.synthesize_mel(tokens, spk_emb)
    assert pred_mel.shape[1] == 80
    assert pred_mel.shape[0] >= 50
