"""
Dataset Validation Module.
Verifies complete dataset integrity and non-zero exit code on failure.
"""

import json
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from itsme.utils.exceptions import DatasetValidationError
from itsme.utils.logging import get_logger

logger = get_logger("itsme.dataset.validator")

def validate_dataset(
    manifests_dir: str = "data/manifests",
    cosyvoice_dir: str = "data/cosyvoice",
    target_sample_rate: int = 24000
) -> dict[str, Any]:
    """
    Validate dataset for missing files, leakage, duration, sample rate, embeddings, and tokens.
    Raises DatasetValidationError if any critical check fails.
    """
    errors = []
    warnings = []
    
    man_path = Path(manifests_dir)
    cosy_path = Path(cosyvoice_dir)
    
    splits = ["train", "validation", "test"]
    split_utts: dict[str, set[str]] = {}
    
    all_audio_paths = set()
    total_utterances = 0
    
    # 1. Manifests & Leakage Checks
    for split in splits:
        m_file = man_path / f"{split}.jsonl"
        split_utts[split] = set()
        
        if not m_file.exists():
            warnings.append(f"Manifest missing for split '{split}': {m_file}")
            continue
            
        with open(m_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    utt_id = rec.get("utt_id")
                    audio = rec.get("audio")
                    text = rec.get("text", "")
                    
                    if not utt_id:
                        errors.append(f"[{split}:{line_no}] Missing utt_id")
                        continue
                        
                    if utt_id in split_utts[split]:
                        errors.append(f"[{split}] Duplicate utt_id: {utt_id}")
                        
                    split_utts[split].add(utt_id)
                    total_utterances += 1
                    
                    # Empty text check
                    if not text.strip():
                        errors.append(f"Empty transcript for utt_id: {utt_id}")
                        
                    # Audio existence, sample rate, duration check
                    if not audio or not Path(audio).exists():
                        errors.append(f"Missing audio file for {utt_id}: {audio}")
                    else:
                        all_audio_paths.add(audio)
                        try:
                            info = sf.info(audio)
                            if info.samplerate != target_sample_rate:
                                errors.append(f"Invalid sample rate for {utt_id}: {info.samplerate} != {target_sample_rate}")
                            if info.duration < 1.0 or info.duration > 20.0:
                                errors.append(f"Invalid duration for {utt_id}: {info.duration:.2f}s")
                        except Exception as ex:
                            errors.append(f"Corrupted audio file for {utt_id}: {ex}")
                except Exception as e:
                    errors.append(f"Invalid JSON on line {line_no} in {m_file}: {e}")

    # Check leakage between train, validation, test
    train_utts = split_utts.get("train", set())
    val_utts = split_utts.get("validation", set())
    test_utts = split_utts.get("test", set())
    
    train_val_leak = train_utts.intersection(val_utts)
    if train_val_leak:
        errors.append(f"Data leakage between train and validation: {train_val_leak}")
        
    train_test_leak = train_utts.intersection(test_utts)
    if train_test_leak:
        errors.append(f"Data leakage between train and test: {train_test_leak}")
        
    val_test_leak = val_utts.intersection(test_utts)
    if val_test_leak:
        errors.append(f"Data leakage between validation and test: {val_test_leak}")

    # 2. CosyVoice Embeddings and Tokens Checks
    utt2emb_file = cosy_path / "utt2embedding.pt"
    utt2tok_file = cosy_path / "utt2speech_token.pt"
    
    if not utt2emb_file.exists():
        errors.append(f"Missing speaker embeddings file: {utt2emb_file}")
    else:
        try:
            utt2emb = torch.load(utt2emb_file)
            missing_embs = train_utts - set(utt2emb.keys())
            if missing_embs:
                errors.append(f"Missing speaker embeddings for {len(missing_embs)} train utterances")
        except Exception as e:
            errors.append(f"Failed to load speaker embeddings {utt2emb_file}: {e}")
            
    if not utt2tok_file.exists():
        errors.append(f"Missing speech tokens file: {utt2tok_file}")
    else:
        try:
            utt2tok = torch.load(utt2tok_file)
            missing_toks = train_utts - set(utt2tok.keys())
            if missing_toks:
                errors.append(f"Missing speech tokens for {len(missing_toks)} train utterances")
        except Exception as e:
            errors.append(f"Failed to load speech tokens {utt2tok_file}: {e}")

    summary = {
        "valid": len(errors) == 0,
        "total_utterances": total_utterances,
        "train_utterances": len(train_utts),
        "validation_utterances": len(val_utts),
        "test_utterances": len(test_utts),
        "unique_audio_files": len(all_audio_paths),
        "errors": errors,
        "warnings": warnings
    }
    
    if errors:
        logger.error(f"Dataset validation FAILED with {len(errors)} errors:")
        for err in errors[:10]:
            logger.error(f"  - {err}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more errors.")
        raise DatasetValidationError(f"Dataset validation failed with {len(errors)} errors.")
    else:
        logger.info(f"Dataset validation PASSED ({total_utterances} total utterances).")
        
    return summary
