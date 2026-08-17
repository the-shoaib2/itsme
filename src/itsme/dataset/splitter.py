"""
Dataset Splitting Module.
Splits verified transcripts into train/validation/test sets deterministically.
"""

import json
import random
from pathlib import Path
from typing import Any

from itsme.utils.exceptions import DatasetValidationError
from itsme.utils.logging import get_logger

logger = get_logger("itsme.dataset.splitter")

def split_dataset_files(
    verified_jsonl: str = "data/transcripts/verified.jsonl",
    output_dir: str = "data/manifests",
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42
) -> dict[str, list[dict[str, Any]]]:
    """
    Deterministically split verified transcripts into train, validation, and test sets.
    """
    in_path = Path(verified_jsonl)
    if not in_path.exists():
        raise DatasetValidationError(f"Verified transcripts file not found: {verified_jsonl}")
        
    records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    if not records:
        raise DatasetValidationError("Cannot split empty dataset.")

    # Shuffle deterministically
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Ensure at least 1 record in train, val, test if total >= 3
    if total >= 3 and train_end == total:
        train_end = total - 2
        val_end = total - 1
        
    train_records = shuffled[:train_end]
    val_records = shuffled[train_end:val_end]
    test_records = shuffled[val_end:]
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train": (out_dir / "train.jsonl", train_records),
        "validation": (out_dir / "validation.jsonl", val_records),
        "test": (out_dir / "test.jsonl", test_records)
    }
    
    for split_name, (filepath, items) in splits.items():
        with open(filepath, "w", encoding="utf-8") as out:
            out.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in items)

    # Generate dataset_manifest.json and version fingerprint (Section 41 & 42)
    import hashlib
    import time
    
    train_dur = sum(r.get("duration", 2.5) for r in train_records)
    val_dur = sum(r.get("duration", 2.5) for r in val_records)
    test_dur = sum(r.get("duration", 2.5) for r in test_records)
    tot_dur = train_dur + val_dur + test_dur
    
    # Calculate dataset version fingerprint hash
    content_hash = hashlib.md5(f"{total}_{tot_dur}_{seed}".encode()).hexdigest()[:8]
    data_version = f"itsme-v{content_hash}"
    
    manifest_doc = {
        "speaker": "itsme",
        "sample_rate": 24000,
        "num_utterances": total,
        "total_hours": round(tot_dur / 3600.0, 3),
        "train_hours": round(train_dur / 3600.0, 3),
        "validation_hours": round(val_dur / 3600.0, 3),
        "test_hours": round(test_dur / 3600.0, 3),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "whisper_model": "large-v3",
        "data_version": data_version
    }
    
    with open("dataset_manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest_doc, mf, indent=2)
        
    logger.info(
        f"Split {total} items into: train={len(train_records)}, "
        f"val={len(val_records)}, test={len(test_records)} (seed={seed}). "
        f"Manifest saved to dataset_manifest.json (version: {data_version})"
    )
    
    return {
        "train": train_records,
        "validation": val_records,
        "test": test_records
    }
