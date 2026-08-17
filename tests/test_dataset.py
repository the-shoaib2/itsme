"""
Unit Tests for Dataset Splitting and CosyVoice Metadata Preparation.
"""

import json
from pathlib import Path

import pytest

from itsme.dataset.cosyvoice_prep import prepare_cosyvoice_metadata_split
from itsme.dataset.splitter import split_dataset_files


@pytest.fixture
def verified_jsonl(tmp_path):
    v_file = tmp_path / "verified.jsonl"
    records = [
        {"utt_id": f"utt_{i:06d}", "audio": f"/path/to/utt_{i:06d}.wav", "text": f"Sample sentence {i}."}
        for i in range(1, 21)
    ]
    with open(v_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    return str(v_file)

def test_split_dataset_files(verified_jsonl, tmp_path):
    out_dir = tmp_path / "manifests"
    splits = split_dataset_files(verified_jsonl, output_dir=str(out_dir), seed=42)
    assert len(splits["train"]) > 0
    assert len(splits["validation"]) > 0
    assert len(splits["test"]) > 0
    assert (out_dir / "train.jsonl").exists()

def test_prepare_cosyvoice_metadata_split(verified_jsonl, tmp_path):
    out_dir = tmp_path / "cosyvoice"
    files = prepare_cosyvoice_metadata_split(verified_jsonl, output_dir=str(out_dir), speaker_id="itsme")
    assert Path(files["wav.scp"]).exists()
    assert Path(files["text"]).exists()
    assert Path(files["utt2spk"]).exists()
    assert Path(files["spk2utt"]).exists()
