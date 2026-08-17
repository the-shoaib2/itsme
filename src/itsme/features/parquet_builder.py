"""
Parquet Dataset Generation Module for CosyVoice Training Pipeline.
Converts prepared audio, text, speaker embeddings, and speech tokens into PyArrow Parquet.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from itsme.utils.exceptions import FeatureExtractionError
from itsme.utils.logging import get_logger

logger = get_logger("itsme.features.parquet_builder")

def build_parquet_dataset_split(
    cosyvoice_split_dir: str,
    root_cosyvoice_dir: str = "data/cosyvoice"
) -> str:
    """
    Build data.parquet for a specific CosyVoice split directory (e.g. data/cosyvoice/train).
    """
    split_path = Path(cosyvoice_split_dir)
    root_path = Path(root_cosyvoice_dir)
    
    wav_scp = split_path / "wav.scp"
    text_file = split_path / "text"
    utt2spk_file = split_path / "utt2spk"
    
    utt2emb_file = root_path / "utt2embedding.pt"
    utt2tok_file = root_path / "utt2speech_token.pt"
    
    if not (wav_scp.exists() and text_file.exists() and utt2spk_file.exists()):
        raise FeatureExtractionError(f"Missing Kaldi metadata files in {cosyvoice_split_dir}")
        
    if not (utt2emb_file.exists() and utt2tok_file.exists()):
        raise FeatureExtractionError(f"Missing utt2embedding.pt or utt2speech_token.pt in {root_cosyvoice_dir}")

    # Load torch tensors
    utt2emb = torch.load(utt2emb_file)
    utt2tok = torch.load(utt2tok_file)
    
    wav_map = {}
    with open(wav_scp, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(maxsplit=1)
                wav_map[parts[0]] = parts[1]
                
    text_map = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    text_map[parts[0]] = parts[1]
                    
    spk_map = {}
    with open(utt2spk_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(maxsplit=1)
                spk_map[parts[0]] = parts[1]

    records = []
    for utt_id, wav_path in wav_map.items():
        text = text_map.get(utt_id, "")
        speaker = spk_map.get(utt_id, "itsme")
        
        emb = utt2emb.get(utt_id)
        tok = utt2tok.get(utt_id)
        
        if emb is None or tok is None or not text:
            logger.warning(f"Skipping incomplete record {utt_id} for parquet build.")
            continue
            
        records.append({
            "utt_id": utt_id,
            "audio_path": wav_path,
            "text": text,
            "speaker": speaker,
            "speaker_embedding": emb.squeeze().tolist(),
            "speech_tokens": tok.squeeze().tolist()
        })
        
    if not records:
        raise FeatureExtractionError(f"No complete records found to build parquet in {cosyvoice_split_dir}")

    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df)
    
    out_parquet = split_path / "data.parquet"
    pq.write_table(table, out_parquet)
    
    logger.info(f"Built Parquet dataset with {len(records)} records -> {out_parquet}")
    return str(out_parquet.resolve())

def build_all_parquets(cosyvoice_dir: str = "data/cosyvoice") -> dict[str, str]:
    """
    Build Parquet datasets for train, validation, and root datasets.
    """
    root_path = Path(cosyvoice_dir)
    results = {}
    
    for split in ["train", "validation", "test"]:
        s_dir = root_path / split
        if s_dir.exists() and (s_dir / "wav.scp").exists():
            results[split] = build_parquet_dataset_split(str(s_dir), root_cosyvoice_dir=cosyvoice_dir)
            
    return results
