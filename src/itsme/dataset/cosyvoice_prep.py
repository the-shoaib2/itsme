"""
CosyVoice Standard Data Preparation Module.
Generates wav.scp, text, utt2spk, spk2utt files.
"""

import json
from pathlib import Path

from itsme.utils.exceptions import DatasetValidationError
from itsme.utils.logging import get_logger

logger = get_logger("itsme.dataset.cosyvoice_prep")

def prepare_cosyvoice_metadata_split(
    manifest_jsonl: str,
    output_dir: str,
    speaker_id: str = "itsme"
) -> dict[str, str]:
    """
    Generate wav.scp, text, utt2spk, spk2utt in output_dir for given manifest split.
    """
    manifest_path = Path(manifest_jsonl)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.exists():
        raise DatasetValidationError(f"Manifest file not found: {manifest_jsonl}")
        
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    wav_scp_lines = []
    text_lines = []
    utt2spk_lines = []
    
    utt_ids = []
    for rec in records:
        utt_id = rec.get("utt_id")
        audio_path = rec.get("audio")
        text = rec.get("text", "")
        
        if not utt_id or not audio_path:
            continue
            
        wav_scp_lines.append(f"{utt_id} {audio_path}")
        text_lines.append(f"{utt_id} {text}")
        utt2spk_lines.append(f"{utt_id} {speaker_id}")
        utt_ids.append(utt_id)
        
    spk2utt_line = f"{speaker_id} " + " ".join(utt_ids)
    
    files = {
        "wav.scp": out_dir / "wav.scp",
        "text": out_dir / "text",
        "utt2spk": out_dir / "utt2spk",
        "spk2utt": out_dir / "spk2utt"
    }
    
    with open(files["wav.scp"], "w", encoding="utf-8") as f:
        f.write("\n".join(wav_scp_lines) + "\n")
        
    with open(files["text"], "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines) + "\n")
        
    with open(files["utt2spk"], "w", encoding="utf-8") as f:
        f.write("\n".join(utt2spk_lines) + "\n")
        
    with open(files["spk2utt"], "w", encoding="utf-8") as f:
        f.write(spk2utt_line + "\n")
        
    logger.info(f"Generated CosyVoice Kaldi metadata in {output_dir}: {len(utt_ids)} utterances")
    return {k: str(v.resolve()) for k, v in files.items()}

def prepare_all_cosyvoice_metadata(
    manifests_dir: str = "data/manifests",
    cosyvoice_dir: str = "data/cosyvoice",
    speaker_id: str = "itsme"
) -> dict[str, dict[str, str]]:
    """
    Prepare CosyVoice metadata for train, validation, and test splits as well as full dataset root.
    """
    manifest_path = Path(manifests_dir)
    res = {}
    
    # Root metadata
    verified_file = Path("data/transcripts/verified.jsonl")
    if verified_file.exists():
        res["root"] = prepare_cosyvoice_metadata_split(
            str(verified_file), cosyvoice_dir, speaker_id=speaker_id
        )
        
    for split in ["train", "validation", "test"]:
        m_file = manifest_path / f"{split}.jsonl"
        if m_file.exists():
            out_d = Path(cosyvoice_dir) / split
            res[split] = prepare_cosyvoice_metadata_split(
                str(m_file), str(out_d), speaker_id=speaker_id
            )
            
    return res
