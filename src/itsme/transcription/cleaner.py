"""
Transcript Cleaning and Text Normalization Module.
"""

import json
import re
from pathlib import Path
from typing import Any

from itsme.utils.exceptions import TranscriptionError
from itsme.utils.logging import get_logger

logger = get_logger("itsme.transcription.cleaner")

def normalize_text(text: str) -> str:
    """
    Clean transcript text:
    - Collapse multiple whitespace
    - Normalize quotes and dashes
    - Remove hallucinated whisper artifacts like [BLANK_AUDIO] or (music)
    - Remove duplicated punctuation like '!!' -> '!'
    - Retain original spelling and casing
    """
    if not text:
        return ""
        
    # Remove Whisper noise tags
    text = re.sub(r'\[(BLANK_AUDIO|MUSIC|LAUGHTER|NOISE|SILENCE)\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\((music|applause|laughter|chuckle)\)', '', text, flags=re.IGNORECASE)
    
    # Normalize quotes and smart characters
    text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    
    # Fix repeated punctuation like "!!" -> "!", "??" -> "?"
    text = re.sub(r'!+', '!', text)
    text = re.sub(r'\?+', '?', text)
    text = re.sub(r'\.+', '.', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_transcripts_file(
    input_jsonl: str = "data/transcripts/raw.jsonl",
    output_jsonl: str = "data/transcripts/clean.jsonl"
) -> list[dict[str, Any]]:
    """
    Read raw.jsonl, apply normalize_text, and save to clean.jsonl.
    """
    in_path = Path(input_jsonl)
    if not in_path.exists():
        raise TranscriptionError(f"Raw transcripts file not found: {input_jsonl}")
        
    cleaned_records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            raw_text = item.get("text", "")
            cleaned_text = normalize_text(raw_text)
            item["text"] = cleaned_text
            cleaned_records.append(item)
            
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        out.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in cleaned_records)
            
    logger.info(f"Cleaned {len(cleaned_records)} transcripts -> {output_jsonl}")
    return cleaned_records
