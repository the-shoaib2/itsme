"""
Human Transcript Review and Verification Tool.
Supports interactive CLI review [y/e/s/q] and automated auto-accept mode.
Saves to data/transcripts/verified.jsonl.
"""

import json
from pathlib import Path
from typing import Any

from itsme.utils.exceptions import TranscriptionError
from itsme.utils.logging import get_logger

logger = get_logger("itsme.transcription.reviewer")

def review_transcripts_cli(
    input_jsonl: str = "data/transcripts/clean.jsonl",
    output_jsonl: str = "data/transcripts/verified.jsonl",
    auto_accept: bool = False
) -> list[dict[str, Any]]:
    """
    Run transcript review session and write verified.jsonl.
    """
    in_path = Path(input_jsonl)
    if not in_path.exists():
        raise TranscriptionError(f"Clean transcripts file not found: {input_jsonl}")
        
    records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    verified_records = []
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoaded {len(records)} transcripts from {input_jsonl}")
    
    if auto_accept:
        logger.info("Auto-accept mode enabled. Accepting all transcripts into verified.jsonl")
        verified_records = records
    else:
        print("Interactive Review Options: [y] accept | [e] edit | [s] skip | [q] quit & save\n")
        for idx, item in enumerate(records, 1):
            utt_id = item.get("utt_id", f"utt_{idx:06d}")
            text = item.get("text", "")
            audio = item.get("audio", "")
            
            print(f"[{idx}/{len(records)}] Utterance ID: {utt_id}")
            print(f"Audio File:   {audio}")
            print(f"Transcript:   {text}")
            
            try:
                choice = input("Action [y/e/s/q]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                choice = 'q'
                
            if choice == 'y' or choice == '':
                verified_records.append(item)
                print("-> Accepted\n")
            elif choice == 'e':
                new_text = input("Enter corrected transcript: ").strip()
                if new_text:
                    item["text"] = new_text
                verified_records.append(item)
                print("-> Updated and Accepted\n")
            elif choice == 's':
                print("-> Skipped\n")
            elif choice == 'q':
                print("-> Saving verified transcripts and quitting review.\n")
                break
            else:
                print("-> Unknown option, defaulting to accept.\n")
                verified_records.append(item)

    with open(out_path, "w", encoding="utf-8") as out:
        out.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in verified_records)
            
    logger.info(f"Verified {len(verified_records)} transcripts -> {output_jsonl}")
    return verified_records
