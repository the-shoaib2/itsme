"""
Streaming Inference Module for CosyVoice 3.
Chunk text into sentences/phrases and stream PCM audio bytes.
"""

import re
import time
from collections.abc import Generator

import numpy as np

from itsme.inference.engine import CosyVoiceInferenceEngine
from itsme.utils.logging import get_logger

logger = get_logger("itsme.inference.streaming")

def chunk_text_by_clauses(text: str, max_chars: int = 80) -> list[str]:
    """
    Split long text into sentence/phrase chunks suitable for streaming TTS.
    """
    if not text:
        return []
        
    # Split by sentence boundaries (. ! ? ; \n)
    raw_chunks = re.split(r'([.!?;\n]+)', text)
    chunks = []
    current = ""
    
    for i in range(0, len(raw_chunks), 2):
        sentence = raw_chunks[i]
        punct = raw_chunks[i+1] if i+1 < len(raw_chunks) else ""
        full_sentence = (sentence + punct).strip()
        
        if not full_sentence:
            continue
            
        if len(current) + len(full_sentence) <= max_chars:
            current += (" " if current else "") + full_sentence
        else:
            if current:
                chunks.append(current)
            current = full_sentence
            
    if current:
        chunks.append(current)
        
    return chunks

class CosyVoiceStreamer:
    """
    Streaming Audio Generator for CosyVoice.
    """
    def __init__(self, engine: CosyVoiceInferenceEngine):
        self.engine = engine
        
    def stream_pcm_chunks(
        self,
        text: str,
        chunk_size_ms: int = 200
    ) -> Generator[bytes, None, None]:
        """
        Yield PCM 16-bit 24kHz audio byte chunks.
        """
        text_chunks = chunk_text_by_clauses(text)
        logger.info(f"Streaming text split into {len(text_chunks)} sentence/phrase chunks.")
        
        for t_chunk in text_chunks:
            res = self.engine.synthesize(t_chunk)
            audio_np = res["audio_np"]
            
            # Convert float32 [-1.0, 1.0] to int16 PCM bytes
            pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
            pcm_bytes = pcm16.tobytes()
            
            # Chunk into smaller byte packets according to chunk_size_ms
            bytes_per_sample = 2
            samples_per_chunk = int(24000 * (chunk_size_ms / 1000.0))
            chunk_byte_size = samples_per_chunk * bytes_per_sample
            
            for offset in range(0, len(pcm_bytes), chunk_byte_size):
                chunk = pcm_bytes[offset : offset + chunk_byte_size]
                yield chunk
                time.sleep(0.01) # Low latency pacing
