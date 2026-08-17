"""
Unit Tests for Transcript Cleaning and Normalization.
"""

from itsme.transcription.cleaner import normalize_text


def test_normalize_text_whitespace():
    raw = "Hello   world!  This   is  a test."
    cleaned = normalize_text(raw)
    assert cleaned == "Hello world! This is a test."

def test_normalize_text_artifacts():
    raw = "[BLANK_AUDIO] (music) Hello world!!"
    cleaned = normalize_text(raw)
    assert cleaned == "Hello world!"

def test_normalize_text_quotes():
    raw = "It’s a “great” day—isn’t it?"
    cleaned = normalize_text(raw)
    assert cleaned == 'It\'s a "great" day-isn\'t it?'
