"""
Custom Exception Classes for ItsMe Pipeline.
"""

class ItsMeError(Exception):
    """Base exception class for all ItsMe errors."""

class ConfigurationError(ItsMeError):
    """Raised when there is an invalid or missing configuration."""

class AudioProcessingError(ItsMeError):
    """Raised when audio validation, conversion, or preprocessing fails."""

class TranscriptionError(ItsMeError):
    """Raised when transcription or transcript cleaning fails."""

class FeatureExtractionError(ItsMeError):
    """Raised when speaker embedding or speech token extraction fails."""

class DatasetValidationError(ItsMeError):
    """Raised when dataset validation checks fail."""

class TrainingError(ItsMeError):
    """Raised when model fine-tuning or checkpointing fails."""

class InferenceError(ItsMeError):
    """Raised when TTS inference or streaming fails."""
