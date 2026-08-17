"""
Unified CLI Entry Point for ItsMe.
Command structure:
  itsme audio validate
  itsme audio prepare
  itsme audio segment
  itsme transcribe
  itsme transcripts review
  itsme dataset prepare
  itsme dataset validate
  itsme dataset features
  itsme dataset parquet
  itsme model download
  itsme train start
  itsme train resume
  itsme evaluate
  itsme infer
  itsme api
"""

import click


@click.group()
def cli():
    """ItsMe — The Voice of Me CLI Pipeline Manager"""

@cli.command("system-check")
def system_check_cmd():
    """Check hardware, PyTorch, CUDA, MPS, and environment tools."""
    from itsme.utils.hardware import get_system_status, print_system_check_report
    status = get_system_status()
    print_system_check_report(status)

# Audio Subcommands
@cli.group("audio")
def audio_group():
    """Audio processing and validation commands."""

@audio_group.command("validate")
@click.option("--raw-dir", default="data/raw", help="Path to raw audio directory")
def audio_validate(raw_dir):
    """Validate quality of raw audio recordings."""
    from itsme.audio.validator import validate_audio_dir
    recs = validate_audio_dir(raw_dir=raw_dir)
    click.echo(f"Validated {len(recs)} raw audio files.")

@audio_group.command("prepare")
@click.option("--raw-dir", default="data/raw", help="Path to raw audio directory")
@click.option("--output-dir", default="data/processed", help="Path to processed output directory")
def audio_prepare(raw_dir, output_dir):
    """Preprocess audio to mono 24kHz normalized WAV files."""
    from itsme.audio.preprocessor import prepare_all_audio
    recs = prepare_all_audio(raw_dir=raw_dir, output_dir=output_dir)
    click.echo(f"Preprocessed {len(recs)} files -> {output_dir}")

@audio_group.command("segment")
@click.option("--processed-dir", default="data/processed", help="Path to processed audio directory")
@click.option("--output-dir", default="data/segments", help="Path to segments output directory")
def audio_segment(processed_dir, output_dir):
    """Apply VAD segmentation to create 2s-15s utterances."""
    from itsme.audio.filter import filter_and_report_segments
    from itsme.audio.vad import segment_all_audio
    recs = segment_all_audio(processed_dir=processed_dir, output_dir=output_dir)
    report = filter_and_report_segments(segments_dir=output_dir)
    click.echo(f"Created {report['accepted']} valid speech segments in {output_dir}")

# Transcription Subcommands
@cli.command("transcribe")
@click.option("--segments-dir", default="data/segments", help="Path to segments directory")
@click.option("--model", default="large-v3", help="Whisper model size")
def transcribe_cmd(segments_dir, model):
    """Transcribe speech segments with Whisper."""
    from itsme.transcription.cleaner import clean_transcripts_file
    from itsme.transcription.whisper import transcribe_segments
    recs = transcribe_segments(segments_dir=segments_dir, model_name=model)
    clean_transcripts_file()
    click.echo(f"Transcribed {len(recs)} segments -> data/transcripts/raw.jsonl & clean.jsonl")

@cli.group("transcripts")
def transcripts_group():
    """Transcript cleaning and review commands."""

@transcripts_group.command("review")
@click.option("--auto-accept", is_flag=True, help="Auto-accept clean transcripts")
def transcripts_review(auto_accept):
    """Interactive or automated transcript review."""
    from itsme.transcription.reviewer import review_transcripts_cli
    recs = review_transcripts_cli(auto_accept=auto_accept)
    click.echo(f"Verified {len(recs)} transcripts -> data/transcripts/verified.jsonl")

# Dataset Subcommands
@cli.group("dataset")
def dataset_group():
    """Dataset splitting, CosyVoice metadata, features, and validation."""

@dataset_group.command("prepare")
def dataset_prepare():
    """Split dataset and prepare CosyVoice Kaldi metadata."""
    from itsme.dataset.cosyvoice_prep import prepare_all_cosyvoice_metadata
    from itsme.dataset.splitter import split_dataset_files
    splits = split_dataset_files()
    res = prepare_all_cosyvoice_metadata()
    click.echo("Dataset manifests and CosyVoice metadata prepared.")

@dataset_group.command("validate")
def dataset_validate():
    """Run full dataset validation checks."""
    from itsme.dataset.validator import validate_dataset
    summary = validate_dataset()
    click.echo("Dataset validation PASSED!")

@dataset_group.command("features")
def dataset_features():
    """Extract speaker embeddings and speech tokens."""
    from itsme.features.embeddings import extract_speaker_embeddings
    from itsme.features.speech_tokens import extract_speech_tokens
    emb_res = extract_speaker_embeddings()
    tok_res = extract_speech_tokens()
    click.echo("Extracted speaker embeddings and speech tokens.")

@dataset_group.command("parquet")
def dataset_parquet():
    """Build Parquet files for CosyVoice training."""
    from itsme.features.parquet_builder import build_all_parquets
    p_res = build_all_parquets()
    click.echo(f"Parquet datasets built: {p_res}")

# Model Subcommands
@cli.group("model")
def model_group():
    """Model management commands."""

@model_group.command("download")
@click.option("--model-name", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512", help="Model name")
def model_download(model_name):
    """Download base CosyVoice 3 model."""
    from itsme.training.model_downloader import download_cosyvoice_model
    path = download_cosyvoice_model(model_name=model_name)
    click.echo(f"Base model ready at {path}")

# Training Subcommands
@cli.group("train")
def train_group():
    """Model fine-tuning and training resume commands."""

@train_group.command("start")
@click.option("--config", default="configs/training.yaml", help="Path to training config")
def train_start(config):
    """Start model fine-tuning run."""
    from itsme.config.config import load_yaml
    from itsme.training.trainer import CosyVoiceTrainer
    cfg = load_yaml(config)
    trainer = CosyVoiceTrainer(config=cfg)
    path = trainer.train()
    click.echo(f"Training finished: {path}")

@train_group.command("resume")
@click.option("--config", default="configs/training.yaml", help="Path to training config")
@click.option("--checkpoint", default="runs/itsme/latest", help="Checkpoint to resume")
def train_resume(config, checkpoint):
    """Resume model fine-tuning run."""
    from itsme.config.config import load_yaml
    from itsme.training.trainer import CosyVoiceTrainer
    cfg = load_yaml(config)
    trainer = CosyVoiceTrainer(config=cfg, resume_from=checkpoint)
    path = trainer.train()
    click.echo(f"Resumed training finished: {path}")

# Evaluation & Inference Subcommands
@cli.command("evaluate")
def evaluate_cmd():
    """Run quality evaluation suite."""
    from itsme.evaluation.evaluator import run_evaluation_suite
    report = run_evaluation_suite()
    click.echo("Evaluation suite complete.")

@cli.command("infer")
@click.option("--text", required=True, help="Text to synthesize")
@click.option("--output", default="outputs/test.wav", help="Output WAV path")
def infer_cmd(text, output):
    """Synthesize speech from text."""
    from itsme.inference.engine import CosyVoiceInferenceEngine
    engine = CosyVoiceInferenceEngine()
    res = engine.synthesize(text=text, output_path=output)
    click.echo(f"Synthesized voice audio saved -> {output}")

@cli.command("api")
@click.option("--host", default="0.0.0.0", help="Host address")
@click.option("--port", default=8000, help="Port number")
def api_cmd(host, port):
    """Start REST & WebSocket API server."""
    import uvicorn
    uvicorn.run("api.app:app", host=host, port=port, reload=False)

def main():
    cli()

if __name__ == "__main__":
    main()
