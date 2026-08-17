# ItsMe — The Voice of Me

**Production-Quality Personal Voice Cloning and Fine-Tuning System using FunAudioLLM/Fun-CosyVoice3-0.5B-2512**

---

## 🌟 Overview

**ItsMe** is an end-to-end, production-grade speech synthesis, voice fine-tuning, and personal voice cloning platform. It is engineered specifically for authorized personal voice cloning, allowing you to train a high-fidelity, expressive, human-like voice model preserving your unique vocal identity.

The system utilizes **FunAudioLLM/Fun-CosyVoice3-0.5B-2512** as its core TTS foundation model, providing state-of-the-art zero-shot voice cloning, fine-tuning capabilities, real-time bi-directional audio streaming, and REST/WebSocket API endpoints.

---

## ⚠️ Safety & Authorization

> **IMPORTANT LEGAL & ETHICAL NOTICE**
> 
> This software is strictly intended for fine-tuning on your own voice or voices for which you have explicit, documented authorization.
> Do NOT use this platform to create unauthorized voice clones, impersonate real individuals without consent, or build deceptive audio applications.

---

## 🏗 System Architecture

```
RAW AUDIO (.m4a, .mp3, .wav, .flac)
    ↓
Audio Validation & Diagnostics
    ↓
Audio Normalization (24 kHz Mono WAV)
    ↓
Voice Activity Detection (VAD)
    ↓
Speech Segmentation (2s – 15s Utterances)
    ↓
Silence Trimming & Quality Filtering
    ↓
Whisper Speech-to-Text Transcription (large-v3)
    ↓
Transcript Cleaning & Text Normalization
    ↓
Human Transcript Review CLI Workflow
    ↓
Deterministic Train/Validation/Test Split (90/5/5)
    ↓
CosyVoice Kaldi Metadata Preparation (wav.scp, text, utt2spk, spk2utt)
    ↓
Speaker Embedding Extraction (CAMPPlus / spk2embedding.pt)
    ↓
Speech Token Extraction (S3Tokenizer / utt2speech_token.pt)
    ↓
PyArrow Parquet Dataset Generation
    ↓
CosyVoice 3 Model Fine-Tuning & Gradient Accumulation
    ↓
Checkpoint Manager & Reproducibility Metadata
    ↓
Automatic Validation Audio Generation & Quality Suite (JSON, MD, HTML)
    ↓
Low-Latency Inference Engine & PCM WebSocket Streaming API
```

---

## 💻 Supported Hardware

- **NVIDIA GPU (CUDA 12.1+ / 16GB+ VRAM)**: Recommended for full model fine-tuning, multi-GPU training, and high-throughput production inference.
- **Apple Silicon (Mac Mini M4 / M-Series MPS)**: Supported for audio preprocessing, VAD segmentation, Whisper transcription, dataset feature extraction, local testing, and inference experiments.
- **CPU**: Fallback mode supported across all preprocessing, validation, and inference tasks.

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/the-shoaib2/itsme.git
cd itsme

# Run complete automated environment setup
make setup
```

### 2. Run System Environment Check

```bash
python scripts/system_check.py
# or using unified CLI:
itsme system-check
```

---

## 🎙 Dataset Recording & Preparation Guide

### Recording Recommendations
- **Environment**: Quiet room with minimal acoustic reflection (avoid echo).
- **Microphone**: Consistent cardioid or lavalier microphone positioned 6–8 inches from mouth.
- **Format**: `.m4a`, `.wav`, `.flac`, or `.mp3`.
- **Speech Style**: Speak naturally at your normal conversational cadence. Include varied emotions, statements, questions, numbers, dates, short clips, and longer paragraphs.

### Pipeline Execution Commands

```bash
# 1. Place raw voice recordings into data/raw/
cp /path/to/my_recordings/* data/raw/

# 2. Validate raw audio quality
make validate

# 3. Preprocess and segment audio into 2s-15s utterances
make prepare

# 4. Transcribe segments using Whisper large-v3
make transcribe

# 5. Review transcripts (Interactive CLI or Auto-accept)
make review

# 6. Generate train/validation/test splits and CosyVoice Kaldi metadata
make dataset

# 7. Extract speaker embeddings, speech tokens, and build Parquet files
make features
```

---

## 🧠 Model Download & Fine-Tuning

```bash
# Download Fun-CosyVoice3-0.5B-2512 foundation model
python scripts/download_model.py

# Start CosyVoice 3 fine-tuning run
make train

# Resume training from latest checkpoint if interrupted
python scripts/resume_training.py
```

---

## 📊 Quality Evaluation & Checkpoint Comparison

```bash
# Run quality evaluation suite
make evaluate

# Compare multiple checkpoints on standard evaluation prompts
python scripts/compare_checkpoints.py --checkpoints runs/itsme/checkpoints/step-001000 runs/itsme/checkpoints/best
```

---

## 🔊 Synthesis & API Deployment

### Local CLI Inference
```bash
make infer
# or custom text:
python scripts/infer.py --text "Hello, this is my personal voice fine-tuned with ItsMe." --output outputs/my_voice.wav
```

### REST & WebSocket API Server
```bash
make api
```
- **REST Synthesis**: `POST http://localhost:8000/api/v1/tts`
- **WebSocket Streaming**: `WS ws://localhost:8000/api/v1/tts/stream`
- **Voice Info**: `GET http://localhost:8000/api/v1/voices/itsme`
- **Health Check**: `GET http://localhost:8000/api/v1/health`

---

## 🐳 Docker Deployment

```bash
# Build and launch CUDA-enabled container
docker-compose up --build -d
```

---

## 📁 Repository Structure

```
itsme/
├── data/                  # Audio data, segments, transcripts, manifests, and Parquet
├── models/                # Base foundation models and trained checkpoints
├── runs/                  # Fine-tuning run logs, samples, and metadata
├── evaluation/            # Prompts, generated audio, and quality reports (JSON, MD, HTML)
├── configs/               # YAML configurations (config.yaml, dataset.yaml, training.yaml, etc.)
├── scripts/               # Clean executable scripts for each pipeline step
├── src/itsme/             # Package core modules (audio, dataset, transcription, features, training, inference, api)
├── api/                   # FastAPI routes, schemas, and app
├── tests/                 # Comprehensive unit test suite
├── Dockerfile             # CUDA-compatible Docker configuration
├── docker-compose.yml     # Docker Compose definition with GPU passthrough
├── Makefile               # Automated build & execution targets
├── pyproject.toml         # Package dependencies & CLI entrypoints
└── README.md
```

---

## 📄 License

Licensed under the MIT License.
