.PHONY: setup validate prepare transcribe review dataset features train evaluate infer api test lint format clean

PYTHON := .venv/bin/python3
PIP := .venv/bin/pip

setup:
	@echo "Setting up virtual environment and dependencies..."
	@/opt/homebrew/bin/python3.11 -m venv .venv || python3 -m venv .venv
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -e .[dev]

validate:
	$(PYTHON) scripts/validate_audio.py

prepare:
	$(PYTHON) scripts/prepare_audio.py
	$(PYTHON) scripts/segment_audio.py

transcribe:
	$(PYTHON) scripts/transcribe.py
	$(PYTHON) scripts/clean_transcripts.py

review:
	$(PYTHON) scripts/review_transcripts.py --auto-accept

dataset:
	$(PYTHON) scripts/split_dataset.py
	$(PYTHON) scripts/prepare_cosyvoice.py

features:
	$(PYTHON) scripts/extract_embeddings.py
	$(PYTHON) scripts/extract_speech_tokens.py
	$(PYTHON) scripts/build_parquet.py
	$(PYTHON) scripts/validate_dataset.py

train:
	$(PYTHON) scripts/train.py --config configs/training.yaml

evaluate:
	$(PYTHON) scripts/evaluate.py

infer:
	$(PYTHON) scripts/infer.py --text "Hello, this is my authorized personal voice created with ItsMe." --output outputs/demo.wav

api:
	$(PYTHON) -m uvicorn api.app:app --host 0.0.0.0 --port 8000

test:
	.venv/bin/pytest -v tests/

lint:
	.venv/bin/ruff check src/ scripts/ api/ tests/

format:
	.venv/bin/ruff format src/ scripts/ api/ tests/
	.venv/bin/black src/ scripts/ api/ tests/

clean:
	rm -rf data/processed/* data/segments/* data/transcripts/* data/manifests/* data/cosyvoice/* data/reports/* evaluation/generated/* outputs/* .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
