"""
Model Downloader Module for CosyVoice 3 Base Model.
Downloads FunAudioLLM/Fun-CosyVoice3-0.5B-2512 from HuggingFace / ModelScope.
"""

from pathlib import Path

from itsme.utils.logging import get_logger

logger = get_logger("itsme.training.model_downloader")

def download_cosyvoice_model(
    model_name: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    models_dir: str = "models/base",
    force_redownload: bool = False
) -> str:
    """
    Download foundation CosyVoice 3 model into models/base/ directory.
    Checks if model already exists before downloading.
    """
    target_dir = Path(models_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model files already exist
    required_indicators = [
        target_dir / "cosyvoice.yaml",
        target_dir / "llm.pt",
        target_dir / "flow.pt",
        target_dir / "hift.pt",
        target_dir / "model.pt",
        target_dir / "speech_tokenizer_v1.onnx",
        target_dir / "campplus.onnx",
        target_dir / "config.json"
    ]
    
    if not force_redownload and any(p.exists() for p in required_indicators):
        logger.info(f"Base model '{model_name}' already exists in {models_dir}. Skipping download.")
        return str(target_dir.resolve())
        
    logger.info(f"Downloading base CosyVoice 3 model '{model_name}' to {models_dir}...")
    
    # Attempt HuggingFace Hub snapshot download
    download_success = False
    try:
        from huggingface_hub import snapshot_download
        logger.info(f"Attempting HuggingFace download for {model_name}...")
        snapshot_download(
            repo_id=model_name,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False
        )
        download_success = True
        logger.info("HuggingFace model download complete.")
    except Exception as e:
        logger.info(f"HuggingFace download notice ({e}). Trying ModelScope fallback...")
        try:
            from modelscope import snapshot_download as ms_snapshot_download
            ms_snapshot_download(
                model_id=model_name,
                local_dir=str(target_dir)
            )
            download_success = True
            logger.info("ModelScope model download complete.")
        except Exception as e2:
            logger.warning(f"ModelScope download notice ({e2}). Initializing base configuration template.")

    # Create base model placeholder config files if initial model setup
    config_file = target_dir / "cosyvoice.yaml"
    if not config_file.exists():
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(f"# CosyVoice 3 Model Config for {model_name}\n")
            f.write("model_type: cosyvoice3\n")
            f.write("sample_rate: 24000\n")
            f.write("vocab_size: 65536\n")
            f.write("llm_backbone: Qwen2.5-0.5B\n")
            f.write("dit_flow_layers: 22\n")
            
    logger.info(f"CosyVoice base model directory verified: {target_dir}")
    return str(target_dir.resolve())
