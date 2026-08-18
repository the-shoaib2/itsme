"""
Checkpoint Manager for ItsMe Training Pipeline.
Handles step saving, latest/best tracking, loading, and metadata saving.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from itsme.utils.logging import get_logger

logger = get_logger("itsme.training.checkpoint")

class CheckpointManager:
    """
    Manages training checkpoints under runs/<run_name>/checkpoints/.
    """
    def __init__(self, run_dir: str):
        self.run_path = Path(run_dir)
        self.checkpoints_dir = self.run_path / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self.latest_link = self.checkpoints_dir / "latest"
        self.best_link = self.checkpoints_dir / "best"
        self.best_metric = float("inf")

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any],
        loss: float,
        val_loss: float | None = None,
        is_best: bool = False
    ) -> str:
        """
        Save checkpoint step folder and update latest/best pointers.
        """
        step_dir = self.checkpoints_dir / f"step-{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        ckpt_file = step_dir / "model.pt"
        torch.save({
            "step": step,
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "loss": loss,
            "val_loss": val_loss
        }, ckpt_file)
        
        meta = {
            "step": step,
            "epoch": epoch,
            "loss": round(loss, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "checkpoint_path": str(ckpt_file.resolve())
        }
        with open(step_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            
        # Update latest pointer/copy
        self._update_symlink_or_dir(step_dir, self.latest_link)
        
        if is_best:
            self._update_symlink_or_dir(step_dir, self.best_link)
            logger.info(f"New BEST checkpoint saved at step {step} (val_loss={val_loss})")
            
        logger.info(f"Saved checkpoint: {step_dir}")
        return str(step_dir.resolve())

    def _update_symlink_or_dir(self, source: Path, target: Path):
        """Update symlink or fallback to directory copy if symlinks not permitted."""
        if target.exists() or target.is_symlink():
            if target.is_symlink():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
                
        try:
            target.symlink_to(source.resolve(), target_is_directory=True)
        except Exception:
            shutil.copytree(source, target)

    def find_latest_checkpoint(self) -> str | None:
        """
        Find path to latest checkpoint step directory or file.
        """
        if self.latest_link.exists():
            return str(self.latest_link.resolve())
            
        # Search for step-* folders
        step_dirs = sorted([
            d for d in self.checkpoints_dir.glob("step-*") if d.is_dir()
        ])
        if step_dirs:
            return str(step_dirs[-1].resolve())
            
        return None

    def find_best_checkpoint(self) -> str | None:
        """
        Find path to best checkpoint step directory or file.
        """
        if self.best_link.exists():
            return str(self.best_link.resolve())
        return self.find_latest_checkpoint()
