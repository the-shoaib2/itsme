"""
Hardware and Environment Inspection Utilities for ItsMe.
"""

import os
import shutil
import sys
from typing import Any


def detect_device(preferred: str = "auto") -> str:
    """
    Detect available execution device.
    Priority: CUDA > MPS > CPU
    """
    if preferred != "auto" and preferred in ["cuda", "mps", "cpu"]:
        return preferred
        
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
        
    return "cpu"

def get_system_status() -> dict[str, Any]:
    """
    Inspect environment status including Python, PyTorch, CUDA, MPS, FFmpeg, Git, Disk space.
    """
    status = {
        "python": {"status": "OK", "version": sys.version.split()[0]},
        "pytorch": {"status": "FAIL", "version": "Not installed"},
        "cuda": {"status": "N/A", "available": False, "device_name": None, "vram_gb": None},
        "mps": {"status": "N/A", "available": False},
        "ffmpeg": {"status": "FAIL", "path": None},
        "git": {"status": "FAIL", "path": None},
        "disk": {"status": "OK", "free_gb": 0.0},
    }
    
    # PyTorch & Accelerators
    try:
        import torch
        status["pytorch"] = {"status": "OK", "version": torch.__version__}
        
        if torch.cuda.is_available():
            status["cuda"] = {
                "status": "OK",
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            }
        else:
            status["cuda"] = {"status": "N/A", "available": False, "device_name": None, "vram_gb": None}
            
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            status["mps"] = {"status": "OK", "available": True}
        else:
            status["mps"] = {"status": "N/A", "available": False}
    except Exception as e:
        status["pytorch"]["error"] = str(e)
        
    # FFmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        status["ffmpeg"] = {"status": "OK", "path": ffmpeg_path}
    else:
        # Check brew / default paths
        if os.path.exists("/opt/homebrew/bin/ffmpeg"):
            status["ffmpeg"] = {"status": "OK", "path": "/opt/homebrew/bin/ffmpeg"}
            
    # Git
    git_path = shutil.which("git")
    if git_path:
        status["git"] = {"status": "OK", "path": git_path}
        
    # Disk Space
    try:
        stat = shutil.disk_usage(".")
        status["disk"]["free_gb"] = round(stat.free / (1024**3), 2)
    except Exception:
        pass
        
    return status

def print_system_check_report(status: dict[str, Any]):
    """
    Format and print readable system status report.
    """
    print("\n==================================================")
    print("ItsMe System Check")
    print("==================================================")
    print(f"Python:   {status['python']['status']} ({status['python']['version']})")
    print(f"PyTorch:  {status['pytorch']['status']} ({status['pytorch'].get('version', 'N/A')})")
    
    if status["cuda"]["available"]:
        print(f"CUDA:     OK ({status['cuda']['device_name']}, {status['cuda']['vram_gb']} GB VRAM)")
    else:
        print("CUDA:     N/A (Not available / CPU or MPS mode)")
        
    if status["mps"]["available"]:
        print("MPS:      OK (Apple Silicon Hardware Acceleration)")
    else:
        print("MPS:      N/A")
        
    print(f"FFmpeg:   {status['ffmpeg']['status']} ({status['ffmpeg'].get('path', 'Not found')})")
    print(f"Git:      {status['git']['status']} ({status['git'].get('path', 'Not found')})")
    print(f"Disk:     OK ({status['disk']['free_gb']} GB free)")
    print("==================================================\n")
