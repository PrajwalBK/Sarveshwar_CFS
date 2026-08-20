"""
PyTorch CUDA diagnostics and a lightweight runtime probe for Jetson / desktop.

Driver ↔ PyTorch alignment is resolved on the device by installing a torch build
that matches the NVIDIA driver / JetPack (see log messages from log_cuda_setup).
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Optional, Tuple


def cuda_total_memory_bytes(device_index: int = 0) -> Optional[int]:
    """Return torch-reported total memory for a CUDA device, or None if unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.get_device_properties(device_index).total_memory)
    except Exception:
        return None


def suggest_ocr_model_for_memory(model_name: str, use_gpu: bool) -> Tuple[str, Optional[str]]:
    """
    Pick a smaller / FP8 HF id when the default Qwen3-VL-4B full weights are likely to OOM
    the edge device during from_pretrained (Linux OOM killer shows as ``Killed``).

    Returns (model_name, log_reason_or_None).
    Caller should skip this when the user set OCR_MODEL explicitly or disabled auto mode.
    """
    if not use_gpu:
        return model_name, None
    try:
        import torch

        if not torch.cuda.is_available():
            return model_name, None
        total = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return model_name, None

    gb = total / (1024**3)

    # Already a lighter checkpoint
    if any(
        x in model_name
        for x in (
            "2.5-VL-3B",
            "Qwen2-VL-2B",
            "2-VL-2B",
        )
    ):
        return model_name, None

    qwen3_4b_line = "Qwen3-VL-4B" in model_name
    is_fp8 = "FP8" in model_name

    # Prefer staying on Qwen3-VL-4B family (user-facing default): FP8 before swapping architectures.
    if gb < 16.0 and qwen3_4b_line and not is_fp8:
        return (
            "Qwen/Qwen3-VL-4B-Instruct-FP8",
            (
                f"CUDA device total memory ~{gb:.1f} GiB: prefer Qwen/Qwen3-VL-4B-Instruct-FP8 "
                f"over full-precision {model_name} to avoid Jetson NvMap/OOM during load."
            ),
        )

    if gb < 15.0:
        if ("7B" in model_name and "VL" in model_name) or "32B" in model_name:
            return (
                "Qwen/Qwen3-VL-4B-Instruct-FP8",
                (
                    f"CUDA device total memory ~{gb:.1f} GiB: {model_name} is likely too large; "
                    f"using Qwen/Qwen3-VL-4B-Instruct-FP8."
                ),
            )
        return model_name, None

    if gb < 24.0 and qwen3_4b_line and not is_fp8:
        return (
            "Qwen/Qwen3-VL-4B-Instruct-FP8",
            (
                f"CUDA device total memory ~{gb:.1f} GiB: using Qwen/Qwen3-VL-4B-Instruct-FP8 "
                f"instead of {model_name} to reduce peak memory."
            ),
        )

    return model_name, None


def is_likely_jetson() -> bool:
    if platform.system() != "Linux":
        return False
    if "tegra" in platform.release().lower():
        return True
    try:
        return os.path.isfile("/etc/nv_tegra_release")
    except OSError:
        return False


def _nvidia_smi_driver_line() -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().split("\n")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def log_cuda_setup_summary() -> None:
    """Print one block to stdout to help fix driver / torch mismatches."""
    try:
        import torch
    except ImportError:
        print("[torch_cuda_compat] PyTorch not installed; cannot use GPU for VLM/OCR.")
        return

    print("[torch_cuda_compat] PyTorch:", torch.__version__)
    print("[torch_cuda_compat] torch.version.cuda (build):", getattr(torch.version, "cuda", None))
    drv = _nvidia_smi_driver_line()
    if drv:
        print("[torch_cuda_compat] nvidia-smi driver_version:", drv)

    avail = torch.cuda.is_available()
    print("[torch_cuda_compat] torch.cuda.is_available():", avail)

    if avail:
        try:
            print("[torch_cuda_compat] device 0:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("[torch_cuda_compat] could not read device name:", e)
    else:
        if is_likely_jetson():
            print(
                "[torch_cuda_compat] Jetson detected: install PyTorch from NVIDIA's Jetson wheel "
                "for your JetPack (do not use generic x86 cu12 wheels). See:\n"
                "  https://developer.download.nvidia.com/compute/redist/jp/\n"
                "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/"
            )
        else:
            print(
                "[torch_cuda_compat] CUDA not usable: upgrade the NVIDIA driver or install a "
                "torch+torchvision wheel whose CUDA version matches your driver "
                "(see https://pytorch.org/get-started/locally/)."
            )


def probe_cuda_alloc() -> Tuple[bool, str]:
    """
    Verify PyTorch can allocate on CUDA (catches driver/runtime mismatch).
    Returns (ok, message).
    """
    try:
        import torch
    except ImportError as e:
        return False, f"PyTorch not importable: {e}"

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"

    try:
        t = torch.zeros(1, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        del t
        torch.cuda.empty_cache()
        return True, "ok"
    except Exception as e:
        return False, str(e)
