from __future__ import annotations

import torch

from workbench.config import FORCE_CPU


def accelerator_kind() -> str:
    """Coarse accelerator label for telemetry and UX copy."""

    hip_ver = getattr(getattr(torch, "version", None), "hip", None)

    if FORCE_CPU:
        return "cpu (forced)"

    if torch.cuda.is_available():
        cuda_name = ""
        try:
            cuda_name = torch.cuda.get_device_name(0) or ""
        except Exception:
            pass

        if hip_ver:
            base = "ROCm (CUDA API)"
            if cuda_name:
                return f"{base}: {cuda_name} (HIP={hip_ver})"
            return f"{base} (HIP={hip_ver})"

        if cuda_name:
            return f"CUDA: {cuda_name}"

        return "CUDA"

    mps_ready = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    return "Apple MPS" if mps_ready else "cpu"


def use_gpu_inference() -> bool:
    """Whether we attempt to load generation models on GPU (CUDA/ROCm/MPS)."""

    if FORCE_CPU:
        return False

    if torch.cuda.is_available():
        return True

    mps_ready = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    return bool(mps_ready)
