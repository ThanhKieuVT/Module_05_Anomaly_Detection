from __future__ import annotations

import logging
import random
import subprocess
import time
from contextlib import contextmanager

import numpy as np
import torch

log = logging.getLogger("fscad")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def timer(store: dict, key: str, device: torch.device | None = None):
    if device is not None:
        sync(device)
    t0 = time.perf_counter()
    yield
    if device is not None:
        sync(device)
    store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def git_commit() -> tuple[str, bool]:
    """(short commit hash, working tree dirty?)"""
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True
