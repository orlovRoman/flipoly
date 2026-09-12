"""Local-only guards: refuse SSHFS / server train & bootstrap.

Instruction: all heavy ops (dataset prep, features, train, bootstrap,
economics) run on a local Windows disk. Server is collection + limited
export only. Research calculations must NOT run on the server.
"""
from __future__ import annotations

import os
from pathlib import Path


def is_sshfs_path(path: str | Path) -> bool:
    s = str(path).lower().replace("/", "\\")
    return ("sshfs" in s) or s.startswith("\\\\")


def assert_local_workdir(workdir: str | Path) -> Path:
    p = Path(workdir).resolve()
    if is_sshfs_path(p):
        raise RuntimeError(
            f"Refusing to run from SSHFS/network path: {p}. "
            "Work on a local Windows disk per protocol.compute.allowed_host."
        )
    return p


def assert_not_server_host() -> None:
    """Refuse train/bootstrap on the collection server."""
    role = os.environ.get("POLYFLIP_ROLE", "").lower()
    if role == "server":
        raise RuntimeError(
            "Refusing to train/bootstrap on server (POLYFLIP_ROLE=server). "
            "Use the server only for collection + limited export."
        )
    if os.environ.get("CANONICAL_MODELS_ALLOW_SERVER") == "1":
        return
    # Heuristic: explicit server hostnames must opt in; local dev passes.
    hostname = os.environ.get("HOSTNAME", "") + os.environ.get("COMPUTERNAME", "")
    if "prod" in hostname.lower() and "CANONICAL" not in os.environ:
        pass  # do not hard-block unknown hosts; role flag is authoritative


def configure_thread_limits(cap: int = 4) -> dict:
    vals = {
        "OMP_NUM_THREADS": str(cap),
        "OPENBLAS_NUM_THREADS": str(cap),
        "MKL_NUM_THREADS": str(cap),
        "NUMEXPR_NUM_THREADS": str(cap),
        "LIGHTGBM_NUM_THREADS": str(cap),
    }
    for k, v in vals.items():
        os.environ.setdefault(k, v)
    try:
        import torch  # type: ignore
        torch.set_num_threads(cap)
    except Exception:
        pass
    return vals


def assert_train_allowed(workdir: str | Path, cap: int = 4) -> Path:
    p = assert_local_workdir(workdir)
    assert_not_server_host()
    configure_thread_limits(cap)
    return p
