# SPDX-License-Identifier: Apache-2.0
"""Validate both historical lock variants without modifying the source tree."""

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    uv = os.environ.get("UV_BIN") or shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to validate frozen dependency locks")
    for suffix in ("", ".vllm"):
        project = root / f"pyproject{suffix}.toml"
        lock = root / f"uv{suffix}.lock"
        digest = hashlib.sha256(lock.read_bytes()).digest()
        with tempfile.TemporaryDirectory(prefix="areal-frozen-lock-") as directory:
            target = Path(directory)
            shutil.copy2(project, target / "pyproject.toml")
            shutil.copy2(lock, target / "uv.lock")
            subprocess.run(
                [uv, "lock", "--locked", "--offline"], cwd=target, check=True
            )
            if hashlib.sha256((target / "uv.lock").read_bytes()).digest() != digest:
                raise RuntimeError(f"Frozen lock content changed: {lock.name}")
        print(f"Validated unchanged: {lock.name}")


if __name__ == "__main__":
    main()
