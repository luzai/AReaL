"""Regress worker logging wrappers that must preserve child failures."""

import subprocess
import time

from areal.infra.utils.proc import build_streaming_log_cmd


def test_streaming_log_pipeline_preserves_worker_exit_code(tmp_path):
    log_file = tmp_path / "worker.log"
    merged_log = tmp_path / "merged.log"
    command = build_streaming_log_cmd(
        ["bash", "-c", "echo worker-failed; exit 23"],
        str(log_file),
        str(merged_log),
        "actor",
    )

    result = subprocess.run(
        command,
        shell=True,
        executable="/bin/bash",
        check=False,
    )

    assert result.returncode == 23
    assert "worker-failed" in log_file.read_text()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if merged_log.exists() and "worker-failed" in merged_log.read_text():
            break
        time.sleep(0.01)
    assert "worker-failed" in merged_log.read_text()
