"""Real two-process Gloo receipt for the native dense DDP backend."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("torch")


def test_two_process_dense_ddp_commits_identical_parameters(tmp_path):
    result_path = tmp_path / "result.json"
    environment = dict(os.environ)
    # The suite pins BLAS threads in the parent. On macOS, inheriting those
    # variables into torchrun can stall the elastic rendezvous before workers
    # spawn; torchrun sets OMP_NUM_THREADS for its workers itself.
    environment.pop("OMP_NUM_THREADS", None)
    environment.pop("MKL_NUM_THREADS", None)
    environment["MIXLE_RESULT_PATH"] = str(result_path)
    environment["MASTER_ADDR"] = "127.0.0.1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--rdzv_endpoint=127.0.0.1:0",
            "--local-addr=127.0.0.1",
            "--nproc_per_node=2",
            "-m",
            "mixle.tests._distributed_training_worker",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload == {
        "parameters_equal": True,
        "step": 1,
        "global_examples": 8,
        "global_tokens": 32,
    }
