#!/usr/bin/env python3
import base64
import json
import zlib
from pathlib import Path

# Read the example files
eval_cu = Path("examples/eval.cu").read_text()
task_h = Path("examples/identity_cuda/task.h").read_text()
utils_h = Path("examples/utils.h").read_text()
reference_cuh = Path("examples/identity_cuda/reference.cuh").read_text()
submission_cu = Path("test_kernel.cu").read_text()

# Create the config
config = {
    "lang": "cu",
    "mode": "test",
    "arch": None,
    "sources": {
        "eval.cu": eval_cu,
        "submission.cu": submission_cu
    },
    "headers": {
        "task.h": task_h,
        "utils.h": utils_h,
        "reference.cuh": reference_cuh
    },
    "defines": {},
    "include_dirs": [],
    "libraries": [],
    "tests": [
        {"size": 127, "seed": 4242},
        {"size": 128, "seed": 5236}
    ],
    "benchmarks": [
        {"size": 1024, "seed": 54352}
    ],
    "test_timeout": 180,
    "benchmark_timeout": 180,
    "ranked_timeout": 180,
    "ranking_by": "last",
    "seed": None,
    "multi_gpu": False
}

# Compress and encode
payload = base64.b64encode(zlib.compress(json.dumps(config).encode("utf-8"))).decode("utf-8")

print("=" * 80)
print("PAYLOAD FOR GITHUB ACTIONS:")
print("=" * 80)
print(payload)
print("=" * 80)
print("\nCopy the payload above and use it in the GitHub Actions workflow")
print("Inputs:")
print(f"  run_id: test-{hash(payload) % 10000}")
print(f"  payload: <paste the payload above>")
print(f"  runner: self-hosted")


