#!/usr/bin/env python3
"""
Minimal local eval script - mirrors submit_github.py but runs locally.

Just like submit_github.py builds a simple config and sends it to GitHub runners,
this builds the same config and calls run_config() directly on your local machine.

Usage:
    python scripts/run_local_eval.py test_kernel.cu --mode profile
    python scripts/run_local_eval.py test_kernel.cu --mode test
    python scripts/run_local_eval.py test_kernel.cu --mode benchmark
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / ".git").exists():
            return parent
    return cur


REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from libkernelbot.run_eval import run_config  # noqa: E402

try:
    from submit_github import format_result_summary  # noqa: E402
except Exception:
    def format_result_summary(result: dict) -> str:
        lines = ["=== Result ===", f"Success: {result.get('success', False)}"]
        if result.get('error'):
            lines.append(f"Error: {result['error']}")
        for phase, data in result.get("runs", {}).items():
            comp = data.get("compilation", {})
            run = data.get("run", {})
            lines.append(f"[{phase}] compiled={comp.get('success', False)} passed={run.get('passed', False)}")
        return "\n".join(lines)


def build_config_from_cuda(submission_path: Path, mode: str = "test") -> dict:
    """
    Build the exact same config dict that submit_github.py creates.
    Mirrors build_payload_from_cuda() but returns dict instead of encoding it.
    """
    examples = REPO_ROOT / "examples"
    
    # Read all required files
    files = {
        "eval.cu": (examples / "eval.cu").read_text(),
        "task.h": (examples / "identity_cuda" / "task.h").read_text(),
        "utils.h": (examples / "utils.h").read_text(),
        "reference.cuh": (examples / "identity_cuda" / "reference.cuh").read_text(),
        "submission.cu": submission_path.read_text(),
    }

    # Build the config - exactly like submit_github.py
    config = {
        "lang": "cu",
        "mode": mode,
        "arch": None,
        "sources": {
            "eval.cu": files["eval.cu"],
            "submission.cu": files["submission.cu"],
        },
        "headers": {
            "task.h": files["task.h"],
            "utils.h": files["utils.h"],
            "reference.cuh": files["reference.cuh"],
        },
        "defines": {},
        "include_dirs": [],
        "libraries": [],
        "tests": [{"size": 128, "seed": 1}],
        "benchmarks": [],
        "test_timeout": 180,
        "benchmark_timeout": 180,
        "ranked_timeout": 180,
        "ranking_by": "last",
        "seed": None,
        "multi_gpu": False,
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run kernel eval locally (mirrors submit_github.py workflow)"
    )
    parser.add_argument("submission", help="Path to .cu file")
    parser.add_argument(
        "--mode",
        default="test",
        choices=["test", "benchmark", "profile"],
        help="Execution mode",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full JSON result",
    )
    parser.add_argument(
        "--save-result",
        help="Save result JSON to this path",
    )
    args = parser.parse_args()

    submission_path = Path(args.submission).resolve()
    if not submission_path.exists():
        print(f"Error: {submission_path} not found")
        sys.exit(1)

    print(f"Running {submission_path.name} in '{args.mode}' mode...")
    
    # Build config exactly like submit_github.py does
    config = build_config_from_cuda(submission_path, args.mode)
    
    # Change to repo root so relative paths work
    os.chdir(REPO_ROOT)
    
    # Run it!
    result = run_config(config)
    result_dict = asdict(result)

    # Show summary
    print("\n" + "="*70)
    print(format_result_summary(result_dict))
    
    # Show where profile data is if profiling was done
    if args.mode == "profile" and result_dict.get("success"):
        profile_dir = Path.cwd() / "profile_data"
        if profile_dir.exists():
            print(f"\n📊 Profile data saved to: {profile_dir.resolve()}")

    # Optional: print/save full JSON
    if args.print_json:
        print("\n=== Full JSON ===")
        print(json.dumps(result_dict, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o), indent=2))
    
    if args.save_result:
        Path(args.save_result).write_text(
            json.dumps(result_dict, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o), indent=2)
        )
        print(f"💾 Saved result to: {args.save_result}")


if __name__ == "__main__":
    main()


