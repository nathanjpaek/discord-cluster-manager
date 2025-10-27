#!/usr/bin/env python3
import argparse
import base64
import io
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


@dataclass
class GitHubConfig:
    token: str
    repo: str
    workflow_file: str = ".github/workflows/test.yml"
    ref: str = "main"
    runner_label: str = "self-hosted"


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / ".git").exists():
            return parent
    return start


def build_payload_from_cuda(submission_path: Path, repo_root: Path) -> str:
    """Build base64(zlib(json)) payload expected by src/runners/github-runner.py.

    The config mirrors libkernelbot.task.build_task_config for a CUDA task, but
    we inline the minimal files from examples/ to make this standalone.
    """
    import zlib

    examples = repo_root / "examples"
    # Required support files
    files = {
        "eval.cu": (examples / "eval.cu").read_text(),
        "task.h": (examples / "identity_cuda" / "task.h").read_text(),
        "utils.h": (examples / "utils.h").read_text(),
        "reference.cuh": (examples / "identity_cuda" / "reference.cuh").read_text(),
        "submission.cu": submission_path.read_text(),
    }

    # Minimal set of tests; benchmarks optional
    config = {
        "lang": "cu",
        "mode": "test",
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

    payload_json = json.dumps(config).encode("utf-8")
    payload_b64 = base64.b64encode(zlib.compress(payload_json)).decode("utf-8")
    return payload_b64


def dispatch_workflow(
    cfg: GitHubConfig,
    payload_b64: str,
    *,
    run_id: Optional[str] = None,
    runner: Optional[str] = None,
) -> None:
    url = f"https://api.github.com/repos/{cfg.repo}/actions/workflows/{cfg.workflow_file}/dispatches"
    headers = {
        "Authorization": f"Bearer {cfg.token}",
        "Accept": "application/vnd.github+json",
    }
    inputs = {"payload": payload_b64}
    if run_id:
        inputs["run_id"] = run_id
    if runner:
        inputs["runner"] = runner
    data = {"ref": cfg.ref, "inputs": inputs}
    r = requests.post(url, headers=headers, json=data, timeout=30)
    if r.status_code != 204:
        raise RuntimeError(f"Failed to dispatch workflow: {r.status_code} {r.text}")


def _find_recent_run_id(cfg: GitHubConfig, since_ts: float) -> Optional[int]:
    url = f"https://api.github.com/repos/{cfg.repo}/actions/workflows/{cfg.workflow_file}/runs"
    headers = {
        "Authorization": f"Bearer {cfg.token}",
        "Accept": "application/vnd.github+json",
    }
    # Iterate a few pages in case there are many runs
    for page in range(1, 6):
        r = requests.get(url, headers=headers, params={"event": "workflow_dispatch", "per_page": 50, "page": page}, timeout=30)
        r.raise_for_status()
        runs = r.json().get("workflow_runs", [])
        for run in runs:
            created_at = run.get("created_at")
            # Try to ensure it's after we triggered (approx)
            # Convert created_at RFC3339 to epoch
            try:
                from datetime import datetime
                ca = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
            except Exception:
                ca = 0
            if ca >= since_ts - 30:
                return int(run.get("id"))
        if not runs:
            break
    return None


def wait_for_run_completion(cfg: GitHubConfig, run_pk: int, poll_s: int = 5, timeout_s: int = 1800) -> dict:
    url = f"https://api.github.com/repos/{cfg.repo}/actions/runs/{run_pk}"
    headers = {
        "Authorization": f"Bearer {cfg.token}",
        "Accept": "application/vnd.github+json",
    }
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        conclusion = data.get("conclusion")
        if status != last_status:
            print(f"status: {status}")
            last_status = status
        if status == "completed":
            return data
        time.sleep(poll_s)
    raise TimeoutError("Run did not complete in time")


def download_result_artifact(cfg: GitHubConfig, run_pk: int) -> Optional[dict]:
    list_url = f"https://api.github.com/repos/{cfg.repo}/actions/runs/{run_pk}/artifacts"
    headers = {
        "Authorization": f"Bearer {cfg.token}",
        "Accept": "application/vnd.github+json",
    }
    r = requests.get(list_url, headers=headers, timeout=30)
    r.raise_for_status()
    artifacts = r.json().get("artifacts", [])
    target = None
    for a in artifacts:
        if a.get("name") == "run-result":
            target = a
            break
    if not target:
        return None

    download_url = target.get("archive_download_url")
    r = requests.get(download_url, headers=headers, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for name in zf.namelist():
            if name.endswith("result.json"):
                with zf.open(name) as f:
                    return json.loads(f.read().decode("utf-8"))
    return None


def format_result_summary(result: dict) -> str:
    if not result:
        return "No result.json artifact found."
    # Best-effort concise summary
    lines = []
    runs = result.get("runs", {})
    for phase in ["test", "benchmark", "leaderboard"]:
        if phase in runs:
            run = runs[phase]
            passed = run.get("run", {}).get("passed")
            dur = run.get("run", {}).get("duration")
            lines.append(f"{phase}: passed={passed} duration={dur}")
    if not lines:
        lines.append(json.dumps(result, indent=2))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Submit a CUDA kernel via GitHub Actions and print results")
    parser.add_argument("submission", type=str, help="Path to .cu file to submit")
    parser.add_argument("--run-id", type=str, default=None, help="If provided, include 'run_id' input (required for some workflows)")
    args = parser.parse_args()

    gh_token = os.getenv("GITHUB_TOKEN")
    if not gh_token:
        print("GITHUB_TOKEN is required in environment (with 'workflow' scope)")
        sys.exit(1)

    gh_repo = os.getenv("GITHUB_REPO", "Willy-Chan/discord-cluster-manager")
    gh_workflow = os.getenv("GITHUB_WORKFLOW", ".github/workflows/test.yml")
    gh_ref = os.getenv("GITHUB_REF", "main")
    gh_runner = os.getenv("GITHUB_RUNNER_LABEL", "self-hosted")

    cfg = GitHubConfig(token=gh_token, repo=gh_repo, workflow_file=gh_workflow, ref=gh_ref, runner_label=gh_runner)

    submission_path = Path(args.submission)
    if not submission_path.exists():
        print(f"Submission file not found: {submission_path}")
        sys.exit(1)

    repo_root = _find_repo_root(Path(__file__).parent)
    payload_b64 = build_payload_from_cuda(submission_path, repo_root)

    default_run_id = f"cli-{int(time.time())}"
    effective_run_id = args.run_id if args.run_id else None
    print(f"Dispatching workflow {cfg.workflow_file} on {cfg.repo} ...")
    started_ts = time.time()
    dispatch_workflow(cfg, payload_b64, run_id=effective_run_id, runner=(gh_runner or None))

    print("Waiting for GitHub to create the run...")
    run_pk = None
    for _ in range(60):  # up to 60 * 2s = 2 minutes to appear
        run_pk = _find_recent_run_id(cfg, since_ts=started_ts)
        if run_pk:
            break
        time.sleep(2)
    if not run_pk:
        print("Could not locate created workflow run. Check Actions UI.")
        sys.exit(1)

    print(f"Run id: {run_pk}")
    run_data = wait_for_run_completion(cfg, run_pk)
    conclusion = run_data.get("conclusion")
    html_url = run_data.get("html_url")
    print(f"conclusion: {conclusion}")
    print(f"url: {html_url}")

    result = download_result_artifact(cfg, run_pk)
    print("\n=== Result Summary ===")
    print(format_result_summary(result))


if __name__ == "__main__":
    main()


