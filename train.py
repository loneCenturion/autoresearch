"""
Single entry point for Safe-OS minimal experiments.

This script preserves the autoresearch pattern:
- one fixed preparation/evaluation layer in prepare.py;
- one primary experiment file (this file) that the agent iterates on;
- one standardized summary that can be grepped after each run.

Usage:
    python train.py
    python train.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from prepare import (
    BASELINE_PROGRESS_PATH,
    CODE_ROOT,
    DATA_ROOT,
    FULL_DATA_PATH,
    MINIMAL_DATA_PATH,
    REPO_ROOT,
    RESULTS_ROOT,
    RUNS_ROOT,
    SOURCE_SKILLS_DB,
    compare_to_baseline,
    derive_metrics,
    load_json,
    load_progress_metrics,
    verify_required_paths,
    write_json,
)

# ---------------------------------------------------------------------------
# Experiment configuration (edit these directly)
# ---------------------------------------------------------------------------

RUN_NAME: str | None = None
RUN_PREFIX = "safeos_minimal"
DATASET_KIND = "minimal"  # "minimal" or "full"
MODEL_NAME = "gpt-5.2"
MAX_SAMPLES = 20
CHECKPOINT_EVERY = 10
USE_V5 = True
RESUME = False

SQUIRL_REQUIRED_MODULES = [
    "numpy",
    "tqdm",
    "openai",
    "pydantic",
]
RUNTIME_VENV_DIR = RESULTS_ROOT / "runtime-venv"
EVOLVE_WRAPPER = REPO_ROOT / "run_evolve_train.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Safe-OS autoresearch experiment")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and paths without launching the experiment",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Override the auto-generated run directory name",
    )
    return parser.parse_args()


def python_candidates() -> list[Path]:
    candidates: list[Path] = []

    env_python = os.environ.get("AUTORESEARCH_PYTHON")
    if env_python:
        candidates.append(Path(env_python))

    candidates.extend(
        [
            RUNTIME_VENV_DIR / "bin" / "python",
            REPO_ROOT / ".venv" / "bin" / "python",
            CODE_ROOT / ".venv" / "bin" / "python",
        ]
    )

    system_python = shutil.which("python3")
    if system_python:
        candidates.append(Path(system_python))

    candidates.append(Path(sys.executable))
    return candidates


def resolve_python_bin() -> Path:
    seen: set[str] = set()
    for candidate in python_candidates():
        resolved = str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return Path(sys.executable)


def runtime_missing_modules(python_bin: Path) -> list[str]:
    script = (
        "import importlib.util; "
        f"mods = {SQUIRL_REQUIRED_MODULES!r}; "
        "missing = [m for m in mods if importlib.util.find_spec(m) is None]; "
        "print('\\n'.join(missing))"
    )
    completed = subprocess.run(
        [str(python_bin), "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ["__python_check_failed__"]
    stdout = completed.stdout.strip()
    return [line for line in stdout.splitlines() if line.strip()]


def print_runtime_help(python_bin: Path, missing_modules: list[str]) -> None:
    print(f"Runtime python:  {python_bin}")
    if missing_modules == ["__python_check_failed__"]:
        print("Dependency preflight failed before the experiment could start.")
    else:
        print("Missing SQUIRL runtime modules:")
        for module in missing_modules:
            print(f"  - {module}")
    print()
    print("Recommended fix:")
    print(
        "  "
        f"env UV_CACHE_DIR=/tmp/uv-cache uv venv {RUNTIME_VENV_DIR}"
    )
    print(
        "  "
        f"env UV_CACHE_DIR=/tmp/uv-cache uv pip install -p {RUNTIME_VENV_DIR / 'bin' / 'python'} -r /data/Agent_Defense/code/SQUIRL/requirements.txt"
    )
    print("Then rerun `python train.py`.")


def selected_dataset_path() -> Path:
    if DATASET_KIND == "minimal":
        return MINIMAL_DATA_PATH
    if DATASET_KIND == "full":
        return FULL_DATA_PATH
    raise ValueError(f"Unsupported DATASET_KIND: {DATASET_KIND}")


def dataset_slice_summary(dataset_path: Path, max_samples: int) -> dict[str, int]:
    data = load_json(dataset_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {dataset_path}")

    if max_samples < 0:
        sliced = data
    else:
        sliced = data[:max_samples]

    safe_count = sum(1 for item in sliced if item.get("labels", 1) == 0)
    unsafe_count = sum(1 for item in sliced if item.get("labels", 0) == 1)
    return {
        "slice_total": len(sliced),
        "slice_safe": safe_count,
        "slice_unsafe": unsafe_count,
    }


def build_output_name(cli_output_name: str | None) -> str:
    if cli_output_name:
        return cli_output_name
    if RUN_NAME:
        return RUN_NAME

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{RUN_PREFIX}_{DATASET_KIND}_{MAX_SAMPLES}_{timestamp}"


def build_command(python_bin: Path, dataset_path: Path, output_path: Path) -> list[str]:
    command = [
        str(python_bin),
        str(EVOLVE_WRAPPER),
        "--train_data",
        str(dataset_path),
        "--skills_db",
        str(SOURCE_SKILLS_DB),
        "--output_path",
        str(output_path),
        "--model",
        MODEL_NAME,
        "--max_samples",
        str(MAX_SAMPLES),
        "--checkpoint_every",
        str(CHECKPOINT_EVERY),
    ]
    if USE_V5:
        command.append("--v5")
    if not RESUME:
        command.append("--no_resume")
    return command


def build_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("SQUIRL_EMBEDDING_BACKEND", "hash")
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    env.setdefault("USE_TF", "0")
    return env


def read_tail(path: Path, num_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-num_lines:])


def analyze_launcher_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "connection_error_count": 0,
            "mock_mode": False,
        }

    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "connection_error_count": text.count("Connection error."),
        "mock_mode": "MOCK mode" in text,
    }


def summary_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_summary(summary: dict[str, Any]) -> None:
    ordered_keys = [
        "output_path",
        "launcher_log",
        "summary_json",
        "dataset_path",
        "dataset_kind",
        "slice_total",
        "slice_safe",
        "slice_unsafe",
        "run_status",
        "return_code",
        "run_seconds",
        "connection_error_count",
        "mock_mode",
        "processed",
        "skipped",
        "evaluated",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "benign_learned",
        "updates",
        "precision",
        "recall",
        "f1",
        "specificity",
        "accuracy",
        "skip_rate",
        "failure_rate",
        "baseline_recall",
        "baseline_specificity",
        "delta_false_positive",
        "delta_specificity",
        "delta_skip_rate",
        "relative_recall_drop",
        "recall_within_tolerance",
        "hard_gate_pass",
    ]

    print("---")
    for key in ordered_keys:
        if key in summary:
            print(f"{key + ':':20s} {summary_value(summary[key])}")


def main() -> int:
    args = parse_args()

    missing = verify_required_paths()
    if missing:
        print("Missing required external inputs:")
        for name, path in missing:
            print(f"  - {name}: {path}")
        return 1

    dataset_path = selected_dataset_path()
    if not dataset_path.exists():
        print(f"Missing prepared dataset: {dataset_path}")
        print("Run `python prepare.py` first.")
        return 1

    python_bin = resolve_python_bin()
    missing_modules = runtime_missing_modules(python_bin)
    if missing_modules:
        print_runtime_help(python_bin, missing_modules)
        return 1
    if not EVOLVE_WRAPPER.exists():
        print(f"Missing local wrapper: {EVOLVE_WRAPPER}")
        return 1

    slice_summary = dataset_slice_summary(dataset_path, MAX_SAMPLES)
    output_name = build_output_name(args.output_name)
    output_path = RUNS_ROOT / output_name
    launcher_log = output_path / "launcher.log"
    summary_json = output_path / "summary.json"

    if output_path.exists() and not RESUME:
        print(f"Refusing to overwrite existing run directory: {output_path}")
        print("Set RUN_NAME / --output-name to a new value or enable RESUME.")
        return 1

    command = build_command(python_bin, dataset_path, output_path)

    print(f"Repository root: {Path(__file__).resolve().parent}")
    print(f"Results root:    {RESULTS_ROOT}")
    print(f"Data root:       {DATA_ROOT}")
    print(f"Working dir:     {CODE_ROOT}")
    print(f"Runtime python:  {python_bin}")
    print(f"Output path:     {output_path}")
    print(f"Command:         {shlex.join(command)}")
    print(
        "Dataset slice:   "
        f"{slice_summary['slice_safe']} safe / {slice_summary['slice_unsafe']} unsafe "
        f"(total={slice_summary['slice_total']})"
    )

    if args.dry_run:
        return 0

    output_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with launcher_log.open("w", encoding="utf-8") as handle:
        handle.write("# Autoresearch Safe-OS launcher log\n")
        handle.write(f"# command = {shlex.join(command)}\n")
        handle.write(f"# cwd = {CODE_ROOT}\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=CODE_ROOT,
            env=build_runtime_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    run_seconds = time.time() - t0

    progress_path = output_path / "checkpoints" / "progress.json"
    if not progress_path.exists():
        print(f"Experiment did not produce progress.json: {progress_path}")
        tail = read_tail(launcher_log)
        if tail:
            print()
            print("Last launcher log lines:")
            print(tail)
        return completed.returncode or 1

    progress_payload = load_json(progress_path)
    stats = progress_payload.get("stats", {})
    metrics = derive_metrics(stats)
    baseline_metrics = load_progress_metrics(BASELINE_PROGRESS_PATH) if BASELINE_PROGRESS_PATH.exists() else None
    comparison = compare_to_baseline(metrics, baseline_metrics)
    launcher_diagnostics = analyze_launcher_log(launcher_log)

    evaluated = int(metrics.get("evaluated", 0))
    if completed.returncode != 0:
        run_status = "launcher_failed"
    elif evaluated == 0 and launcher_diagnostics["connection_error_count"] > 0:
        run_status = "blocked_by_model_connectivity"
    elif evaluated == 0:
        run_status = "no_evaluated_samples"
    else:
        run_status = "ok"

    summary: dict[str, Any] = {
        "output_path": output_path,
        "launcher_log": launcher_log,
        "summary_json": summary_json,
        "dataset_path": dataset_path,
        "dataset_kind": DATASET_KIND,
        "slice_total": slice_summary["slice_total"],
        "slice_safe": slice_summary["slice_safe"],
        "slice_unsafe": slice_summary["slice_unsafe"],
        "run_status": run_status,
        "return_code": completed.returncode,
        "run_seconds": run_seconds,
    }
    summary.update(launcher_diagnostics)
    summary.update(metrics)
    summary.update(comparison)

    write_json(summary_json, summary)
    print_summary(summary)

    if completed.returncode != 0:
        print()
        print("Experiment process exited non-zero. Inspect the launcher log.")
        tail = read_tail(launcher_log)
        if tail:
            print()
            print("Last launcher log lines:")
            print(tail)
        return completed.returncode

    if evaluated == 0:
        print()
        print("Experiment did not evaluate any samples. Treat this run as invalid.")
        if launcher_diagnostics["connection_error_count"] > 0:
            print(
                "Detected model API connection failures in launcher.log. "
                "Rerun with network access or verify OPENAI_BASE_URL / OPENAI_API_KEY."
            )
        else:
            print("Inspect the launcher log for skipped-sample causes before iterating on metrics.")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
