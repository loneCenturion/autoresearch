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
SKILLS_SOURCE_MODE = "latest-keep"  # "base" or "latest-keep"
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
RESULTS_TSV = REPO_ROOT / "results.tsv"


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
    parser.add_argument(
        "--dataset-kind",
        choices=("minimal", "full"),
        default=None,
        help="Override the dataset view for this run without editing train.py constants",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override the training slice size for this run (-1 means full prepared dataset)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Override the checkpoint frequency for this run",
    )
    parser.add_argument(
        "--skills-source",
        choices=("base", "latest-keep"),
        default=None,
        help="Choose whether to start from the upstream base skills or the latest keep run's evolved skills.",
    )
    parser.add_argument(
        "--skills-db-path",
        default=None,
        help="Explicit skills_db path override. Takes precedence over --skills-source.",
    )
    parser.add_argument(
        "--allow-debug-git-state",
        action="store_true",
        help="Bypass formal experiment git checks (master / non-autoresearch branch / dirty worktree). Use only for debug runs.",
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


def selected_dataset_path(dataset_kind: str) -> Path:
    if dataset_kind == "minimal":
        return MINIMAL_DATA_PATH
    if dataset_kind == "full":
        return FULL_DATA_PATH
    raise ValueError(f"Unsupported DATASET_KIND: {dataset_kind}")


def run_git_command(*git_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *git_args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def resolve_git_state() -> dict[str, Any]:
    branch_cmd = run_git_command("branch", "--show-current")
    commit_cmd = run_git_command("rev-parse", "--short", "HEAD")
    status_cmd = run_git_command("status", "--short")

    available = all(cmd.returncode == 0 for cmd in (branch_cmd, commit_cmd, status_cmd))
    branch = branch_cmd.stdout.strip() if branch_cmd.returncode == 0 else ""
    commit = commit_cmd.stdout.strip() if commit_cmd.returncode == 0 else ""
    status_lines = [line for line in status_cmd.stdout.splitlines() if line.strip()] if status_cmd.returncode == 0 else []

    return {
        "available": available,
        "branch": branch or "unknown",
        "commit": commit or "unknown",
        "status_lines": status_lines,
        "worktree_dirty": bool(status_lines),
        "formal_branch": branch.startswith("autoresearch/"),
        "on_master": branch == "master",
    }


def validate_formal_git_state(git_state: dict[str, Any]) -> list[str]:
    if not git_state["available"]:
        return ["Unable to resolve git branch / commit / status information."]

    errors: list[str] = []
    if git_state["on_master"]:
        errors.append("Formal experiments cannot run on `master`.")
    elif not git_state["formal_branch"]:
        errors.append(
            f"Formal experiments must run on an `autoresearch/*` branch (current: {git_state['branch']})."
        )

    if git_state["worktree_dirty"]:
        errors.append("Formal experiments require a clean worktree.")

    return errors


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


def build_output_name(cli_output_name: str | None, dataset_kind: str, max_samples: int, git_commit: str) -> str:
    if cli_output_name:
        return cli_output_name
    if RUN_NAME:
        return RUN_NAME

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{RUN_PREFIX}_{dataset_kind}_{max_samples}_{git_commit}_{timestamp}"


def load_results_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return []

    header = lines[0].split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        if len(values) < len(header):
            values.extend([""] * (len(header) - len(values)))
        rows.append(dict(zip(header, values)))
    return rows


def latest_keep_skills_db() -> Path | None:
    for row in reversed(load_results_rows(RESULTS_TSV)):
        if row.get("status") != "keep":
            continue
        run_name = row.get("run_name", "").strip()
        if not run_name:
            continue
        candidate = RUNS_ROOT / run_name / "skills_evolved"
        if (candidate / "skills.json").exists() and (candidate / "skills_v5.json").exists():
            return candidate
    return None


def resolve_skills_db(cli_path: str | None, source_mode: str) -> tuple[Path, str]:
    if cli_path:
        return Path(cli_path), "explicit"

    if source_mode == "base":
        return SOURCE_SKILLS_DB, "base"

    latest_keep = latest_keep_skills_db()
    if latest_keep is not None:
        return latest_keep, "latest_keep"

    return SOURCE_SKILLS_DB, "base_fallback"


def build_command(
    python_bin: Path,
    dataset_path: Path,
    skills_db: Path,
    output_path: Path,
    model_name: str,
    max_samples: int,
    checkpoint_every: int,
) -> list[str]:
    command = [
        str(python_bin),
        str(EVOLVE_WRAPPER),
        "--train_data",
        str(dataset_path),
        "--skills_db",
        str(skills_db),
        "--output_path",
        str(output_path),
        "--model",
        model_name,
        "--max_samples",
        str(max_samples),
        "--checkpoint_every",
        str(checkpoint_every),
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
        "skills_db",
        "skills_db_source",
        "git_branch",
        "git_commit",
        "git_worktree_dirty",
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
    dataset_kind = args.dataset_kind or DATASET_KIND
    max_samples = args.max_samples if args.max_samples is not None else MAX_SAMPLES
    checkpoint_every = args.checkpoint_every if args.checkpoint_every is not None else CHECKPOINT_EVERY
    skills_source_mode = args.skills_source or SKILLS_SOURCE_MODE
    git_state = resolve_git_state()
    git_errors = validate_formal_git_state(git_state)

    if not args.dry_run and git_errors and not args.allow_debug_git_state:
        print("Refusing to start a formal experiment with the current git state:")
        for error in git_errors:
            print(f"  - {error}")
        print(f"Git branch:      {git_state['branch']}")
        print(f"Git commit:      {git_state['commit']}")
        if git_state["status_lines"]:
            print("Git status:")
            for line in git_state["status_lines"]:
                print(f"  {line}")
        else:
            print("Git status:      clean")
        print()
        print("Commit or switch branches first, or rerun with `--allow-debug-git-state` for a non-formal debug run.")
        return 1

    missing = verify_required_paths()
    if missing:
        print("Missing required external inputs:")
        for name, path in missing:
            print(f"  - {name}: {path}")
        return 1

    dataset_path = selected_dataset_path(dataset_kind)
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

    skills_db_path, skills_db_source = resolve_skills_db(args.skills_db_path, skills_source_mode)
    if not skills_db_path.exists():
        print(f"Missing skills_db: {skills_db_path}")
        return 1

    slice_summary = dataset_slice_summary(dataset_path, max_samples)
    output_name = build_output_name(args.output_name, dataset_kind, max_samples, git_state["commit"])
    output_path = RUNS_ROOT / output_name
    launcher_log = output_path / "launcher.log"
    summary_json = output_path / "summary.json"

    if output_path.exists() and not RESUME:
        print(f"Refusing to overwrite existing run directory: {output_path}")
        print("Set RUN_NAME / --output-name to a new value or enable RESUME.")
        return 1

    command = build_command(
        python_bin,
        dataset_path,
        skills_db_path,
        output_path,
        MODEL_NAME,
        max_samples,
        checkpoint_every,
    )

    print(f"Repository root: {Path(__file__).resolve().parent}")
    print(f"Results root:    {RESULTS_ROOT}")
    print(f"Data root:       {DATA_ROOT}")
    print(f"Working dir:     {CODE_ROOT}")
    print(f"Runtime python:  {python_bin}")
    print(f"Git branch:      {git_state['branch']}")
    print(f"Git commit:      {git_state['commit']}")
    print(f"Git status:      {'dirty' if git_state['worktree_dirty'] else 'clean'}")
    print(f"Skills DB:       {skills_db_path}")
    print(f"Skills source:   {skills_db_source}")
    if git_state["status_lines"]:
        for line in git_state["status_lines"]:
            print(f"  {line}")
    if git_errors and args.allow_debug_git_state:
        print("Git policy:      debug override enabled")
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
        handle.write(f"# git_branch = {git_state['branch']}\n")
        handle.write(f"# git_commit = {git_state['commit']}\n")
        handle.write(f"# git_worktree_dirty = {git_state['worktree_dirty']}\n")
        handle.write(f"# skills_db = {skills_db_path}\n")
        handle.write(f"# skills_db_source = {skills_db_source}\n")
        if git_state["status_lines"]:
            handle.write("# git_status_short =\n")
            for line in git_state["status_lines"]:
                handle.write(f"#   {line}\n")
        else:
            handle.write("# git_status_short = <clean>\n")
        if git_errors and args.allow_debug_git_state:
            handle.write("# git_policy_override = allow-debug-git-state\n")
        handle.write("\n")
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
        "dataset_kind": dataset_kind,
        "skills_db": skills_db_path,
        "skills_db_source": skills_db_source,
        "git_branch": git_state["branch"],
        "git_commit": git_state["commit"],
        "git_worktree_dirty": git_state["worktree_dirty"],
        "git_status_short": git_state["status_lines"],
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
