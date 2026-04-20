# `.codex/` tracking policy

This repository keeps **only sanitized Codex/OMX integration notes** in git.

Tracked here:
- this `README.md`

Local-only and intentionally ignored:
- `config.toml`
- `hooks.json`
- `auth.json`
- `state_*.sqlite*`
- `history.jsonl`
- `log/`
- `shell_snapshots/`
- generated `agents/`, `prompts/`, and `skills/`
- any other runtime/session-specific files

Why:
- these files often contain local machine paths, proxy settings, auth/session data, hook commands, or generated runtime state
- they are useful for the current machine, but unsafe or noisy for repository history

Repository convention:
- the root `AGENTS.md` remains a **local ignored** OMX/Codex entrypoint
- `program.md` remains the source of truth for repo-specific experiment/workflow rules
- if a new machine initializes OMX, recreate the local `AGENTS.md` bridge described in `README.md`

## Local self-check

For a healthy local Codex/OMX setup in this repo, check:

- required local config:
  - `.codex/config.toml`
  - `.codex/hooks.json`
- expected local runtime files (auto-generated, do not commit):
  - `.codex/history.jsonl`
  - `.codex/logs_2.sqlite*`
  - `.codex/state_5.sqlite*`
- auth sources:
  - repo-local `.codex/auth.json` is optional
  - global `~/.codex/auth.json` may be used instead
  - environment credentials such as `OPENAI_API_KEY` may also satisfy auth

If Codex is already running, losing a local config/auth file may not break the **current** session immediately because the process has already started with loaded config and credentials. It can still break the **next** new session, so restore missing local config files before restarting.

## File classification

### Keep locally
- `config.toml`
- `hooks.json`

### Runtime-only / re-creatable
- `history.jsonl`
- `logs_2.sqlite*`
- `state_5.sqlite*`

### Sensitive or machine-specific
- `auth.json`
- local provider URLs / proxy settings / hook commands in `config.toml`

### Safe to clean up when unused
- stale nested runtime directories such as `.codex/.codex/`
- abandoned sqlite/wal/shm leftovers that are not opened by a current Codex process
