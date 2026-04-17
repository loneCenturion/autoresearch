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
