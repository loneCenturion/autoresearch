# `.omx/` tracking policy

This repository tracks only **non-sensitive planning artifacts** under `.omx/`.

Tracked:
- `specs/deep-interview-agents-md-conflict.md`
- `plans/prd-agents-md-bridge.md`
- `plans/test-spec-agents-md-bridge.md`
- this `README.md`

Ignored on purpose:
- `state/`
- `logs/`
- `context/`
- `interviews/`
- `metrics.json`
- `hud-config.json`
- `tmux-hook.json`
- `setup-scope.json`
- any session/runtime-specific OMX files

Why:
- runtime files may include session IDs, local paths, PIDs, pane IDs, or operator history
- the tracked artifacts above are the minimum durable record of why the bridge model was chosen and how it should be validated
