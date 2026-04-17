# Test Spec — AGENTS/program bridge

## Source
- PRD: generated in same planning run for `agents-md-bridge`
- Deep interview spec: `.omx/specs/deep-interview-agents-md-conflict.md`

## Test objective
Prove that the repository's active root instruction entrypoint remains compatible with OMX while project-specific rules become explicit and discoverable through `program.md`.

## Test matrix

### T1 — Root AGENTS entrypoint preserved
- Check: `AGENTS.md` still exists at repo root
- Check: it still contains `<!-- omx:generated:agents-md -->`
- Pass condition: OMX root contract remains intact

### T2 — Explicit bridge to program.md exists
- Check: `AGENTS.md` contains a visible reference to `program.md`
- Check: wording states that `program.md` is the source for project-specific repo rules
- Pass condition: agent/operator can discover local rules from the active root entrypoint

### T3 — README matches short-term architecture
- Check: `README.md` no longer says the active recommended setup is only `AGENTS.md` as a symlink
- Check: README describes the chosen bridge model accurately
- Pass condition: repo docs and active behavior are aligned

### T4 — No misleading duplicate authority file remains
- Check: `AGENTS copy.md` is removed or replaced with a non-authoritative artifact outside active agent discovery naming
- Pass condition: no misleading near-authoritative alias remains at repo root

### T5 — Scope remains narrow
- Check: changed files are limited to `AGENTS.md`, `README.md`, optional `program.md`, and cleanup of `AGENTS copy.md`
- Check: no experiment code files such as `train.py` or `prepare.py` changed
- Pass condition: bridge-only work stayed within bounds

## Verification commands
```bash
ls -l AGENTS.md program.md
[ -e 'AGENTS copy.md' ] && ls -l 'AGENTS copy.md' || true
grep -n '<!-- omx:generated:agents-md -->' AGENTS.md
grep -n 'program.md' AGENTS.md
sed -n '34,50p' README.md
git diff -- AGENTS.md README.md program.md 'AGENTS copy.md'
```

## Risks to watch
- Accidentally deleting or corrupting the OMX-generated root guidance
- Creating contradictory precedence language between `AGENTS.md` and `program.md`
- Leaving README with stale symlink-only wording

## Exit criteria
All T1-T5 pass and the resulting bridge language is clear enough that a future agent session will not need to infer where project-local instructions live.
