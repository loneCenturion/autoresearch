# PRD — Bridge OMX root AGENTS.md to project-specific rules in program.md

## Metadata
- Source spec: `.omx/specs/deep-interview-agents-md-conflict.md`
- Planned on: 2026-04-17
- Scope risk: narrow
- Reversibility: clean

## Problem
OMX initialization created a root `AGENTS.md` that is now the active repository instruction entrypoint. The repository's documented convention in `README.md` still says `program.md` is the real instruction source and `AGENTS.md` should only be an alias/symlink. The active behavior and documented source of truth are currently misaligned.

## Goal
Preserve the OMX-generated root `AGENTS.md` as the short-term entrypoint while making `program.md` explicitly authoritative for project-specific repo rules.

## Non-goals
- Do not switch execution into Ralph/Team for experiment work yet.
- Do not force `program.md` to immediately become the root file again.
- Do not redesign the Safe-OS experiment workflow.
- Do not merge the full OMX orchestration contract into `program.md`.

## Users / Stakeholders
- Primary: user maintaining this existing repository
- Secondary: agent/Codex sessions that need a reliable root entrypoint plus clear project-local instructions

## Requirements
1. Root `AGENTS.md` remains present so OMX continues to work.
2. Root `AGENTS.md` must explicitly point readers/agents to `program.md` for project-specific repository rules.
3. The bridge language must be unambiguous about precedence for repo-local experiment rules.
4. The repository docs should no longer imply a pure symlink setup if that is no longer the chosen short-term architecture.
5. The plan should leave a clean future path toward deeper OMX workflow adoption.

## Proposed solution
### Option chosen
Keep the OMX-generated root `AGENTS.md`, but add a small, prominent repo-local bridge section near the top that says:
- OMX orchestration rules remain active at the root.
- For this repository's project-specific experiment and workflow rules, treat `program.md` as the source of truth.
- If project-local instructions conflict with generic repo commentary elsewhere, the explicit bridge to `program.md` is the intended source for this repo's local operating rules.

### Supporting changes
1. Update `README.md` so it no longer says `AGENTS.md` should only be a symlink in the current short-term setup.
2. Remove or rename `AGENTS copy.md` so it stops suggesting a live authority path.
3. Add a brief note to `program.md` clarifying that it is consumed via the root AGENTS bridge during the OMX transition period.

## Alternatives considered
### A. Restore `AGENTS.md -> program.md` immediately
- Pros: matches old repo convention exactly
- Cons: discards the OMX-generated root entrypoint the user chose to preserve for continuity
- Rejected because the user explicitly prioritized OMX continuity first

### B. Keep current OMX `AGENTS.md` unchanged and rely on human memory
- Pros: zero edits
- Cons: active behavior remains misaligned with documented repo source of truth; future agents may miss project-local rules
- Rejected because it keeps the ambiguity unresolved

### C. Merge all project rules into root `AGENTS.md`
- Pros: one file
- Cons: forces premature migration and duplicates repo-local guidance
- Rejected because the user does not want a full rule-system replacement now

## Decision drivers
1. Preserve OMX continuity
2. Make project-local rules explicit and machine-discoverable
3. Keep the change small and reversible
4. Avoid premature workflow migration

## Acceptance criteria
1. Root `AGENTS.md` still contains the OMX-generated contract.
2. Root `AGENTS.md` also contains a clear bridge to `program.md` for project-specific rules.
3. `README.md` accurately describes the new bridge model.
4. `AGENTS copy.md` no longer creates false authority cues.
5. No experiment logic files are changed as part of this bridge-only work.

## Implementation slices
### Slice 1 — establish bridge
- Edit `AGENTS.md`
- Add a concise repo-local bridge section near the top

### Slice 2 — align docs
- Edit `README.md`
- Update the “Agent 文件约定” section to describe the bridge model instead of a pure symlink recommendation

### Slice 3 — remove misleading alias
- Delete `AGENTS copy.md` or convert it into a clearly non-authoritative note outside agent-discovery conventions

### Slice 4 — optional clarification
- If needed, add one short note in `program.md` that it remains the source of repo-specific instructions during OMX transition

## Verification plan
- Confirm `AGENTS.md` still contains the OMX marker after edits
- Confirm `AGENTS.md` contains an explicit `program.md` bridge reference
- Confirm `README.md` no longer claims the active short-term setup is only a symlink
- Confirm `AGENTS copy.md` is removed or obviously non-authoritative
- Confirm only documentation/instruction files changed

## ADR
### Decision
Adopt a bridge model: keep OMX root `AGENTS.md` as entrypoint, and explicitly direct repo-specific rules to `program.md`.

### Drivers
- User chose OMX continuity over immediate restoration of `program.md` as root entrypoint
- Existing docs still treat `program.md` as the true local rules file
- Current setup leaves active behavior and documented intent misaligned

### Alternatives considered
- Immediate symlink restoration
- No-op / rely on memory
- Full migration into root `AGENTS.md`

### Why chosen
It resolves the current ambiguity with the smallest reversible change while preserving future OMX adoption.

### Consequences
- Root `AGENTS.md` becomes a hybrid: OMX entrypoint plus explicit repo-local bridge
- `README.md` must be updated to match reality
- Future cleanup may still choose to re-centralize later, but that is deferred intentionally

### Follow-ups
- After bridge edits, optionally run `$ralph` for the bounded doc alignment implementation
- Later, if OMX workflow adoption deepens, revisit whether `program.md` should remain separate or be absorbed into a different structure
