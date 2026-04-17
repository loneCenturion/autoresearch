# Deep Interview Spec — agents-md-conflict

## Metadata
- Profile: standard
- Rounds: 4
- Final ambiguity: 0.12
- Threshold: 0.20
- Context type: brownfield
- Context snapshot: `.omx/context/agents-md-conflict-20260417T031301Z.md`

## Clarity Breakdown
| Dimension | Score |
|---|---:|
| Intent | 0.94 |
| Outcome | 0.88 |
| Scope | 0.82 |
| Constraints | 0.86 |
| Success | 0.70 |
| Context | 0.94 |

## Intent
The user wants to integrate oh-my-codex into an existing project without losing control of existing project-specific instructions. Immediate need: make agent/Codex behave correctly in this repo. Medium-term need: gradually adopt OMX workflows like `plan`, `ralph`, and `team`.

## Desired Outcome
Keep OMX usable in this repository while making the project-specific rule source explicit and reliable, using a low-risk bridge rather than an abrupt full migration.

## In Scope
- Clarify the authoritative relationship among `AGENTS.md`, `program.md`, and the OMX-generated files.
- Define the preferred short-term bridge strategy for this repository.
- Preserve future compatibility with gradual OMX workflow adoption.

## Out-of-Scope / Non-goals
- Do not redesign the experiment workflow yet.
- Do not immediately switch execution to `ralph` / `team` just because their state was seeded by keyword detection.
- Do not require `program.md` to immediately become the root entrypoint again.
- Do not merge the entire OMX-generated root `AGENTS.md` content into `program.md` as a forced migration step.

## Decision Boundaries
- Short-term priority: preserve OMX integration continuity.
- Root `AGENTS.md` may remain the OMX entrypoint for now.
- Project-specific repository rules should still be treated as coming from `program.md`.
- Preferred next change should be a bridge/reference strategy, not a full rule-system replacement.

## Constraints
- Brownfield repository with existing documented convention in `README.md` that says `program.md` is the source of truth and `AGENTS.md` should only be an alias/symlink.
- Current root `AGENTS.md` is an OMX-generated regular file, not a symlink.
- `AGENTS copy.md` is a symlink to `program.md` but is not authoritative because the filename is not `AGENTS.md`.
- Current branch is `master` and workspace is dirty after OMX initialization.

## Testable Acceptance Criteria
1. The chosen next-step plan preserves the current OMX root entrypoint instead of removing it immediately.
2. The chosen next-step plan makes `program.md` explicitly discoverable as the project-specific rules source.
3. The plan does not assume that seeded `ralph` state means execution should start immediately.
4. The plan supports a later gradual transition into OMX workflows (`plan`, `ralph`, `team`) without forcing that transition now.

## Assumptions Exposed + Resolutions
- Assumption: wanting `program.md` as source of truth means it must immediately return as the root authoritative file.
  - Resolution: false; user prefers preserving OMX continuity first.
- Assumption: mentioning `ralph` means the user wants immediate Ralph execution.
  - Resolution: false; current need is rules integration, not execution-mode launch.
- Assumption: `AGENTS copy.md` preserves agent behavior.
  - Resolution: false; renamed files do not govern automatic agent instruction loading.

## Pressure-pass Findings
- Revisiting the initial “keep `program.md` as source of truth” answer exposed a tradeoff.
- When forced to choose, the user preferred keeping the OMX root entry intact first and layering a bridge to `program.md` afterward.

## Brownfield Evidence vs Inference
### Verified evidence
- `README.md` says `program.md` is the real instruction file and `AGENTS.md` should only be an alias/symlink.
- Root `AGENTS.md` contains an OMX-generated marker and has no `program.md` reference.
- `AGENTS.md` is not a symlink.
- `AGENTS copy.md` is a symlink to `program.md`.
- `program.md` and `README.md` are tracked in git; current generated `AGENTS.md` is not present in `HEAD` as a tracked project file.
- Current branch is `master`.

### Inference
- The safest adoption path is to keep OMX’s root control surface while explicitly bridging project-specific repository rules from `program.md`.

## Technical Context Findings
- Relevant files: `AGENTS.md`, `program.md`, `README.md`, `AGENTS copy.md`
- Current mismatch: documented source of truth (`program.md`) differs from active root instruction entrypoint (`AGENTS.md`)
- Likely next planning topic: how to express a stable bridge from OMX root guidance to project-local rules without causing recursive or conflicting instruction ownership.

## Recommended Handoff
### Recommended: `$ralplan`
- Input artifact: `.omx/specs/deep-interview-agents-md-conflict.md`
- Why: requirements are now clear enough, but the bridge strategy should be planned carefully before editing instruction files.
- Expected output: a concrete repo-safe plan for bridging OMX root guidance with `program.md`, including verification steps and migration boundaries.

## Optional alternative handoffs
- `$autopilot`: only if you want direct planning/execution now.
- `$ralph`: only after planning artifacts exist and you want persistent execution.
- `$team`: only if this expands into a coordinated multi-file migration.
