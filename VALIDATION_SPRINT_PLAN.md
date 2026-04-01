# Insta360 validation sprint plan

Date: 2026-04-01
Owner: basketball-360-autoframe

## Objective

Empirically validate whether the sidecar-first workflow is production-viable:

1. Generate `.insprj` from model predictions.
2. Load in Insta360 Studio desktop.
3. Export final video with acceptable reliability/quality.

In parallel, de-risk with a minimal Media SDK render fallback.

## Scope

- In-scope: Studio desktop sidecar compatibility, import/export round-trip behavior, and fallback feasibility via official Media SDK.
- Out-of-scope: mobile app editing parity and unsupported transfer workflows (`.insdata`) for this sprint.

## Track A — Sidecar compatibility matrix

## Automation helpers

Use the helper script to generate and summarize matrix logs:

```bash
# Generate a full matrix (example)
python scripts/validation_matrix.py generate   --tester jason   --camera-models X4   --studio-versions 5.4.4 5.3.2   --clips game01_q1 game01_q2 game02_q1

# Summarize progress and print go/no-go signal
python scripts/validation_matrix.py summarize
```

## Test matrix dimensions

- Camera model: start with X4 (expand later).
- Source clips: at least 5 representative game clips.
- Studio versions: current + one prior stable version.
- Sidecar source type:
  - `studio_generated`
  - `ours_generated`
  - `ours_mutated_from_studio`

## Per-case test procedure

1. Open source media in Studio.
2. Attach/import candidate sidecar.
3. Validate keyframe trajectory visually (yaw/pitch/fov motion).
4. Export using a fixed preset.
5. Re-open exported project and verify edit persistence.
6. Record result in CSV log.

## Pass/fail definitions

A single case is PASS only if all are true:

- sidecar imports without error/warning that blocks rendering,
- timeline behavior matches expected camera motion,
- export succeeds with expected duration/audio/stabilization,
- round-trip reopen does not lose keyframe intent.

## Track B — Media SDK fallback spike

Goal: demonstrate one deterministic SDK render path independent of Studio project-sidecar ingestion.

Minimum deliverable:

- input: one `.insv` reference clip,
- output: one stitched `.mp4`,
- parameters documented and repeatable.

## Decision gates

Stay sidecar-first if all are true:

- overall pass rate >= 95% on matrix cases,
- no blocker failures across both Studio versions,
- no manual XML repair required for normal operation.

Trigger pivot to SDK-backed renderer if any are true:

- repeated import failures on valid clips,
- behavior drifts significantly between minor Studio versions,
- required camera behavior cannot be represented reliably in sidecar output.

## Execution checklist

- [ ] Prepare reference clips and baseline Studio-generated sidecars.
- [ ] Generate `ours_generated` sidecars from current pipeline.
- [ ] Generate `ours_mutated_from_studio` sidecars for controlled diffing.
- [ ] Execute full matrix and fill CSV log.
- [ ] Run fallback SDK spike and archive command/config details.
- [ ] Summarize outcomes and produce go/no-go recommendation.

## Artifacts

- Matrix log: `artifacts/validation/compatibility-matrix.csv`
- This plan: `VALIDATION_SPRINT_PLAN.md`
- Prior strategic assessment: `INSTA360_VIABILITY_ASSESSMENT.md`
