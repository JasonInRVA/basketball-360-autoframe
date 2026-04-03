# Insta360 integration viability assessment (April 1, 2026)

## Executive summary

The current pivot is **viable** if we target **Insta360 Studio desktop** and keep the architecture as:

1. Generate `.insproj` project data from our model.
2. Pair project + original source media.
3. Let Studio perform final render/export.

The pivot is **not guaranteed viable** if we require:

- direct automation of Insta360 Studio itself (no official Studio automation API discovered), or
- seamless parity with mobile-app transfer flows (`.insdata`) instead of Studio project-page projects (`.insproj`).

The good news is that your newly approved SDK access gives us a strong fallback path: if sidecar compatibility is fragile, the official Media SDKs can export stitched MP4/JPG outputs directly without depending on Studio project-sidecar import behavior.

## What official docs indicate

### 1) `.insproj` is an official Studio project artifact (Project page editing)

Insta360 Studio's own manual states:

- `.insprj` stores editing data generated while editing media on the **Media page**.
- `.insproj` corresponds to **Project page** projects.
- project files are tied to exact media and naming assumptions.

**Our tool targets `.insproj` (Project page).** The real test project examined during development is a `.insproj` directory with a JSON-based structure. This is the format our parser reads and writes. The `.insprj` Media page format is a separate, older artifact we do not target.

This supports the core assumption that a sidecar file is the data carrier for framing/keyframes in Studio.

### 2) App → Studio transfer exists, but via `.insdata` and scoped support

Insta360's forum announcement for "App to Studio edits" describes a workflow where the app exports edit data to camera storage as `.insdata`, then Studio imports it.

The same announcement explicitly limits support (e.g., specific edit types and version requirements). This implies transfer paths are productized but constrained, and formats may differ by workflow (mobile transfer format vs Studio-native project format).

### 3) Official SDK surface focuses on camera control + media stitch/export, not Studio project automation

Current official SDK sources and docs emphasize:

- Camera SDK: connection, capture, download, control.
- Media SDK/Desktop MediaSDK-Cpp: stitching and exporting media (`insv` input → `mp4` output, etc.).

I did **not** find official documentation exposing a Studio-desktop API for importing arbitrary project sidecars and triggering export via API.

That does not prove impossible, but it does lower confidence in a Studio-automation-by-API plan.

## Practical interpretation for our roadmap

### What looks strong

- **Strongly plausible:** model generates keyframe trajectory and injects it into an existing `.insproj` project, then user opens in Studio and exports.
- **Strongly plausible:** if needed, bypass Studio and render with official Media SDK (more engineering effort, but official route).

### What looks risky

- Assuming `.insproj` schema is stable, complete, and easy to synthesize without strict coupling to Studio version/media metadata.
- Assuming mobile (`.insdata`) and Studio Project page (`.insproj`) are interchangeable.
- Assuming there is an official Studio automation API endpoint for batch import/render.

## Recommendation

Use a **two-track validation plan** immediately:

### Track A (keep pivot): prove sidecar compatibility empirically

Create a compatibility matrix for:

- camera models (start with X4),
- Studio versions (at least current + one prior),
- media types/resolutions you care about,
- sidecar source (`Studio-generated`, `ours-generated`, `ours-mutated-from-Studio`).

Pass criteria:

1. Studio loads the modified `.insproj` without error.
2. Keyframes/FOV transitions match expected motion profile.
3. Export succeeds with correct duration/audio/stabilization.
4. Re-open round-trip preserves edits.

If this passes consistently, the pivot is validated.

### Track B (risk hedge): parallel Media SDK spike

In parallel, build a minimal desktop MediaSDK export spike that:

- ingests a sample `insv`,
- sets output and stitch params,
- exports deterministic MP4.

This derisks the business if sidecar loading is brittle or breaks with Studio updates.

## Go / no-go criteria

Proceed with sidecar-first as primary path if:

- >=95% success across matrix cases,
- no major drift after Studio patch update,
- no manual repair required for generated projects.

Pivot back to own renderer (or SDK renderer wrapper) if:

- repeated import failures,
- non-deterministic behavior across minor Studio versions,
- unsupported keyframe attributes needed for quality output.

## Bottom line

You are **not** on a dead-end path.

- There is strong evidence that `.insproj` is the correct Studio-side project artifact for Project page edits, which is what our tool targets.
- There is also strong evidence that official SDKs can cover rendering/export if we need to move away from Studio-sidecar dependency.

So the right move is not an immediate pivot-back; it is a short, decisive validation sprint with a documented fallback to official Media SDK rendering.
