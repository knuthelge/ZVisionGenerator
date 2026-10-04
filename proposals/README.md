# Proposals

Planned and possible future work, written up before it is built. Each proposal is one Markdown file describing the problem, the evidence, the proposed change and the alternatives considered, so work can be picked up later without re-deriving the context.

These files are not part of the published docs site.

## Conventions

- One file per proposal, named with a short kebab-case slug (e.g. `enhance-before-model-load.md`).
- Start each file with a **Status** line: `Proposed`, `Accepted`, `In progress`, `Done` or `Rejected`, plus the date it last changed.
- When a proposal ships, set it to `Done` and link the release or commit; keep the file as a design record.
- Add every proposal to the index below.

## Index

| Proposal | Status | Summary |
|---|---|---|
| [Enhance prompts before the model loads](enhance-before-model-load.md) | Done (v0.13.0b8) | Fix out-of-memory failures in auto-enhanced batches by rewriting every prompt before the generation model loads |
| [Controls for Web UI video jobs](video-job-controls.md) | Proposed | Let Web UI video jobs be stopped (and possibly skipped or paused), including while prompts are enhanced |
