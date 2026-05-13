# Repository Instructions

## Change Control

- Make only a small batch of changes at a time. Prefer a handful of related edits over broad repo-wide updates.
- After each batch, stop so the user can review before continuing with additional changes.
- Do not fix linter warnings/errors or make formatting or style-only changes unless the user explicitly requests that work.
- Keep changes narrowly targeted to the requested task. Avoid opportunistic cleanup, unrelated refactors, or broad code churn.
- When reviewing new code, look for opportunities to remove unnecessary generality before adding more structure.

## Engineering Style

- Prefer the smallest clear solution that satisfies the current task. Do not add configurability, extension points, abstractions, or alternate code paths unless the repository already needs them.
- Keep utilities task-specific and concrete. Generalize only after a real second use case appears.
- Documentation should explain behavior, assumptions, inputs/outputs, and non-obvious implementation choices. Avoid commentary that merely repeats the code.
- Add docstrings for reusable functions, persisted data models, public helpers, and metric/evaluation logic where the semantics are easy to misunderstand.
- Prefer examples over long prose when an example makes the behavior clearer.
- Add doctests where they are practical and stable, especially for pure functions, parsing/serialization helpers, metric formulas, and small deterministic transformations.
- Do not force doctests into I/O-heavy, plotting, CLI, filesystem, or long-running workflow functions where normal tests or smoke checks are more appropriate.

## Communication

- Summarize changes briefly after editing so the user can review them efficiently.
- If a request is ambiguous, ask for clarification instead of making assumptions and editing files.
