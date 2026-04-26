# Sprint S01: iteration_smoke

## Value

Check that one neutralization iteration can reduce explicit identifying markers without immediately destroying the forecast-relevant signal.

## Scope / Tasks

* Create `notebooks/04_neutering_iteration_smoke.ipynb`.
* Load `calm_2021` summary from SQLite by `run_date=2021-10-20`.
* Implement OpenRouter helper for chat completions.
* Implement J1 Q1-evidence prompt with strict JSON contract.
* Implement J1 Q3 baseline prompt with strict JSON contract.
* Implement N rewriter prompt using `identifiers` and `signals_to_preserve`.
* Run one logical iteration in the notebook.
* Save raw/candidate/Q1/Q3 outputs under `data/runs/e06_smoke/calm_2021/`.
* Leave manual review hooks in the notebook.

## DoD

* Notebook exists and is valid `.ipynb`.
* Notebook can be opened by Jupyter.
* Notebook cells are ordered for top-to-bottom execution.
* No production code is changed.
* No DB migrations are added.
* Q1-evidence prompt forbids true-period leakage.
* Q3 prompt uses the closed category list.
* N prompt contains the preservation invariants.
* Artifact paths are deterministic.
* `value_dod.md` exists.

## Non-goals

* Full iterative cycle.
* J2 probabilistic sampling.
* Q2 personas.
* Semantic anchor test.
* DB schema for neutering snapshots.
* CLI command `neuter`.
* Integration into `amnesiac/neuter`.

## Risks / Notes

* Model IDs may require adjustment if OpenRouter availability changes.
* If JSON format is unstable, the next task is prompt hardening, not production integration.
* If the first candidate becomes too generic, rewrite the N prompt before implementing the loop.
