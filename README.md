# 826_Code
biostat 826 code

This repository now includes reusable utilities and tests for Assignment 1 (mortality prediction in MIMIC-IV).

## Environment

Use your existing micromamba environment:

```bash
micromamba activate medai
```

All testing and coding for this repo can be done directly in `medai`.

## Layout

- `Assignment1/`: assignment-specific specs and the final notebook deliverable.
- `utils/`: reusable Python modules for data prep, model training, and evaluation.
- `tests/`: reusable pytest suite to validate utilities before each push.
- `docs/`: notes about MIMIC-IV file structure and schema.

## Typical Workflow

```bash
pytest tests/ -v
pre-commit run --all-files
```

Run `pytest -m slow` for full integration tests against real MIMIC files.
