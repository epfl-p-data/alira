# ALIRA — Agent Guide

## What it does
ALIRA bootstraps a binary classifier with **HyDE (Hypothetical Document Embeddings)** — LLM-generated synthetic texts — then runs an active-learning loop: iteratively evaluating real corpus items via an LLM, training a Logistic Regression classifier, and selecting the next batch based on prediction uncertainty. Early stopping triggers when positive-zone prediction drift (RMSE) stabilises.

## Project Structure
- Single Python package `alira` using `src/` layout (`src/alira/`).
- Build system: Hatchling (see `pyproject.toml`).
- Requires Python >= 3.11.
- **No test suite, lint, or type-check configuration exists.**

## Environment & Secrets
- LLM credentials are read from environment variables via `alira.config.CONFIG` (a thin wrapper around `os.environ.get`).
- Required env vars: `ALIRA_LLM_BASE_URL`, `ALIRA_LLM_API_KEY`, `ALIRA_LLM_EMBEDDING_MODEL`, `ALIRA_LLM_BASE_MODEL`.
- **Examples use `python-dotenv`, but the core package does not.** Any new entrypoint script should call `load_dotenv()` itself if it relies on a `.env` file.
- `.env` is gitignored and private. Don't read it.

## Running Code
- Install locally: `pip install -e .` (core deps: numpy, openai, pandas, pydantic, scikit-learn).
- To run the example scripts: `pip install -e ".[demo]"` (adds `python-dotenv` and `epfl-data-index`).
- Example script: `python examples/demo.py`
  - Expects `data/movies.csv` relative to the working directory.
  - Writes outputs to `results/` (gitignored).
- The public API exports `ActiveLearner` from `alira`.

