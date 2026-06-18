# ALIRA

[![PyPI version](https://img.shields.io/pypi/v/alira.svg)](https://pypi.org/project/alira/)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Active Learning Iterative Retrieval Agent**

ALIRA classifies a large text corpus according to a natural-language query when exhaustive LLM evaluation is impractical. It iteratively discovers relevant documents using active learning, LLM validation, and classifier refinement.

## Overview

Given a text corpus and a natural-language query, ALIRA bootstraps a binary classifier using LLM-generated synthetic examples (HyDE), then iteratively:

1. **Evaluates** candidate documents via an LLM (up to 5 concurrent calls)
2. **Trains** a Logistic Regression classifier on the accumulated labels
3. **Predicts** relevance scores for the full corpus
4. **Selects** the next batch of candidates via stratified sampling across confidence zones
5. **Stops early** when the positive-zone prediction drift (RMSE) falls below a threshold

The result is a ranked list of documents predicted to match the query, requiring far fewer LLM calls than exhaustive evaluation.

## Quick Start

### Requirements

- Python >= 3.11
- LLM provider supporting the OpenAI API format (for both chat and embedding endpoints)

### Installation

Install from PyPI:

```bash
pip install alira
```

Or clone the repository and install the development version locally:

```bash
git clone git@github.com:epfl-p-data/alira.git
cd alira
pip install -e .
```

### Configuration

Set the following environment variables (or use a `.env` file loaded by your entrypoint script):

| Variable | Description |
|----------|-------------|
| `ALIRA_LLM_BASE_URL` | Base URL of the LLM API |
| `ALIRA_LLM_API_KEY` | API key for authentication |
| `ALIRA_LLM_EMBEDDING_MODEL` | Model name for embedding requests |
| `ALIRA_LLM_BASE_MODEL` | Model name for chat/evaluation requests |

### Example

```python
import pandas as pd
from alira import ActiveLearner

# Load corpus
df = pd.read_csv("data/movies.csv")

# Fit active learner
learner = ActiveLearner(corpus=df["text"])
learner.fit(query="sports")

# Get ranked results
df["score"] = learner.predict_proba()
results = df[df["score"] >= 0.5].sort_values("score", ascending=False)
```

See [`examples/demo.py`](examples/demo.py) for a complete runnable script with logging and result persistence.

## API

### `ActiveLearner`

Main entrypoint exported by the `alira` package.

```python
ActiveLearner(
    corpus: list[str] | pd.Series | np.ndarray,
    embeddings: np.ndarray | pd.Series | None = None,
    n_synthetic: int = 10,
    min_iterations: int = 3,
    max_iterations: int = 20,
    n_eval_per_iteration: int = 30,
    c_value: float = 1.0,
    positive_zone_rmse_threshold: float = 0.01,
    cluster_candidates: bool = False,
    generation_prompt: str | None = None,
    evaluation_prompt: str | None = None,
)
```

| Parameter | Description                                                     |
|-----------|-----------------------------------------------------------------|
| `corpus` | Collection of texts to search                                   |
| `embeddings` | Optional pre-computed embeddings aligned 1-to-1 with `corpus`   |
| `n_synthetic` | Number of synthetic texts to generate for bootstrapping (HyDE)  |
| `min_iterations` | Minimum iterations before early stopping is evaluated           |
| `max_iterations` | Maximum active learning iterations                              |
| `n_eval_per_iteration` | Number of texts evaluated per iteration                         |
| `c_value` | Inverse regularization strength for LogisticRegression          |
| `positive_zone_rmse_threshold` | Early-stopping threshold for prediction drift in the positive zone |
| `cluster_candidates` | Whether to cluster candidates within each stratum for diversity |
| `generation_prompt` | Custom prompt for synthetic text generation                     |
| `evaluation_prompt` | Custom prompt for LLM evaluation                                |

#### Methods

- `fit(query: str) -> Self` — Run the active-learning loop and train the classifier.
- `predict_proba(corpus=None, embeddings=None) -> pd.Series` — Return predicted probabilities of relevance.
- `predict(corpus=None, embeddings=None) -> pd.Series` — Return binary predictions.

## Project Structure

```
.
├── src/
│   └── alira/
│       ├── __init__.py           # Package entrypoint, exports ActiveLearner
│       ├── active_learner.py     # Core ActiveLearner implementation
│       ├── classifiers.py        # LogisticRegressionClassifier
│       ├── evaluation.py         # LLM-based binary evaluation (async, max 5 concurrent)
│       ├── llms.py               # OpenAI API client for chat and embeddings
│       ├── synthetic.py          # Synthetic text generation via LLM (HyDE)
│       └── config.py             # Environment-based configuration
├── examples/
│   ├── demo.py                   # Example script with logging and CSV output
│   ├── lab_explorer.py           # Example using external data source
│   ├── compare.py                # Utility to compare result sets
│   ├── aists.py                  # Batch runner for AISTS themes
│   ├── embeddings.py             # Generate and cache embeddings
│   └── utils.py                  # Shared example utilities
├── pyproject.toml
└── README.md
```

## Dependencies

Core dependencies (see `pyproject.toml`):

- numpy
- openai
- pandas
- pydantic
- scikit-learn

Optional dependencies for the example scripts (install with `pip install -e ".[demo]"`):

- python-dotenv
- epfl-data-index
