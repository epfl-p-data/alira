# Changelog

## 0.1.1

- Disable the thinking mode of reasoning models on chat requests. Evaluations are capped at 16 tokens, so models that think by default (e.g. Qwen3) failed with `LengthFinishReasonError`.
- New optional `ALIRA_LLM_DISABLE_THINKING` environment variable (default `true`); set it to `false` for providers that reject `chat_template_kwargs`.

## 0.1.0 (2026-06-18)

- Initial release.
- `ActiveLearner` class for binary text classification via active learning.
- HyDE (Hypothetical Document Embeddings) bootstrapping with synthetic text generation.
- LLM-based evaluation with concurrent requests.
- Logistic Regression classifier with balanced class weights.
- OpenAI-compatible chat and embedding client helpers.
- Example scripts: `demo.py`, `lab_explorer.py`, `compare.py`, `aists.py`, and `embeddings.py`.
