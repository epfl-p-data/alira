# Changelog

## 0.1.3

- Stop capping evaluations at 16 tokens. The answer is a small strict JSON object, and models that reject `max_tokens` (e.g. OpenAI reasoning models such as `gpt-6-luna`) failed on every evaluation.

## 0.1.2

- Omit unset optional chat parameters instead of sending them as `null`. Strict providers such as the OpenAI API rejected synthetic text generation with `Invalid type for 'max_tokens'`.

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
