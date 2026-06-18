"""Unit tests for LLM client helpers."""

from unittest.mock import patch

from alira.llms import get_client


def test_get_client_uses_config():
    with (
        patch("alira.llms.CONFIG") as mock_config,
        patch("alira.llms.OpenAI") as MockOpenAI,
    ):
        mock_config.__getitem__ = lambda self, key: {
            "ALIRA_LLM_BASE_URL": "https://api.example.com",
            "ALIRA_LLM_API_KEY": "test-key",
        }[key]

        get_client()
        MockOpenAI.assert_called_once_with(
            base_url="https://api.example.com",
            api_key="test-key",
            timeout=None,
        )


def test_get_client_passes_timeout():
    with (
        patch("alira.llms.CONFIG") as mock_config,
        patch("alira.llms.OpenAI") as MockOpenAI,
    ):
        mock_config.__getitem__ = lambda self, key: {
            "ALIRA_LLM_BASE_URL": "https://api.example.com",
            "ALIRA_LLM_API_KEY": "test-key",
        }[key]

        get_client(timeout=60)
        MockOpenAI.assert_called_once_with(
            base_url="https://api.example.com",
            api_key="test-key",
            timeout=60,
        )
