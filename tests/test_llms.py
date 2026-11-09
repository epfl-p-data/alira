"""Unit tests for LLM client helpers."""

from unittest.mock import patch

import pytest

from alira.llms import get_client, get_extra_body, send_llm_request


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


@pytest.mark.parametrize("value", [None, "", "true", "1"])
def test_get_extra_body_disables_thinking_by_default(value):
    with patch("alira.llms.CONFIG") as mock_config:
        mock_config.__getitem__ = lambda self, key: value
        assert get_extra_body() == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("value", ["false", "False", "0", "no"])
def test_get_extra_body_can_be_turned_off(value):
    with patch("alira.llms.CONFIG") as mock_config:
        mock_config.__getitem__ = lambda self, key: value
        assert get_extra_body() is None


def test_send_llm_request_sends_extra_body():
    with (
        patch("alira.llms.get_client") as mock_get_client,
        patch("alira.llms.get_extra_body", return_value={"x": 1}),
    ):
        send_llm_request([{"role": "user", "content": "hi"}])
        create = mock_get_client.return_value.chat.completions.create
        assert create.call_args.kwargs["extra_body"] == {"x": 1}
