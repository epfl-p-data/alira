import logging

from openai import OpenAI, BadRequestError

from alira.config import CONFIG

logger = logging.getLogger(__name__)


def get_client(timeout: int | None = None) -> OpenAI:
    """Return an OpenAI client configured from the environment."""
    return OpenAI(
        base_url=CONFIG['ALIRA_LLM_BASE_URL'],
        api_key=CONFIG['ALIRA_LLM_API_KEY'],
        timeout=timeout,
    )


def send_embedding_request(texts: list[str]) -> list[list[float]]:
    logger.info("Sending embedding request for %s texts", len(texts))
    client = get_client()
    embeddings: list[list[float]] = []
    batch_size = 500

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(
            "Processing embedding batch %s/%s (%s items)",
            i // batch_size + 1,
            (len(texts) - 1) // batch_size + 1,
            len(batch),
        )
        r = client.embeddings.create(model=CONFIG['ALIRA_LLM_EMBEDDING_MODEL'], input=batch)
        embeddings += [item.embedding for item in r.data]

    logger.info("Received %s embeddings", len(embeddings))
    return embeddings


def send_llm_request_schema(messages, response_format=None, max_tokens=None, timeout=30):
    response_format_schema = None
    if response_format:
        response_format_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": response_format.model_json_schema(),
                "strict": True,
            },
        }

    client = get_client(timeout=timeout)
    response = client.chat.completions.create(
        model=CONFIG['ALIRA_LLM_BASE_MODEL'],
        messages=messages,
        response_format=response_format_schema,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content.strip()

    if response_format:
        return response_format.model_validate_json(content)

    return content


def send_llm_request_parsed(messages, response_format=None, max_tokens=None, timeout=30):
    """Send a chat request and return a parsed Pydantic model directly (or raw text).

    Uses the OpenAI SDK's native structured-output support when a Pydantic
    model is provided, falling back to plain ``create`` otherwise.
    """
    client = get_client(timeout=timeout)

    if response_format is None:
        response = client.chat.completions.create(
            model=CONFIG['ALIRA_LLM_BASE_MODEL'],
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    response = client.chat.completions.parse(
        model=CONFIG['ALIRA_LLM_BASE_MODEL'],
        messages=messages,
        response_format=response_format,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.parsed


def send_llm_request(messages, response_format=None, max_tokens=None, timeout=30):
    """Send a chat request, trying SDK-native parsing and falling back to manual schema."""
    try:
        return send_llm_request_parsed(messages, response_format, max_tokens, timeout)
    except BadRequestError:
        return send_llm_request_schema(messages, response_format, max_tokens, timeout)
