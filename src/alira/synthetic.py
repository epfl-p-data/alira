import logging

from pydantic import BaseModel, Field, ValidationError
from openai import APITimeoutError

from alira.llms import send_llm_request

logger = logging.getLogger(__name__)


class TextList(BaseModel):
    texts: list[str] = Field(..., description="The list of generated texts of the given topic following the specified format.")


def generate_synthetic_texts(
    query: str,
    n: int,
    examples: list[str],
    prompt: str | None = None,
    max_retries: int = 3,
) -> list[str]:
    """Generate n synthetic texts about query, using examples for format reference."""

    if prompt is None:
        examples_block = "\n\n---\n\n".join(examples)
        prompt = f"""
You are an expert generating texts related to a given topic. 
Below are some randomly chosen example texts from a dataset.
Your task is to produce a list of exactly {n} texts with the same format as the example texts, but so that all the texts you produce are related to the topic "{query}".
Produce texts that could be extracted from the same dataset as the examples but somehow having filtered semantically by the given topic.
Do not include the delimiters in your generated texts.

Here are the examples:

---

{examples_block}

---
"""

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        try:
            response = send_llm_request(messages, response_format=TextList)
            if len(response.texts) == n:
                return response.texts
        except APITimeoutError as e:
            logger.warning("Failed to generate texts before timeout. Error: %s", e)
        except ValidationError as e:
            logger.warning("Failed to generate exactly %s texts. Error: %s", n, e)

    raise ValueError(f"Failed to get exactly {n} texts after {max_retries} attempts")
