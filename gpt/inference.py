
import logging
import openai
import json

import pandas as pd

from gpt.config import OPENAI_KEY, GPT_PROMPT


openai.api_key = OPENAI_KEY


def infer_article_gpt_4(article_row: pd.Series, temperature=0.2, max_tokens=1000, frequency_penalty=0.0) -> (str, [str], int):
    """

    :param article_row: (pd.Series)
    :param temperature: (float)
    :param max_tokens: (int)
    :param frequency_penalty: (float)

    :return: (str, [str], int)
    """

    prompt = GPT_PROMPT.format(article_row["title"], article_row["author"], article_row["text"][:11000])

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty
        )

    except Exception as e:
        logging.warning("GPT-4 inference failed with '{}'".format(str(e)))
        return "failed", [], 0

    label = "failed"
    explanation = []

    try:
        response_structured = json.loads(response["choices"][0]["message"]["content"])
        explanation = [criteria.lower() for criteria in response_structured["explanation"]]
        label = response_structured["label"].lower()

    except Exception as e:
        print(response["choices"][0]["message"]["content"])
        logging.warning("GPT-4 output conversion failed with '{}'".format(str(e)))

    return label, explanation, response["usage"]["total_tokens"]


if __name__ == "__main__":
    pass
