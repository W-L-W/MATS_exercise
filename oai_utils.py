import json
from typing import Dict, List
from openai import OpenAI

OAI_MSGS = List[Dict[str, str]]

BOOLEAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "bool_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"response": {"type": "boolean"}},
            "required": ["response"],
            "additionalProperties": False,
        },
    },
}

NUM_MCQ_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "mcq_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "choice": {
                    "type": "string",
                    "enum": ["even", "odd", "div3", "not_div3", "prime", "not_prime"],
                }
            },
            "required": ["choice"],
            "additionalProperties": False,
        },
    },
}


with open("scr.txt", "r", encoding="utf8") as f:
    key = f.read()
    client = OpenAI(api_key=key)


def construct_messages(system_content: str, user_content: str):
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages


def call_api(
    msgs: OAI_MSGS,
    seed: int,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1,
    response_format=None,
    temperature: float = 0.0,
) -> dict:
    """Lightweight wrapper around completions api using my key"""
    return client.chat.completions.create(
        model=model,
        messages=msgs,
        max_tokens=max_tokens,
        seed=seed,
        response_format=response_format,
        temperature=temperature,
    )


def extract_response(response: dict) -> str:
    return response.choices[0].message.content


def extract_bool_response(response: dict) -> bool:
    options = ["True", "False", "true", "false"]
    str_response = extract_response(response)
    assert str_response in options
    return str_response in ["True", "true"]


def extract_bool_from_completion_json_format(response: dict) -> bool:
    str_response = extract_response(response)
    evaluated = json.loads(str_response)
    return evaluated["response"]


def extract_mcq_choice_from_completion_json_format(response: dict) -> str:
    str_response = extract_response(response)
    evaluated = json.loads(str_response)
    return evaluated["choice"]
