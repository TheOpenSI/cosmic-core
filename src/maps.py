LLM_MODEL_DICT: dict[str, str] = {
    "mistral-7b-v0.1": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
    "gemma-7b": "google/gemma-7b",
    "gemma-7b-it": "google/gemma-7b-it",
    "mistral-7b-finetuned": "OpenSI/cognitive_AI_chess"
}


LLM_INSTANCE_DICT: dict[str, str] = {
    "mistral-7b-v0.1": "Mistral7bv01",
    "mistral-7b-instruct-v0.1": "Mistral7bInstructv01",
    "gemma-7b": "Gemma7b",
    "gemma-7b-it": "Gemma7bIt",
    "mistral-7b-finetuned": "MistralFinetuned",
    "gpt": "GPT",
    "ollama": "GPT"
}


# Registry id of the general question service, which has no persona of its own.
# NOTE: assigned by the database; keep in sync with the services table.
GENERAL_SERVICE_ID: str = "4"
