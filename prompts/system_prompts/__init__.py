### Prompt registry ###
# Re-exported so `get_instance(...)` can still resolve a prompt class by name.
from prompts.system_prompts.base import SystemPromptBase
from prompts.system_prompts.slms.gemma import Gemma7b, Gemma7bIt
from prompts.system_prompts.slms.gpt import GPT, Ollama
from prompts.system_prompts.slms.mistral import (
    Mistral7bv01,
    Mistral7bInstructv01,
    MistralFinetuned
)
from prompts.system_prompts.services.chess import (
    FENNextMoveAnalyse,
    FENNextMoveAnalyseMistralFinetuned
)