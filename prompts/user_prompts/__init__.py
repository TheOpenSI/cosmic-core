### Prompt registry ###
# Re-exported so `get_instance(...)` can still resolve a prompt class by name.
from prompts.user_prompts.base import UserPromptBase
from prompts.user_prompts.services.chess import (
    CoTGeneration,
    FENNextMoveAnalyse,
    FENNextMoveAnalyseMistralFinetuned,
    FENNextMovePredict
)
from prompts.user_prompts.services.general import GeneralUserPrompt