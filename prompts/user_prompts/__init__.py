from .models.base import UserPromptBase
from .services.chess.chess import (
    CoTGeneration,
    FENNextMoveAnalyse,
    FENNextMoveAnalyseMistralFinetuned,
    FENNextMovePredict
)
from .services.general_qa_answering.general import GeneralUserPrompt