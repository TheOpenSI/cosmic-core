from .models.base import SystemPromptBase
from .models.chatgpt.chatgpt import GPT
from .models.gemma.gemma import Gemma7b, Gemma7bIt
from .models.mistral.mistral import Mistral7bv01, Mistral7bInstructv01, MistralFinetuned
from .models.ollama.ollama import Ollama
from .services.chess.chess import FENNextMoveAnalyse, FENNextMoveAnalyseMistralFinetuned