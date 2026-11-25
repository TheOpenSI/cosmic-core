### Core modules ###


### Type hint modules ###


### Internal modules ###
from ..openai.chatgpt import GPT


class Ollama(GPT):
    def __init__(self, **kwargs):
        """
        For Ollama model.
        """
        super().__init__(**kwargs)