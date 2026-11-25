from ..chatgpt.chatgpt import GPT


class Ollama(GPT):
    def __init__(
        self,
        llm_name: str = "",
        **kwargs
    ):
        """
        For Ollama model.
        Ollama model does not require tokenizer, just keep the interface.

        Args:
            llm_name (str, optional): LLM name. Defaults to "".
        """
        super().__init__(llm_name, **kwargs)