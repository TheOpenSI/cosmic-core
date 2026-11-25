from ..base import TokenizerBase


class GPT(TokenizerBase):
    def __init__(
        self,
        llm_name: str = "",
        **kwargs
    ):
        """
        For OpenAI GPT.
        GPT does not require tokenizer, just keep the interface.

        Args:
            llm_name (str, optional): LLM name. Defaults to "".
        """
        super().__init__(llm_name, **kwargs)