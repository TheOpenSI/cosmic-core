class TokenizerBase:
    def __init__(
        self,
        llm_name: str,
        device: str = "cuda"
    ):
        """Base class for tokenizer.

        Args:
            llm_name    (str):              LLM name, see src/maps.py, adapting tokenizer to different models.
            device      (str, optional):    use cuda or cpu for LLM. Defaults to "cuda".
        """
        self.llm_name = llm_name
        self.tokenizer = None
        self.device = device


    def encode(
        self,
        system_prompt: str,
        **kwargs
    ):
        """
        Encode prompt for LLM.

        Args:
            system_prompt (str): system prompt containing user prompt and context.

        Returns:
            system_prompt (str): encoded system prompt, can be string or torch.tensor.
        """
        return system_prompt


    def decode(
        self,
        response: str,
        **kwargs
    ):
        """
        Decode response according to the encoder.

        Args:
            response (str): raw response, string or torch.tensor, from LLM.

        Returns:
            response: decoded response.
        """
        return response