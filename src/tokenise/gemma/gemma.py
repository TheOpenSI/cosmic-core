from ..mistral.mistral import Mistral7bv01


class Gemma7b(Mistral7bv01):
    def __init__(
        self,
        llm_name: str = "gemma-7b",
        **kwargs
    ):
        """
        For Gemma 7B.

        Args:
            llm_name (str, optional): LLM name. Defaults to "mistral-gemma-7b".
        """
        super().__init__(llm_name, **kwargs)


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
            system_prompt (str): encoded system prompt, which is torch.tensor.
        """
        return self.tokenizer(
            system_prompt,
            return_tensors="pt",
            padding=True,
            **kwargs
        ).input_ids.to(self.device)


    def decode(
        self,
        response: str,
        **kwargs
    ):
        """
        Decode response according to the encoder.

        Args:
            response (str): raw response, torch.tensor, from LLM.

        Returns:
            response: decoded response.
        """
        return self.tokenizer.decode(
            response,
            skip_special_tokens=True,
            **kwargs
        )



class Gemma7bIt(Mistral7bv01):
    def __init__(
        self,
        llm_name: str="gemma-7b-it",
        **kwargs
    ):
        """
        For Gemma 7B Instruction.

        Args:
            llm_name (str, optional): LLM name. Defaults to "gemma-7b-instruct".
        """
        super().__init__(llm_name, **kwargs)


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
            system_prompt (str): encoded system prompt, which is torch.tensor.
        """
        return self.tokenizer.apply_chat_template(
            system_prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            **kwargs
        ).to(self.device)


    def decode(
        self,
        response: str,
        **kwargs
    ):
        """
        Decode response according to the encoder.

        Args:
            response (str): raw response, torch.tensor, from LLM.

        Returns:
            response: decoded response.
        """
        return self.tokenizer.decode(
            response,
            skip_special_tokens=True,
            **kwargs
        )