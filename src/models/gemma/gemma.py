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
            llm_name (str, optional): LLM name in src/maps.py. Defaults to "gemma-7b".
        """
        super().__init__(llm_name=llm_name, **kwargs)

        # Set up LLM.
        self.llm = lambda system_prompt: self.model.generate(
            system_prompt,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=self.tokenizer.tokenizer.pad_token_id
        )[0]


    def truncate_response(
        self,
        response: str
    ):
        """
        Truncate response by specific system prompt keywords.

        Args:
            response (str): raw response from LLM.

        Returns:
            response (str): truncated response.
        """
        if not self.is_truncate_response:
            return response

        if self.use_example:
            response = response.split("model\n")[2].split("\n")[0]
        else:
            response = response.split("### ANSWER:\n")[-1]

        return response



class Gemma7bIt(Mistral7bv01):
    def __init__(
        self,
        llm_name: str="gemma-7b-it",
        **kwargs
    ):
        """
        For Gemma 7B Instruction.

        Args:
            llm_name (str, optional): LLM name in src/maps.py. Defaults to "gemma-7b-it".
        """
        super().__init__(llm_name=llm_name, **kwargs)

        # Set up LLM.
        self.llm = lambda system_prompt: self.model.generate(
            system_prompt,
            max_new_tokens=1000,
            do_sample=False,
            pad_token_id=self.tokenizer.tokenizer.pad_token_id
        )[0]


    def truncate_response(
        self,
        response: str
    ):
        """
        Truncate response by specific system prompt keywords.

        Args:
            response (str): raw response from LLM.

        Returns:
            response (str): truncated response.
        """
        if not self.is_truncate_response:
            return response

        response = response.split("model\n")[-1]

        return response