from transformers import AutoTokenizer


from ...maps import LLM_MODEL_DICT
from ...models.login import LLMLogin
from ..base import TokenizerBase


class Mistral7bv01(TokenizerBase):
    def __init__(
        self,
        llm_name: str = "mistral-7b-v0.1",
        **kwargs
    ):
        """
        For Mistral 7B.

        Args:
            llm_name (str, optional): LLM name. Defaults to "mistral-7b-v0.1".
        """
        super().__init__(llm_name, **kwargs)

        # Login if model is not downloaded locally.
        LLMLogin(llm_name).login()

        # Load tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_DICT[llm_name],
            add_eos_token=False
        )

        # Set tokenizer pad_token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token



class Mistral7bInstructv01(Mistral7bv01):
    def __init__(
        self,
        llm_name: str = "mistral-7b-instruct-v0.1",
        **kwargs
    ):
        """
        For Mistral 7B Instruction.

        Args:
            llm_name (str, optional): LLM name. Defaults to "mistral-7b-instruct-v0.1".
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
        return self.tokenizer.decode(response, **kwargs)



class MistralFinetuned(Mistral7bv01):
    def __init__(
        self,
        llm_name: str = "",
        **kwargs
    ):
        """
        For Mistral 7B finetuned.
        Since the tokenizer depends on base model, not finetuned model, remaining the definition internally.

        Args:
            llm_name (str, optional): LLM name. Defaults to "".
        """
        super().__init__(llm_name, **kwargs)
        base_llm_name = "mistral-7b-v0.1"

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_DICT[base_llm_name],
            add_bos_token=True
        )


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