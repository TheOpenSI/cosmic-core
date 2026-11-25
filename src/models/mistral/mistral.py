from torch import (
    bfloat16,
    cuda,
    no_grad
)
from transformers import (
    AutoModelForCausalLM,
    pipeline
)
from peft import PeftModel


from ...maps import LLM_MODEL_DICT
from ..base import LLMBase
from ..login import LLMLogin


class Mistral7bv01(LLMBase):
    def __init__(
        self,
        llm_name: str = "mistral-7b-v0.1",
        **kwargs
    ):
        """
        For Mistral 7B.

        Args:
            llm_name (str, optional): LLM name in src/maps.py. Defaults to "mistral-7b-v0.1".
        """
        super().__init__(llm_name=llm_name, **kwargs)

        # Login if model has been downloaded locally.
        LLMLogin(llm_name).login()

        # Load model to GPU.
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_DICT[llm_name],
            use_cache=True,
            device_map=self.device,
            torch_dtype=bfloat16,
            quantization_config=self.quantization_config,
        )  # low_cpu_mem_usage=True

        # Build QA pipeline.
        self.llm = lambda system_prompt: pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer.tokenizer,
            do_sample=False,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )(system_prompt)[0]["generated_text"]


    def quit(self):
        """
        Release model memory and instance.
        """
        if not self.is_quantized:
            self.model = self.model.to("cpu")

        del self.model
        cuda.empty_cache()


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

        if self.use_example:  # with an example in the prompt, can always parse by [INST]
            response = response.split("[/INST]")[0].split("[INST]")[0]

        return response


    def __call__(
        self,
        question: str,
        context: str | dict = ""
    ):
        """
        Process the question answering.
        Set LLM model to evaluation model, which is only applicable to local model but rather OpenAI API.

        Args:
            question    (str):                  user question in string.
            context     (str | dict, optional): context retrieved externally if applicable. Defaults to "".

        Returns:
            response        : truncated response.
            raw_response    : original response without truncation.
        """
        # Set model to evaluation mode.
        self.model.eval()

        with no_grad():  # without modelg gradients
            # Use the parent process interface.
            return super().__call__(
                question,
                context=context
            )



class Mistral7bInstructv01(Mistral7bv01):
    def __init__(
        self,
        llm_name = "mistral-7b-instruct-v0.1",
        **kwargs
    ):
        """
        For Mistral 7B Instruction.

        Args:
            llm_name (str, optional): LLM name in src/maps.py. Defaults to "mistral-7b-instruct-v0.1".
        """
        super().__init__(llm_name=llm_name, **kwargs)

        # Set up LLM.
        self.llm = lambda system_prompt: self.model.generate(
            system_prompt,
            max_new_tokens=1000,
            do_sample=False,
            attention_mask=self.attention_mask(system_prompt),
            pad_token_id=self.tokenizer.tokenizer.pad_token_id,
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

        response = response.split("[/INST]")[-1].split("</s>")[0]

        return response



class MistralFinetuned(Mistral7bv01):
    def __init__(
        self,
        llm_name: str = "mistral-7b-finetuned",
        use_example: bool = False,
        is_quantized: bool = True,
        **kwargs
    ):
        """
        For Mistral 7B finetuned model.

        Args:
            llm_name        (str, optional):    LLM name in src/maps.py. Defaults to "mistral-7b-finetuned".
            use_example     (bool, optional):   use example instance. Defaults to False.
            is_quantized    (bool, optional):   use quantized model, always true. Default to True.
        """
        super().__init__(llm_name=llm_name, use_example=use_example, is_quantized=True, **kwargs)

        # Use the base model to build model.
        base_llm_model = "mistral-7b-v0.1"

        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_DICT[base_llm_model],
            quantization_config=self.quantization_config,
            use_cache=True,
            device_map="auto"
        )  # low_cpu_mem_usage=True

        self.model = PeftModel.from_pretrained(
            base_model,
            LLM_MODEL_DICT[llm_name]
        )

        # Set up LLM.
        self.llm = lambda system_prompt: self.model.generate(
            system_prompt,
            attention_mask=self.attention_mask(system_prompt),
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=self.tokenizer.tokenizer.eos_token_id
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
        # Return the raw response if truncation is not required.
        if not self.is_truncate_response:
            return response

        response = response.split('###')[-1]

        if response.find("<answer>:") > -1 or response.find("<ANSWER>:") > -1:
            response = response.split("<answer>:")[-1].split("<ANSWER>:")[-1]

        return response