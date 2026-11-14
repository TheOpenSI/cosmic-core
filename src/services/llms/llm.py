### Core modules ###
from os import getenv
from dotenv import dotenv_values
from torch import (
    cuda,
    bfloat16,
    no_grad
)
from transformers import AutoModelForCausalLM
from peft import PeftModel
from openai import OpenAI
from transformers import pipeline


### Type hints ###


### Internal modules ###
from ...maps import LLM_INSTANCE_DICT, LLM_MODEL_DICT
from .LLMBase import LLMBase
from .Ollama import Ollama
from .prompts import system_prompt as system_prompt_instances
from .prompts import user_prompt as user_prompt_instances
from . import tokenizer as tokenizer_instances
from ..base import ServiceBase
from .login import LLMLogin
from ....utils.module import get_instance
from ....utils.log_tool import set_color


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


class GPT(LLMBase):
    def __init__(
        self,
        llm_name: str = "gpt-3.5-turbo",
        **kwargs
    ):
        """
        For OpenAI API.

        Args:
            llm_name (str, optional): LLM name in src/maps.py. Defaults to "gpt-3.5-turbo".
        """
        super().__init__(llm_name=llm_name, **kwargs)

        # Get API key stored in .env.
        api_key = self.get_openai_key()

        # OpenAI model entry with key.
        self.model = OpenAI(api_key=api_key)

        # OpenAI API call.
        self.llm = lambda system_prompt: \
            self.model.responses.create(
                model=llm_name,
                max_output_tokens=2048,
                temperature=0.0,
                input=system_prompt
            ).output[0].content[0].text


    def quit(self):
        """
        Close OpenAI API model entry.
        """
        self.model.close()


    def get_openai_key(self):
        """
        Get API key stored in .env.

        Returns:
            openai_key (str): API key.
        """
        # Set the key stored file.
        openai_key = getenv(
            key="OPENAI_API_KEY",
            default=""
        )

        if openai_key == "":
            envs = dotenv_values(
                dotenv_path=f"{self.root}/.env",
                stream=None,
                verbose=False,
                interpolate=True,
                encoding="utf-8"
            )

            if "OPENAI_API_KEY" in envs.keys():
                openai_key = envs["OPENAI_API_KEY"]

            else:
                print(
                    set_color(
                        status="warning",
                        information="OPENAI_API_KEY is required in .env."
                    )
                )
                openai_key = ""

        # Get warning for invalid API key.
        if openai_key == "":
            print(
                set_color(
                    status="warning",
                    information="The OPENAI_API_KEY in .env is invalid."
                )
            )

        return openai_key


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