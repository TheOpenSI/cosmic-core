from prompts.system_prompts.base import SystemPromptBase


class Mistral7bv01(SystemPromptBase):
    def __init__(
        self,
        prefix = "<s>",
        **kwargs
    ):
        """
        For Mistral 7B.

        Args:
            prefix (str): prompt prefix. Default to "<s>".
        """
        super().__init__(prefix=prefix, **kwargs)


    def __call__(
        self,
        user_prompt: str,
        context: str = ""
    ):
        """
        Apply system prompt with user prompt and context.

        Args:
            user_prompt (str):                  question with context.
            context     (str|dict, optional):   context retrieved. Defaults to "".

        Returns:
            system_prompt (str): system prompt with question and context under LLM query format.
        """
        # Get context.
        context = self.get_context(context)

        system_prompt = self.prefix

        if self.use_example:
            if context == "":
                system_prompt += " [INST] What is the capital of Australia? [/INST]\n"
            else:
                system_prompt += \
                    " [INST] Given that 'Canberra is the capital of Australia'," \
                    " what is the capital of Australia? [/INST]\n"

            system_prompt += \
                "Canberra</s>\n" \
                f"[INST] {user_prompt} [/INST]"
        else:
            if context != "":
                system_prompt += \
                    " Always answer the question briefly even if the context isn't useful."

            system_prompt += f" {user_prompt}"

        return system_prompt




class Mistral7bInstructv01(SystemPromptBase):
    def __init__(self, **kwargs):
        """
        For Mistral 7B Instruction.
        """
        super().__init__(**kwargs)


    def __call__(
        self,
        user_prompt: str,
        context: str = ""
    ):
        """
        Apply system prompt with user prompt and context.

        Args:
            user_prompt (str):                  question with context.
            context     (str|dict, optional):   context retrieved. Defaults to "".

        Returns:
            system_prompt (str): system prompt with question and context under LLM query format.
        """
        # Get context.
        context = self.get_context(context)

        system_prompt = []

        if self.use_example:
            if context == "":
                system_prompt.append({"role": "user", "content": "What is the capital of Australia?"})
            else:
                system_prompt.append({
                    "role": "user",
                    "content": "Given that 'Canberra is the capital of Australia', what is the capital of Australia?"
                })

            system_prompt.append({"role": "assistant", "content": "Canberra"})

        system_prompt.append({"role": "user", "content": user_prompt})

        return system_prompt




class MistralFinetuned(SystemPromptBase):
    def __init__(
        self,
        prefix = "<s>",
        **kwargs
    ):
        """
        For Mistral 7B Finetuned LLM.

        Args:
            prefix (str): prompt prefix. Default to "<s>".
        """
        super().__init__(prefix=prefix, **kwargs)


    def __call__(
        self,
        question: str,
        context: str = ""
    ):
        """
        Apply system prompt with user prompt and context.

        Args:
            user_prompt (str):                  question with context.
            context     (str|dict, optional):   context retrieved. Defaults to "".

        Returns:
            system_prompt (str): system prompt with question and context under LLM query format.
        """
        # Get context.
        context = self.get_context(context)

        system_prompt = \
            f"{self.prefix}### Instruction:\n{question}\n### Context: \n{context}\n### Response:"

        return system_prompt