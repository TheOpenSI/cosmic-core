### Core modules ###
from pathlib import Path


### Type hints ###


### Internal modules ###


class SystemPromptBase:
    def __init__(
        self,
        use_example: bool=False,
        prefix: str = "{0:s} {1:s}. {2:s}. {3:s}. {4:s}. {5:s}.".format(
            "SYSTEM IDENTITY:",
            "You are OpenSI-CoSMIC, a helpful assistant developed by Open Source Institute at University of Canberra",
            "If the question is not clear, ask for clarification instead of making assumptions",
            "You would have access to conversation history, this is for your context only",
            "Always answer the question even if the context is not helpful",
            "If you don't know the answer, say you don't know, but try to provide some helpful information if possible"
        )
    ):
        """
        System prompt base.

        Args:
            use_example (bool, optional):   use example in system prompt to detect keywords for
                                            response truncation. Defaults to False.
            prefix      (str):              prefix to start the prompt. Default to "".
        """
        self.use_example = use_example
        self.prefix = prefix


    def set_prefix(
        self,
        prefix: str
    ):
        """
        Set prefix externally.

        Args:
            prefix (str): prefix for system prompt.
        """
        self.prefix = prefix


    def get_context(
        self,
        context: str = ""
    ):
        """
        Get context based on the input type.

        Args:
            context (str|dict, optional):   context, string or dictionary.
                                            Defaults to "".

        Returns:
            context: extract context or an empty string.
        """
        if isinstance(context, dict):
            if "context" in context:
                context = context["context"]
            else:
                context = ""

        return context


    def set_use_example(
        self,
        use_example: bool
    ):
        """
        Set use_example externally.

        Args:
            use_example (bool): use example in system prompt.
        """
        self.use_example = use_example


    def __call__(
        self,
        user_prompt: str,
        context: str = ""
    ):
        """
        Merge user_prompt in system prompt as the question containing context.

        Args:
            user_prompt (str):                  user prompt.
            context     (str|dict, optional):   context retrieved if applicable. Defaults to "".
        """
        # Need to be implemented, otherwise raise error.
        raise NotImplementedError


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


class Gemma7b(SystemPromptBase):
    def __init__(
        self,
        prefix = "<bos>",
        **kwargs
    ):
        """
        For Gemma 7B.

        Args:
            prefix (str): prompt prefix. Default to "<bos>".
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
                system_prompt += \
                    "<start_of_turn>user\n" \
                    "What is the capital of Australia?<end_of_turn>\n"
            else:
                system_prompt += \
                    "<start_of_turn>user\n" \
                    "Given that 'Canberra is the capital of Australia'," \
                    " what is the capital of Australia?<end_of_turn>\n"

            system_prompt += \
                "<start_of_turn>model\n" \
                "Canberra<end_of_turn><eos>\n" \
                "<start_of_turn>user\n" \
                f"{user_prompt}<end_of_turn>\n" \
                "<start_of_turn>model"
        else:
            if context != "":
                system_prompt += \
                    " Always answer the question briefly even if the context isn't useful."

            system_prompt += f" {user_prompt}"

        return system_prompt


class Gemma7bIt(Mistral7bInstructv01):
    def __init__(self, **kwargs):
        """
        For Gemma 7B Instruction.
        """
        super().__init__(**kwargs)


class GPT(SystemPromptBase):
    def __init__(self, **kwargs):
        """
        For GPT API.
        """
        super().__init__(**kwargs)
        self._prompts_root = Path.cwd() / "src" / "services" / "llms" / "prompts"


    def _load_service_prompt(self,service: str) -> str:
        """
        Attempt to load additional sytem prompt content from a text file under:
            ./src/services/llms/prompts/<services> or <services>.txt

        If file is not found or `service` is falsy, return empty string.
        """
        if not service or not isinstance(service, str):
            return ""

        prompts_root = self._prompts_root
        # Try exact filename first (no extension), then .txt
        candidate_paths = [
            prompts_root / service,                 # e.g., prompts/promptA
            prompts_root / f"{service}.txt",        # e.g., prompts/promptA.txt
        ]

        for p in candidate_paths:
            try:
                if p.is_file():
                    return p.read_text(encoding="utf-8")
            except Exception:
                pass

        # If no file found/readable
        return ""


    def __call__(
        self,
        user_prompt: str,
        context: str = "",
        service: str = ""
    ):
        """
        Apply system prompt with user prompt and context.

        Args:
            user_prompt (str):                  question with context.
            context     (str|dict, optional):   context retrieved. Defaults to "".
            services    (str, optional):        service name for loading specific system prompt. Defaults to "".

        Returns:
            system_prompt (str): system prompt with question and context under LLM query format.
        """

        # Compose the system content: base self.prefix + (optional) file content
        prompt_service = self._load_service_prompt(service)
        composed_prefix = self.prefix + (prompt_service if prompt_service else "")

        system_prompt = [
            {
                "role": "system",
                "content": composed_prefix
            },
            {"role": "user", "content": user_prompt}
        ]

        return system_prompt


class Ollama(GPT):
    def __init__(self, **kwargs):
        """
        For Ollama model.
        """
        super().__init__(**kwargs)


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


class FENNextMoveAnalyse(SystemPromptBase):
    def __init__(self, **kwargs):
        """
        For analysis of next move prediction given a FEN.
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

        if context == "":
            system_prompt = user_prompt
        else:
            system_prompt = f"{user_prompt}\nIf the context is useless, ignore it."

        return system_prompt


class FENNextMoveAnalyseMistralFinetuned(SystemPromptBase):
    def __init__(
        self,
        prefix = "<s>",
        **kwargs
    ):
        """
        For analysis of next move prediction given a FEN and a finetuned LLM.

        Args:
            prefix (str): prompt prefix. Default to "<s>".
        """
        super().__init__()


    def __call__(
        self,
        user_prompt: str,
        context: str = ""
    ):
        """
        Apply system prompt with user prompt and context.

        Args:
            user_prompt (str): question with context.
            context (str|dict, optional): context retrieved. Defaults to "".

        Returns:
            system_prompt (str): system prompt with question and context under LLM query format.
        """
        system_prompt = \
            f"{self.prefix}### Instruction:\n{user_prompt}\n### Context: \n \n### Response:"

        return system_prompt