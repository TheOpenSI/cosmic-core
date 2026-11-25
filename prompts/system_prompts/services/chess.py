from prompts.system_prompts.base import SystemPromptBase


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