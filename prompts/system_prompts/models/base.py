### Core modules ###


### Type hint modules ###


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