class UserPromptBase:
    def __init__(self):
        """
        Initialize the instance.
        """
        pass


    def get_context(
        self,
        context: str = ""
    ):
        """
        Get context based on the input type.

        Args:
            context (str|dict, optional): context, string or dictionary.
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