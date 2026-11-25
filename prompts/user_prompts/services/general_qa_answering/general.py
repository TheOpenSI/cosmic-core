from ...models.base import UserPromptBase


class GeneralUserPrompt(UserPromptBase):
    def __init__(self):
        """
        Initialize the instance.
        """
        super().__init__()


    def __call__(
        self,
        question: str,
        context: dict = {}
    ):
        """
        Build user prompt.

        Args:
            question (str): user question.
            context (str|dict, optional): external context. Defaults to "".

        Returns:
            user_prompt (str): user prompt.
        """
        # Get context.
        context = self.get_context(context)

        if context == "":
            user_prompt = question

        else:
            user_prompt = f"Given that '{context}', {question}"

        return user_prompt