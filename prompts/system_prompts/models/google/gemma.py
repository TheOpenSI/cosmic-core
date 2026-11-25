### Core modules ###


### Type hint modules ###


### Internal modules ###
from ..base import SystemPromptBase
from ..mistral.mistral import Mistral7bInstructv01


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