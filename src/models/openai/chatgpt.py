### Core modules ###
from os import getenv
from openai import OpenAI


### Type hint modules ###


### Internal modules ###
from ..base import LLMBase
from ....utils.log_tool import set_color


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

        # Get the API key from the loaded environment.
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
        Get the OpenAI API key from the loaded environment.

        Returns:
            openai_key (str): API key.
        """
        # Set the key stored file.
        openai_key = getenv(
            key="OPENAI_API_KEY",
            default=""
        )

        # Get warning for a missing API key.
        if openai_key == "":
            print(
                set_color(
                    status="warning",
                    information="OPENAI_API_KEY is not set (add it to apis/cores/cosmic_cfg.env)."
                )
            )

        return openai_key