### Core modules ###
from os import getenv
from pathlib import Path
from huggingface_hub import login


### Type hints ###


### Internal modules ###
from ..maps import LLM_MODEL_DICT


class LLMLogin:
    def __init__(
        self,
        llm_name: str
    ):
        """
        Login LLM.

        Args:
            model (str, optional): model name. Defaults to "base".
        """
        self.llm_name = llm_name


    def login(self):
        """
        Login huggingface if no local model found.
        """
        cache_model_name = "models--" + LLM_MODEL_DICT[self.llm_name].replace("/", "--")
        cache_model_directory: Path = Path("~/.cache/huggingface/hub").resolve(strict=True)
        cache_model_path: Path = cache_model_directory.joinpath(cache_model_name)

        if not cache_model_path.exists(follow_symlinks=True):
            # Required token for huggingface login (loaded from the env files).
            if self.llm_name.find("finetune") > -1:
                login(
                    token=getenv(key="hf_token_finetune"),
                    add_to_git_credential=True,
                    skip_if_logged_in=True
                )
            else:
                login(
                    token=getenv(key="hf_token"),
                    add_to_git_credential=True,
                    skip_if_logged_in=True
                )