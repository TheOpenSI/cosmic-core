from utils.env import PROJECT_ROOT
from prompts.system_prompts.base import SystemPromptBase


class GPT(SystemPromptBase):
    def __init__(self, **kwargs):
        """
        For GPT API.
        """
        super().__init__(**kwargs)
        self._prompts_root = PROJECT_ROOT / "prompts" / "system_prompts" / "services"


    def _load_service_prompt(self,service: str) -> str:
        """
        Attempt to load additional system prompt content from a text file under:
            prompts/system_prompts/services/<service> or <service>.txt

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