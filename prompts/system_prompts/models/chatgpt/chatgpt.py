from ..base import SystemPromptBase
from ...services import PROMPT_FILES


class GPT(SystemPromptBase):
    def __init__(self, **kwargs):
        """
        For GPT API.
        """
        super().__init__(**kwargs)


    def _load_service_prompt(self,service: str) -> str:
        """
        Load additional system prompt content for a service persona.

        The persona-to-file mapping lives in
        `prompts/system_prompts/services/__init__.py`.

        If the persona has no file, return an empty string.
        """
        if not service or not isinstance(service, str):
            return ""

        prompt_path = PROMPT_FILES.get(service)

        if prompt_path is None or not prompt_path.is_file():
            return ""

        return prompt_path.read_text(encoding="utf-8")


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