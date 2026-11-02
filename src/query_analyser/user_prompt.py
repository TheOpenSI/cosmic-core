### Core modules ###


### Type hints ###
from typing import Any


### Internal modules ###
from ..services.llms.prompts.user_prompt import UserPromptBase


class QueryAnalyserService(UserPromptBase):
    # Keyed by registry name, so a hint is only given for an enabled service
    # and always at the id the registry assigned.
    HINTS_BY_NAME: dict[str, str] = {
        "chess":                "if the query is to predict the next chess move",
        "code_generation":      "if the query is to generate or modify a code",
        "academic_governance":  "if the query is about Academic Governance"
    }

    # Most recent characters of chat history kept in the routing prompt.
    MAX_HISTORY_CHARS: int = 1500

    def __init__(
        self,
        services: dict,
        service_ids_by_name: dict[str, str] | None = None,
        **kwargs
    ):
        """
        Initialize the instance.

        Args:
            services (dict): a dictionary of services ({id: description}).
            service_ids_by_name (dict[str, str] | None, optional): {service_name: service_id}
                from the live registry, used to pin routing-hint examples to the
                ids the registry actually assigned. When omitted, no hint
                examples are given (ids must never be guessed). Defaults to None.
        """
        super().__init__(**kwargs)

        # Set services.
        self.services = services
        self.num_services = len(self.services)

        # Build the hint chain from whichever hintable services are enabled.
        hints: list[str] = [
            f"{condition}, then select service {service_ids_by_name[name]}"
            for (name, condition) in self.HINTS_BY_NAME.items()
            if service_ids_by_name and name in service_ids_by_name
        ]
        self.routing_hints: str = (
            f" For instance, {'; otherwise, '.join(hints)}."
            if hints else ""
        )

        # Get the service string and option string for user prompt.
        service_string = ""
        option_string = ""

        # Get strings for user prompt.
        for idx, service_tag in enumerate(self.services):
            service = self.services[service_tag]
            service_string += f"service {service_tag}: {service}"
            option_string += f"service {service_tag}"

            if idx < self.num_services - 1:
                service_string += ", "
                option_string += " or "

        # Set as global variables.
        self.service_string = service_string
        self.option_string = option_string

    def __call__(
        self,
        question: str,
        context: dict[str, Any] | str = {}
    ) -> str:
        """
        Build user prompt to analyse the question.

        Args:
            question (str):
                the question.

            context (dict[str, Any] | str, optional):
                routing signals beyond the query text. A dict may hold
                "chat_history" (str), recent turns so follow-ups ("what about its
                committees?") are routed by what they refer to, and
                "file_names" (list[str]), files attached to this message. A
                plain string is treated as chat history. Defaults to {}.

        Returns:
            user_prompt (str): question with instruction.
        """
        # Normalise the routing signals.
        chat_history:   str         = ""
        file_names:     list[str]   = []

        if isinstance(context, dict):
            chat_history = str(context.get("chat_history", ""))
            file_names = [str(file_name) for file_name in context.get("file_names", [])]

        else:
            chat_history = context

        # Keep the routing prompt small for the analyser SLM; the most recent
        # turns matter most for follow-ups.
        chat_history = chat_history[-self.MAX_HISTORY_CHARS:]

        # Optional signal blocks, only included when present, so a bare query
        # costs no extra prompt tokens.
        history_block: str = (
            f" Recent conversation history (oldest first):\n'{chat_history}'\n"
            if chat_history else ""
        )
        files_block: str = (
            f" The user attached the file(s): {', '.join(file_names)}.\n"
            if file_names else ""
        )

        user_prompt: str = (
            f"You are a routing classifier. Given {self.num_services} services:"
            f" '{self.service_string}',"
            f" decide which single service should handle the user's query."
            f"\n{history_block}{files_block}"
            f" The query is '{question}'.\n"
            f" Routing rules, in order:\n"
            f" 1. Route by the INTENT of the query, not by keyword overlap with a"
            f" service name.\n"
            f" 2. If the query refers to earlier turns (for example 'it', 'that',"
            f" 'the previous one', 'what about...'), resolve what it refers to"
            f" from the conversation history and route by that resolved topic.\n"
            f" 3. If the query asks about an attached file (for example"
            f" 'summarise this', 'what does the document say'), that is a"
            f" QUESTION about the file, never a request to store it: route to"
            f" the service whose subject area matches the file and question, or"
            f" to the general question service if none matches. Never choose the"
            f" memory/database-update service just because a file is attached.\n"
            f" 4. An attached file does NOT change the route when the query is"
            f" unrelated to it.\n"
            f" 5. Only choose the memory/database-update service when the user"
            f" explicitly asks to remember, save, or store something for later"
            f" ('remember this', 'save this to memory'), not for questions.\n"
            f" 6. If several services could apply, choose the most specialised"
            f" one; if none apply, choose the general question service.\n"
            f"{self.routing_hints}"
            f" Answer in the form 'service N'."
            f" Just return which service without any explanations."
        )

        return user_prompt

class QueryAnalyserSystemInfo(QueryAnalyserService):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        """
        Initialize the instance.
        """
        super().__init__(*args, **kwargs)

        # Get the service string and option string for user prompt.
        service_string = ""

        # Get a string of services for user prompt.
        for idx, service_tag in enumerate(self.services):
            service = self.services[service_tag]
            service_string += f"{service}"

            if idx < self.num_services - 1:
                service_string += ", "

        # Replace self.service_string.
        self.service_string = service_string

        # Get system information.
        self.system_information = self.get_system_information()


    def get_system_information(self):
        """
        Get system information.

        Returns:
            system_information (str): system information.
        """
        system_information: str = "{0:s}. {1:s}. {2:s}{3:s}. {4:s}, {5:s}. {6:s}, {7:s}. {8:s}. {9:s}. {10:s}{11:s}. {12:s}.".format(
            "My name is OpenSI-CoSMIC.",
            " I am an AI system",
            "OpenSI-CoSMIC stands for the Open Source",
            "Institute-Cognitive System of Machine Intelligent Computing",
            "I am created, developed, and maintained by OpenSI",
            "which is an institute at University of Canberra",
            f"At the moment, I can provide {len(self.services)} services",
            f"including {self.service_string}",
            "I can design and provide more services under an agreement",
            "To request more services, please find the contact information in my profile",
            "My profile and project repository can be found at",
            "<https://github.com/TheOpenSI/CoSMIC>",
            "You take my role"
        )

        return system_information


    def __call__(
        self,
        question: str,
        context: dict = {}
    ):
        """
        Build user prompt to analyse the question.

        Args:
            question    (str):  the question.
            context     (dict): context, not used but reserve for interface uniform. Default to "".

        Returns:
            user_prompt (str): question with instruction.
        """
        # user_prompt = f"Given that '{self.system_information}'," \
        #     f" is the question '{question}' a general question related to the system" \
        #     f" information or OpenSI-CoSMIC?" \
        #     f" Just answer yes or no without any explainations."

        user_prompt: str = "{0:s}, {1:s} {2:s} {3:s}.".format(
            f"A user has asked the following question - '{question}'",
            "is the user asking information about you (the AI assistant called OpenSI-CoSMIC) or",
            "requesting information about how you work/what you can do?",
            "Answer only 'YES' or 'NO' without any explanations"
        )

        return user_prompt
