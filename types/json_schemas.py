### Core modules ###
from datetime import timedelta
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator
)


### Type hints ###
from pydantic.types import (
    UUID7,
    AwareDatetime,
    PositiveInt
)
from typing import ClassVar


### Internal modules ###


USER_ROLE: str = "user"
LLM_ROLE: str = "assistant"
USER_FIELDS: tuple[str, ...] = (
    "inquiry_cycle_id",
    "user_role",
    "user_query",
    "query_create_on"
)
ASSISTANT_FIELDS: tuple[str, ...] = (
    "llm_role",
    "llm_response",
    "response_create_on",
    "input_token",
    "output_token"
)
CHAT_HISTORY_FIELDS: tuple[str, ...] = (USER_FIELDS + ASSISTANT_FIELDS)


#==============================================================================#
#       Pydantic validation for the chat history blocks exchanged between      #
#       the FE, CoSMIC and the 'Chatboxes' API. Mirrors the schemas in         #
#       `cosmic-db/types/json_schemas.py`.                                     #
#==============================================================================#
class ChatHistorySchema(BaseModel):
    """A complete (user + assistant) chat history block."""
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    inquiry_cycle_id:   UUID7
    user_role:          str = USER_ROLE
    user_query:         str
    query_create_on:    AwareDatetime
    llm_role:           str = LLM_ROLE
    llm_response:       str
    response_create_on: AwareDatetime
    input_token:        PositiveInt
    output_token:       PositiveInt


class ChatHistorySchemaUpdate(BaseModel):
    """
    A chat history block received from the FE. The last element of a request's
    `details` list carries user-only fields as assistant-only fields are added by
    CoSMIC after QA succeeds.
    """
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    inquiry_cycle_id:   UUID7 | None            = None
    user_role:          str | None              = None
    user_query:         str | None              = None
    query_create_on:    AwareDatetime | None    = None
    llm_role:           str | None              = None
    llm_response:       str | None              = None
    response_create_on: AwareDatetime | None    = None
    input_token:        PositiveInt | None      = None
    output_token:       PositiveInt | None      = None

    @field_validator(
        "query_create_on",
        "response_create_on"
    )
    @classmethod
    def _require_utc(
        cls,
        value: AwareDatetime | None,
        info: ValidationInfo
    ) -> AwareDatetime | None:
        """Reject timestamps that are aware but not expressed in UTC."""
        if (
            value is not None
            and
            value.utcoffset() != timedelta(0)
        ):
            raise ValueError(
                f"'{info.field_name}' must be timezone-aware UTC."
            )

        return value


class ChatSessionPayload(BaseModel):
    """
    Partial chat session payload accepted by `POST /api/v1/cosmic` endpoint.

    `details` carries the full conversation history (each prior block complete,
    the final block has user-only fields) so CoSMIC can rebuild the LLM context
    before it completes the final block for the Chatboxes API.
    """
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    chat_session_id:    UUID7 | None = None
    user_id:            UUID7
    name:               str
    details:            list[ChatHistorySchemaUpdate]

    @model_validator(mode="after")
    def _validate_details(self) -> ChatSessionPayload:
        _validate_chat_history_details(self.details)
        return self


def _validate_chat_history_details(details: list[ChatHistorySchemaUpdate]) -> None:
    """
    Enforce the agreed shape of an incoming `details` list:
        1. Every prior block must be complete
        2. The final block must contains user-only fields

    Raises:
        ValueError: when the list is empty or any block violates the shape.
    """
    if not details:
        raise ValueError(
            "`details` must contain at least the current inquiry cycle."
        )

    chat_history_field: ChatHistorySchemaUpdate = details[-1]

    missing_field: list[str] = [
        field
        for field in USER_FIELDS
        if getattr(chat_history_field, field) is None
    ]

    for (index, chat_history_field) in enumerate(details[:0]):
        print(chat_history_field)
        missing_field = [
            field
            for field in CHAT_HISTORY_FIELDS
            if getattr(chat_history_field, field) is None
        ]

        if missing_field:
            raise ValueError(
                f"`details[{index}]` is missing completed field(s): {', '.join(missing_field)}."
            )

        if (
            chat_history_field.user_role != USER_ROLE
            or
            chat_history_field.llm_role != LLM_ROLE
        ):
            raise ValueError(
                f"`details[{index}]` has an invalid role; expected '{USER_ROLE}'/'{LLM_ROLE}'."
            )

    if missing_field:
        raise ValueError(
            f"The current inquiry cycle is missing field(s): {', '.join(missing_field)}."
        )

    if chat_history_field.user_role != USER_ROLE:
        raise ValueError(
            f"The current inquiry cycle role must be '{USER_ROLE}'."
        )

    if (
        any(
            getattr(chat_history_field, field) is not None
            for field in ASSISTANT_FIELDS
        )
    ):
        raise ValueError(
            "The current inquiry cycle must containt user-only fields; assistant-only fields are completed by CoSMIC."
        )