### Core modules ###


### Type hints ###
from ..types.json_schemas import ChatHistorySchemaUpdate


### Internal modules ###


def build_context_from_chat_history(
    details:    list[ChatHistorySchemaUpdate],
    num_pairs:  int = 5
) -> str:
    """
    Build the LLM conversation context from the completed blocks of a `details`
    list. Returns an empty string when there are no complete pairs.

    Args:
        details (list[ChatHistorySchemaUpdate]):
            completed chat history blocks (attribute access). Value are passed
            directly as Pydantic types rather than decoded JSON version.

        num_pairs (int):
            number of most recent conversation pairs to include.

    Returns:
        context (str):
            formatted conversation history, or "" when none is available.
    """
    pairs: list[tuple[str, str]] = []

    for chat_history_field in details:
        if chat_history_field.user_query and chat_history_field.llm_response:
            processed_query: str = ""

            if "</files>" in chat_history_field.user_query:
                processed_query = chat_history_field.user_query.split("</files>", 1)[1].strip()

            else:
                processed_query = chat_history_field.user_query

            pairs.append(
                (
                    processed_query,
                    chat_history_field.llm_response
                )
            )

    selected_pairs: list[tuple[str, str]] = pairs[-num_pairs:]

    if not selected_pairs:
        return ""

    context_parts: list[str] = []

    for (
        index,
        (user_msg, assistant_msg)
    ) in enumerate(
        iterable=selected_pairs,
        start=1
    ):
        context_parts.append(
            "{1:s} Previous Conversation Pair {0:d} {1:s}\n{2:s}\n{3:s}".format(
                index,
                ("-" * 15),
                f"**User:** {user_msg}",
                f"**Assistant:** {assistant_msg}"
            )
        )

    context: str = "{0:s} Conversation History {0:s}\n{1:s}\n{0:s} End of Chat History {0:s}\n".format(
        ("=" * 15),
        "\n\n".join(context_parts)
    )

    return context