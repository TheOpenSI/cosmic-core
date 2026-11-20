### Core modules ###


### Type hints ###
from ..types.json_schemas import ChatHistorySchemaUpdate


### Internal modules ###


def _completed_pairs(
    details: list[ChatHistorySchemaUpdate]
) -> list[tuple[str, str]]:
    """
    Collect the (user query, LLM response) pairs of the completed blocks.

    Args:
        details (list[ChatHistorySchemaUpdate]): chat history blocks.

    Returns:
        pairs (list[tuple[str, str]]): pairs in chronological order, with the
            leading `<files>` tag stripped from each user query.
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

    return pairs


def build_messages_from_chat_history(
    details:    list[ChatHistorySchemaUpdate],
    num_pairs:  int = 5
) -> list[dict[str, str]]:
    """
    Build the prior conversation turns as chat messages, so the LLM receives a
    real multi-turn thread instead of only a history text block.

    Args:
        details (list[ChatHistorySchemaUpdate]): completed chat history blocks.
        num_pairs (int, optional): number of most recent pairs to include. Defaults to 5.

    Returns:
        messages (list[dict[str, str]]): alternating {"role": "user" | "assistant",
            "content": ...} entries, oldest first.
    """
    messages: list[dict[str, str]] = []

    for (user_msg, assistant_msg) in _completed_pairs(details=details)[-num_pairs:]:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    return messages


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
    pairs: list[tuple[str, str]] = _completed_pairs(details=details)

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
