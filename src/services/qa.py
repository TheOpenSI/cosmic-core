### Core modules ###
from datetime import (
    datetime,
    timezone
)
from os import getenv
from pathlib import Path
import uuid


### Type hints ###
from typing import Any


### Internal modules ###
from . import chess as chess_instances
from .base import ServiceBase
from .document_index import DocumentMetadata
from .llms.LLMBase import LLMBase
from .rag import RAGBase
from .attachment_context import (
    AttachmentContext,
    AttachmentResolver,
    MISSING_FILE_LLM_NOTE,
    MISSING_FILE_USER_NOTE,
    SESSION_FILE_LIST_ROUTE,
    UNPLACED_SERVICE_IDS,
    session_file_list_answer,
    unused_file_note
)
from ...modules.code_generation.code_generation import CodeGenerator
from .system_information_service import SystemInformationService
from .fallback_service import FallbackService
from ..maps import GENERAL_SERVICE_ID
from ...utils.log_tool import set_color


# System prompt (prompts/document_qa.txt) for questions answered from a file.
DOCUMENT_QA_PERSONA: str = "document_qa"

# Services without a persona of their own: the general service and the
# forced route used when a file is attached to an unsure query.
GENERIC_SERVICE_IDS: tuple[str, ...] = (GENERAL_SERVICE_ID, "rag_fallback")

# Closing line of the block written by `build_context_from_chat_history()`.
CHAT_HISTORY_END_MARKER: str = "End of Chat History"

# Closing line of the block written by `build_context_from_chat_history()`.
CHAT_HISTORY_END_MARKER: str = "End of Chat History"

# Instruction for turning a follow-up question into a standalone search query.
QUERY_REWRITE_PROMPT: str = (
    "Rewrite the user's latest question as a standalone search query, using the "
    "conversation only to resolve references such as 'it', 'that' or 'the previous "
    "one'. Keep names, codes and terms exactly as written. If the question is "
    "already standalone, return it unchanged. Return only the query, without "
    "an answer or an explanation."
)

# Prior messages (user and assistant) given to the query rewrite.
QUERY_REWRITE_HISTORY_MESSAGES: int = 6

# Longest rewrite accepted; anything longer is likely an answer, not a query.
QUERY_REWRITE_MAX_CHARS: int = 500


class QABase(ServiceBase):
    def __init__(
        self,
        query_analyser:             LLMBase,
        llm:                        LLMBase,
        rag:                        RAGBase,
        code_generator:             CodeGenerator,
        system_information_service: SystemInformationService,
        fallback_service:           FallbackService,
        config:                     str | None = None,
        **kwargs:                   Any
    ) -> None:
        """
        Base class for QA.

        Args:
            query_analyser (LLMBase): The query analyser.

            llm (LLMBase): LLM instance.

            rag (RAGBase): RAG instance containing vector database service.

            code_generator (CodeGenerator): code generation service.

            system_information_service (SystemInformationService): system information service.

            fallback_service (FallbackService): fallback service.

            config (str, optional): config file to extract settings. Default to None.
        """
        super().__init__(**kwargs)

        # Set config globally.
        self.query_analyser             = query_analyser
        self.llm                        = llm
        self.rag                        = rag
        self.code_generator             = code_generator
        self.system_information_service = system_information_service
        self.fallback_service           = fallback_service


    def _add_service_0(self, services: dict[str, dict[str, str]]) -> None:
        """
        Add default service 0 to the services dictionary if not present.
        NOTE: This is an inplace mutation of the `services` dictionary.

        Args:
            services (dict[str, dict[str, str]]): Dictionary of services to be updated.
        """
        if "0" not in services:
            services["0"] = {
                "name": "system_information",
                "desc": (
                    "Answer questions about the AI assistant itself such as who created it, "
                    "what OpenSI-CoSMIC is, and what it can do."
                )
            }


    def _get_services_name(self, services: dict[str, dict[str, str]]) -> dict[str, str]:
        """
        Extracts the names of the services from the services dictionary.

        Args:
            services (dict[str, dict[str, str]]): Dictionary of services.

        Returns:
            dict[str, str]: Dictionary mapping service IDs to their names.
        """
        return {service_id: service_info["name"] for service_id, service_info in services.items()}


    @staticmethod
    def _route_session_files(
        service_option: str,
        attachment: AttachmentContext,
        attachment_resolver: AttachmentResolver,
        user_id: str | None,
        session_id: str | None
    ) -> str:
        """
        Route questions about the files shared earlier in the chat session.

        "Which files did I share?" is answered from the session index. A
        question the query analyser could not place (system information or
        fallback) in a session with files is most likely a follow-up about those
        files ("biomedical research data?"), so it searches them instead,
        unless it is about the assistant itself.

        Args:
            service_option (str): service selected by the query analyser.
            attachment (AttachmentContext): resolved files of the question; its
                target files are set to the session's files when rerouted.
            attachment_resolver (AttachmentResolver): looks up the session's files.
            user_id (str | None): owner of the session.
            session_id (str | None): chat session.

        Returns:
            service_option (str): `SESSION_FILE_LIST_ROUTE`, or the unchanged
                option (the force route then sends session-file follow-ups to
                retrieval).
        """
        if attachment.asks_for_file_list:
            return SESSION_FILE_LIST_ROUTE

        if (
            service_option not in UNPLACED_SERVICE_IDS
            or attachment.document_ids
            or attachment.is_system_question
            or not (user_id and session_id)
        ):
            return service_option

        (
            attachment.document_ids,
            attachment.file_names
        ) = attachment_resolver.get_session_files(user_id, session_id)

        if attachment.document_ids:
            print(
                set_color(
                    status="info",
                    information=f"[qa] Unplaced follow-up in a session with files; searching {attachment.file_names}."
                )
            )

        return service_option


    def _force_route_check(
        self,
        service_option: str,
        has_attachment: bool
    ) -> str:
        """
        If the turn is about a file but the query analyser routed to "0" (system info)
        or "-1" (fallback) neither of which use RAG, force it into the
        generic RAG (else) branch so the file actually gets used. This covers
        files attached this turn and files resolved from earlier in the session
        for a follow-up.

        Args:
            service_option (str): The current service option selected by the query analyser.

            has_attachment (bool): Whether the turn has files, attached now or earlier
                in the session.

        Returns:
            str: The updated service option, if necessary.
        """
        if has_attachment and service_option in ("0", "-1"):
            return "rag_fallback"
        return service_option


    def _get_stockfish_binary_path(self) -> str:
        """
        Get the path to the Stockfish binary.
        # TODO: this will be updated once we've service-specific configs implemented

        Returns:
            str: Absolute path to the Stockfish binary.
        """
        return str(
            Path(__file__).resolve(strict=True).parent.parent.parent.joinpath(
                "third_party",
                "stockfish",
                "stockfish-ubuntu-x86-64-avx2"
            )
        )


    def _handle_chess_service(
        self,
        service_option: str,
        service_info_dict: dict,
        context: str | dict,
    ) -> tuple[str, str, int, int]:
        """
        Predict the next chess move ("1.0": from a FEN, "1.1": from a move
        sequence) and ask the LLM to explain why it's the best choice.

        Args:
            service_option (str): "1.0" or "1.1", selecting which Stockfish predictor to use.

            service_info_dict (dict): info dict from the query analyser; carries "fen" for "1.0"
                or "moves" for "1.1".

            context (str | dict): context associated with the question; used to derive the
                move notation mode.

        Returns:
            response (str): truncated answer if applicable.

            raw_response (str): original answer from LLM.

            input_token (int): total amount of input tokens captured by LLM.

            output_token (int): total amount of output tokens generated by LLM.
        """
        binary_path = self._get_stockfish_binary_path()
        move_mode = "algebric" if context == "" else context

        if service_option == "1.0":
            current_fen = service_info_dict["fen"]
            next_move_predictor = chess_instances.StockfishFENNextMove(binary_path=binary_path)
            next_move = next_move_predictor(
                fen=current_fen,
                move_mode=str(object=move_mode),
                topk=5
            )
            move_prediction_context = f"The current chess FEN is {[current_fen]}."

        else: # this is for prediction given moves, service_option == "1.1":
            current_moves = service_info_dict["moves"]
            next_move_predictor = chess_instances.StockfishSequenceNextMove(binary_path=binary_path)
            next_move = next_move_predictor(
                moves=current_moves,
                move_mode=str(object=move_mode),
                topk=5
            )
            move_prediction_context = f"The previous chess moves are {[current_moves]}."

        # Explain why these moves are feasible.
        user_prompt = f"Select the best next move from {next_move} and explain why it is the best."
        (
            response,
            raw_response,
            input_token,
            output_token
        ) = self.llm(
            question=user_prompt,
            context=move_prediction_context
        )

        # Attach all the moves in case LLM cannot select the best one.
        response        = f"The next moves are from {next_move}.\n{response}"
        raw_response    = f"The next moves are from {next_move}.\n{raw_response}"

        return response, raw_response, input_token, output_token


    # TODO: extract into a dedicated MemoryService class, mirroring
    # system_information_service/fallback_service/code_generator
    def _handle_memory_update_service(
        self,
        query: str,
        services: dict[str, dict[str, str]],
        services_name: dict[str, str],
        memory_service_active: bool,
        user_id: str | None,
    ) -> tuple[str, str]:
        """
        Save the query to the user's long-term memory when the memory
        service is active and enabled; otherwise fall back to a generic
        response.

        Args:
            query (str): the question, saved verbatim as the memory text.

            services (dict[str, dict[str, str]]): dictionary of services, passed through 
                to the fallback service if memory can't be used.

            services_name (dict[str, str]): mapping of service id to service name, used 
                to confirm that service "2" is configured as "memory".

            memory_service_active (bool): whether the memory service is enabled for this request.

            user_id (str | None): the current user, required to scope the saved memory.

        Returns:
            response (str): truncated answer.

            raw_response (str): original answer.
        """
        selected_service_name = services_name.get("2", "").lower()

        if (
            memory_service_active
            and (selected_service_name == "memory")
            and (user_id)
        ):
            payload = DocumentMetadata(
                document_id=uuid.uuid4().hex[:8],
                user_id=user_id,
                memory_type="user",
            ).to_vector_payload()

            self.rag.vector_database.update_database_from_text(
                text=query,
                extra_metadata=payload
            )

            return "Saved to your memory.", "Saved to your memory."

        # fallback to generic response if memory service is not active or not configured
        return self.fallback_service(services=services)


    def _is_rag_eligible(
        self, 
        services: dict[str, dict[str, str]], 
        service_option: str
    ) -> bool:
        """
        Check if the selected service is eligible for RAG.

        Args:
            services (dict[str, dict[str, str]]): Dictionary of services.

            service_option (str): Selected service option.

        Returns:
            bool: True if the service is eligible for RAG, False otherwise.
        """
        try:
            target_service = services[service_option]
            return target_service.get("memory_capability") is True
        
        except KeyError:
            print(
                set_color(
                    "info",
                    f"[qa] Service option '{service_option}' has no entry in services "
                    "(expected for the forced-route fallback); treating as not RAG-eligible."
                )
            )
            return False


    def _use_rag(
        self,
        services: dict[str, dict[str, str]],
        service_option: str,
        is_rag: bool,
        has_files: bool,
        attachment: AttachmentContext
    ) -> bool:
        """
        Determine whether to use RAG based on the service option, RAG eligibility,
        and whether files are attached.

        Args:
            services (dict[str, dict[str, str]]): Dictionary of services.

            service_option (str): Selected service option.

            is_rag (bool): Whether RAG is enabled for this request.

            has_files (bool): Whether files are attached to the query.

            attachment (AttachmentContext): resolved files of the question.

        Returns:
            bool: True if RAG should be used, False otherwise.
        """
        is_rag_eligible_service = self._is_rag_eligible(services, service_option)
        use_rag = (
                (is_rag) and
                (
                    is_rag_eligible_service
                    or has_files
                    or bool(attachment.document_ids)
                )
            )

        return use_rag


    @staticmethod
    def _strip_chat_history(
        context: str | dict
    ) -> str | dict:
        """
        Remove the chat history block from a context.

        The block written by `build_context_from_chat_history()` always comes
        first, so everything up to the end of its closing marker line is dropped.
        Only the main answer call uses this: it receives the history as chat
        messages, while the other services still need it in the context.

        Args:
            context (str | dict): context, or a dict holding it under "context".

        Returns:
            context (str | dict): the context without the chat history block.
        """
        if isinstance(context, dict):
            return {
                **context,
                "context": QABase._strip_chat_history(context=context.get("context", ""))
            }

        marker_index: int = context.find(CHAT_HISTORY_END_MARKER)

        if marker_index < 0:
            return context

        line_end: int = context.find("\n", marker_index)

        return "" if line_end < 0 else context[line_end + 1:]


    def _split_chat_history_context(self, context: str) -> tuple[str, str]:
        """
        Split context into its chat-history and non-history parts, based on
        whether it carries the "Conversation History" marker written by
        `build_context_from_chat_history()`.

        Args:
            context (str): context associated with the question.

        Returns:
            chat_history_context (str): context if it holds chat history, else "".

            rag_context (str): context if it doesn't hold chat history, else "".
        """
        is_chat_history = "Conversation History" in context
        return (context, "") if is_chat_history else ("", context)


    @staticmethod
    def _select_persona(
        service_option: str,
        routed_service_name: str,
        answered_from_file: bool
    ) -> str:
        """
        Pick the system prompt that answers this turn.

        A specialist service keeps its own persona. The general service (and
        the forced file route) has no persona of its own, so a question it
        answers from an attached file uses the neutral document-QA persona.

        Args:
            service_option (str): service answering the question.

            routed_service_name (str): name of that service.

            answered_from_file (bool): the file contributed retrieved context.

        Returns:
            persona (str): service name used to load the system prompt.
        """
        if answered_from_file and service_option in GENERIC_SERVICE_IDS:
            return DOCUMENT_QA_PERSONA

        return routed_service_name


    @staticmethod
    def _format_sources(
        sources: list[dict]
    ) -> str:
        """
        List where the retrieved context came from, one line per source.

        Sources keep relevance order and are de-duplicated by (tier, title,
        service). A global source also names the service it belongs to.

        Args:
            sources (list[dict]): origin of each retrieved chunk.

        Returns:
            citation (str): "Sources:" block, or "" if there are no sources.
        """
        if not sources:
            return ""

        seen: set[tuple[str, str, str]] = set()
        lines: list[str] = []
        for source in sources:
            key = (source["tier"], source["title"], source["service_name"])
            if key in seen:
                continue
            seen.add(key)

            label = source["label"]
            if source["tier"] == "global" and source["service_name"]:
                label = f"{label} ({source['service_name']})"
            lines.append(f"- {label}: {source['title'] or '(untitled)'}")

        return "Sources:\n" + "\n".join(lines)


    @staticmethod
    def _build_file_note(
        attachment: AttachmentContext,
        answered_from_file: bool
    ) -> str:
        """
        Tell the LLM which file the question is about, or that it is missing.

        The file is named only if it actually contributed context, so an
        answer from the knowledge base is never credited to the file.

        Args:
            attachment (AttachmentContext): resolved files of the question.

            answered_from_file (bool): the file contributed retrieved context.

        Returns:
            file_note (str): note placed before the retrieved context.
        """
        if attachment.missing_file:
            return MISSING_FILE_LLM_NOTE

        if attachment.file_names and answered_from_file:
            return (
                f"The user attached the file(s): {', '.join(attachment.file_names)}. "
                "Answer using the retrieved context from that file.\n"
            )

        return ""


    def _merge_retrieved_context(
        self,
        context: str | dict,
        chat_history_context: str,
        context_retrieved: str,
        attachment: AttachmentContext,
        answered_from_file: bool,
    ) -> str | dict:
        """
        Fold RAG-retrieved context back into `context`, alongside chat
        history and a note about which attached file(s) the question is
        about.

        Args:
            context (str | dict): context associated with the question.

            chat_history_context (str): chat history part of the context, if any.

            context_retrieved (str): context retrieved by RAG.

            attachment (AttachmentContext): resolved files of the question.

            answered_from_file (bool): the file contributed retrieved context.

        Returns:
            str | dict: context updated with the retrieved information.
        """
        file_note: str = self._build_file_note(
            attachment = attachment,
            answered_from_file = answered_from_file
        )

        suffix = (
            ""
            if   (context_retrieved == "")
            else (f"\n{file_note}Context:\n {context_retrieved}")
        )

        if isinstance(context, dict):
            context.update({"context": f"{chat_history_context}{suffix}"})
        else:
            context = f"{chat_history_context}{suffix}"

        return context


    def _retrieve_context(
        self,
        query: str,
        attachment: AttachmentContext,
        user_id: str | None,
        session_id: str | None,
        global_service_names: list[str] | None,
        memory_service_active: bool,
        kb_service_name: str | None
    ) -> tuple[str, list[float], list[dict[str, Any]]]:
        """
        Retrieve context over the active memory tiers.

        With an attached file, the file and the rest of memory are retrieved
        separately and combined by intent (see
        `RAGBase.retrieve_with_attachment()`). Otherwise, a single search runs
        over all active tiers.

        Args:
            query (str): a question.

            attachment (AttachmentContext): resolved files of the question.

            user_id (str | None): current user.

            session_id (str | None): current chat session.

            global_service_names (list[str] | None): memory-capable services.

            memory_service_active (bool): search the user's saved memory.

            kb_service_name (str | None): memory-capable service the query was
                routed to, if any.

        Returns:
            context (str): retrieved context.

            retrieve_score (list[float]): score of each retrieved chunk.

            sources (list[dict[str, Any]]): origin of each chunk.
        """
        if attachment.document_ids:
            return self.rag.retrieve_with_attachment(
                query,
                document_ids = attachment.document_ids,
                user_id = user_id,
                session_id = session_id,
                global_service_names = global_service_names,
                include_user = memory_service_active,
                file_directed = attachment.file_directed,
                kb_service_name = kb_service_name
            )

        return self.rag( # pyright: ignore[reportReturnType]
            query,
            user_id = user_id,
            session_id = session_id,
            global_service_names = global_service_names,
            include_session = True,
            include_user = memory_service_active,
            include_global = True
        )


    def _rewrite_follow_up(
        self,
        query: str,
        history_messages: list[dict[str, str]] | None
    ) -> tuple[str, int, int]:
        """
        Rewrite a follow-up question into a standalone query for retrieval.

        A follow-up such as "what about its committees?" cannot be matched
        against the vector database on its own, so the LLM resolves what it
        refers to from the recent turns. The answer is still generated from the
        original question. Turned off with `RAG_QUERY_REWRITE=false`.

        Args:
            query (str): the question.
            history_messages (list[dict[str, str]] | None): prior user and assistant turns.

        Returns:
            retrieval_query (str): standalone query, or `query` when there is no
                history, rewriting is off, or the rewrite fails.
            input_token (int): input tokens of the rewrite call.
            output_token (int): output tokens of the rewrite call.
        """
        retrieval_query:    str = query
        input_token:        int = 0
        output_token:       int = 0

        is_enabled: bool = getenv("RAG_QUERY_REWRITE", "true").strip().lower() == "true"

        if not history_messages or not is_enabled:
            return (
                retrieval_query,
                input_token,
                output_token
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": QUERY_REWRITE_PROMPT},
            *history_messages[-QUERY_REWRITE_HISTORY_MESSAGES:],
            {"role": "user", "content": f"Latest question: {query}"}
        ]

        # Broad catch: a failed rewrite must never block the answer.
        try:
            (
                rewritten,
                input_token,
                output_token
            ) = self.llm.chat_messages(messages=messages)

            rewritten = rewritten.strip().strip("\"'")

            if rewritten and len(rewritten) <= QUERY_REWRITE_MAX_CHARS:
                retrieval_query = rewritten

        except Exception as exc:
            print(
                set_color(
                    status="warning",
                    information=f"[qa] Query rewrite skipped: {exc}"
                )
            )

        if retrieval_query != query:
            print(
                set_color(
                    status="info",
                    information=f"[qa] Retrieval query rewritten: '{query}' -> '{retrieval_query}'"
                )
            )

        return (
            retrieval_query,
            input_token,
            output_token
        )


    def _retrieve_rag_context(
        self,
        query: str,
        context: str | dict,
        user_id: str | None,
        session_id: str | None,
        global_service_names: list[str] | None,
        memory_service_active: bool,
        attachment: AttachmentContext,
        kb_service_name: str | None,
        retrieval_query: str | None = None
    ) -> tuple[str, str | dict, float | int, str, list[dict[str, Any]], bool]:
        """
        Retrieve RAG context for the query and fold it into `context`.

        Args:
            query (str): the question.

            context (str | dict): context associated with the question.

            user_id (str | None): the current user, used to scope retrieval.

            session_id (str | None): the current session, used to scope retrieval.

            global_service_names (list[str] | None): global services to include in retrieval.

            memory_service_active (bool): whether the user memory tier should be included.

            attachment (AttachmentContext): resolved files of the question, used to scope
                retrieval and to tell the LLM which file the question is about.

            kb_service_name (str | None): memory-capable service the query was
                routed to, if any.

            retrieval_query (str | None, optional): query used to search the vector
                database instead of `query` (see `_rewrite_follow_up()`). Defaults to None.

        Returns:
            user_prompt (str): prompt built from the query and non-history context.

            context (str | dict): context updated with the retrieved information.

            retrieve_score (float | int): score of the context retrieval.

            context_retrieved (str): context retrieved by RAG.

            retrieved_sources (list[dict[str, Any]]): origin of each retrieved chunk.

            answered_from_file (bool): a chunk of the attached file made it into the context.
        """
        chat_history_context, rag_context = self._split_chat_history_context(context=context)

        # If retrieving context, first generate the user prompt given the
        # user prompter format.
        user_prompt = self.llm.user_prompter(
            query,
            context=rag_context
        ) # pyright: ignore[reportCallIssue]

        # Get the retrieved context, scoped to the active memory tiers
        (
            context_retrieved,
            retrieved_context_score,
            retrieved_sources
        ) = self._retrieve_context(
            query = retrieval_query or query,
            attachment = attachment,
            user_id = user_id,
            session_id = session_id,
            global_service_names = global_service_names,
            memory_service_active = memory_service_active,
            kb_service_name = kb_service_name
        )

        answered_from_file = bool(attachment.document_ids) and any(
            source["tier"] == "session" for source in retrieved_sources
        )

        retrieve_score = (
            max(retrieved_context_score)
            if   (retrieved_context_score)
            else (-1)
        )

        context = self._merge_retrieved_context(
            context = context,
            chat_history_context = chat_history_context,
            context_retrieved = context_retrieved,
            attachment = attachment,
            answered_from_file = answered_from_file
        )

        return (
            user_prompt,
            context,
            retrieve_score,
            context_retrieved,
            retrieved_sources,
            answered_from_file
        )


    def __call__(
        self,
        query:                  str,
        # NOTE:
        # for legacy purposes. Change to `dict[int, dict[str, str]]` type when
        # update to handle `int` properly
        services:               dict[str, dict[str, str]],
        context:                str | dict[str, str]        = "",
        is_rag:                 bool                        = False,
        verbose:                bool                        = False,
        user_id:                str | None                  = None,
        session_id:             str | None                  = None,
        global_service_names:   list[str] | None            = None,
        memory_service_active:  bool                        = False,
        has_files:              bool                        = False,
        file_refs:              list[str] | None            = None,
        history_messages:       list[dict[str, str]] | None = None
    ) -> tuple[str, str, datetime, int, int ,float | int]:
        """
        Process each QA.

        Args:
            query (str): a question.

            services (dict[str, dict[str, str]]): 
                all available services for Query Analyser, which then being used to re-structure question
                with its prompting techniques.

            context (str | dict[str, str], optional):
                contex associated with the question. Defaults to "".

            is_rag  (bool, optional): if retrieve context for the question. Defaults to False.

            verbose (bool, optional): debug mode. Default to False.

            user_id (str | None, optional): owner of the session and user memory. Defaults to None.

            session_id (str | None, optional): chat session of the session memory. Defaults to None.

            global_service_names (list[str] | None, optional): 
                services whose global memory is searched. Defaults to None.

            memory_service_active (bool, optional): if the memory service is enabled. Defaults to False.

            has_files (bool, optional): if the user attached files to the question. Defaults to False.

            file_refs (list[str] | None, optional): 
                attached file references in "<8-hex file_id>_<name>" form. Defaults to None.

            history_messages (list[dict[str, str]] | None, optional):
                prior user and assistant turns, sent to the LLM as chat messages. Defaults to None.

        Returns:
            response (str): truncated answer if applicable.

            raw_response (str): original answer from LLM.

            response_timestamp (datetime):
                timestamp when received raw response from LLM. The format is align
                with Pydantic AwareDateTime format.

            input_token (int):
                total amount of input tokens captured by LLM during each
                QA process.

            output_token (int): total amount of output tokens generated by LLM during each QA process.

            retrieve_score (float | int): 
                highest score among the retrieved chunks (if applicable). Defaults to -1 for non-RAG
                services or when nothing is retrieved.
        """
        # Set initial return answers.
        response:           str | None  = None
        raw_response:       str | None  = None
        retrieve_score:     float | int = -1
        # NOTE:
        # After having a chat with some members in the team regarding the active
        # usage of `chess` & `code_generation` services, I came to a solution to
        # purposefully ignore the support for both of them. We'll still use
        # `code_genertion` service (under active development at the moment) while
        # the `chess` service will be completely removed in the future.
        response_timestamp: datetime    = datetime.now(tz=timezone.utc)
        input_token:        int         = 1
        output_token:       int         = 1

        
        # Add default service 0
        self._add_service_0(services=services)

        # cut down to just "name" of the services not its "desc"
        services_name: dict[str, str] = self._get_services_name(services=services)

        # Give the analyser the chat history so follow-up questions are routed
        # by what they refer to.
        routing_chat_history: str = (
            self._split_chat_history_context(context=context)[0]
            if isinstance(context, str)
            else ""
        )

        # Get service option & minimal extra info about selected service through
        # `query_analyser.py`.
        (
            service_option,
            service_info_dict
        ) = self.query_analyser(
            query=query,                                # pyright: ignore[reportCallIssue]
            services=services,                          # pyright: ignore[reportCallIssue]
            chat_history=routing_chat_history,          # pyright: ignore[reportCallIssue]
            file_names=file_refs                        # pyright: ignore[reportCallIssue]
        )

        # Files the question is about: attached this turn, or attached earlier
        # in the session when the question points at "the file".
        attachment_resolver: AttachmentResolver = AttachmentResolver(self.rag.vector_database)
        attachment: AttachmentContext = attachment_resolver(
            query = query,
            file_refs = file_refs,
            has_files = has_files,
            user_id = user_id,
            session_id = session_id
        )

        service_option = QABase._route_session_files(
            service_option=service_option,
            attachment=attachment,
            attachment_resolver=attachment_resolver,
            user_id=user_id,
            session_id=session_id
        )

        # Check if force route is required.
        service_option = self._force_route_check(
            service_option = service_option,
            has_attachment = has_files or bool(attachment.document_ids)
        )

        if service_option == SESSION_FILE_LIST_ROUTE:
            response        = session_file_list_answer(file_names=attachment.session_file_names)
            raw_response    = response
            input_token     = 0
            output_token    = 0

        # TODO: Remove chess service
        elif service_option.startswith("1."):
            (
                response,
                raw_response,
                response_timestamp,
                input_token,
                output_token
            ) = self._handle_chess_service(
                service_option = service_option,
                service_info_dict = service_info_dict,
                context = context
            )

        elif service_option == "2":
            (
                response,
                raw_response
            ) = self._handle_memory_update_service(
                query = query,
                services = services,
                services_name = services_name,
                memory_service_active = memory_service_active,
                user_id = user_id
            )

        elif service_option == "3":
            (
                response,
                raw_response
            )= self.code_generator(query)

        # System information service 
        elif service_option == "0":
            (
                response,
                raw_response,
                response_timestamp,
                input_token,
                output_token
            ) = self.system_information_service(
                query = query,
                services = services,
                context = context
            )

        # When all services are disabled and service 0 cannot answer, query analyser
        # will return -1.
        elif service_option == "-1":
            response, raw_response = self.fallback_service(services = services)

        else:
            context_retrieved:  str                     = ""
            retrieved_sources:  list[dict[str, Any]]    = []
            answered_from_file: bool                    = False

            if self._use_rag(
                services = services,
                service_option = service_option,
                is_rag = is_rag,
                has_files = has_files,
                attachment = attachment
            ):
                # A memory-capable service owns a knowledge base worth consulting.
                kb_service_name = (
                    services_name.get(service_option)
                    if   (self._is_rag_eligible(services, service_option))
                    else (None)
                )

                # Search with a standalone version of a follow-up question.
                (
                    retrieval_query,
                    rewrite_input_token,
                    rewrite_output_token
                ) = self._rewrite_follow_up(
                    query=query,
                    history_messages=history_messages
                )

                service_info_dict["input_token"]    += rewrite_input_token
                service_info_dict["output_token"]   += rewrite_output_token

                (
                    user_prompt,
                    context,
                    retrieve_score,
                    context_retrieved,
                    retrieved_sources,
                    answered_from_file
                ) = self._retrieve_rag_context(
                    retrieval_query = retrieval_query,
                    query = query,
                    context = context,
                    user_id = user_id,
                    session_id = session_id,
                    global_service_names = global_service_names,
                    memory_service_active = memory_service_active,
                    attachment = attachment,
                    kb_service_name = kb_service_name
                )

            # Non-RAG services (or when RAG is disabled) fall through here.
            else:
                user_prompt = query
                retrieve_score  = -1

            # NOTE:
            # For most of the time when I turn on the debug to test the
            # functionality of CoSMIC, it always found `Ollama` class rather than
            # the base class. As we do support multi AI providers but the linter
            # (e.g., `basedpyright`) isn't happy about the usecase of this one
            # (error-level raised here), I've to temporary disable the notice on
            # purpose until we re-implement/modify the base class structure to
            # properly support this capability again.
            #
            # Get the response with retrieved context if applicable.
            (
                response,
                raw_response,
                response_timestamp,
                input_token,
                output_token
            ) = self.llm( # pyright: ignore[reportAssignmentType]
                question=user_prompt,
                # Prior turns go as chat messages here, so drop their text copy.
                context=(
                    QABase._strip_chat_history(context=context)
                    if history_messages
                    else context
                ),
                service_name=self._select_persona(
                    service_option,
                    routed_service_name=services_name.get(service_option, ""),
                    answered_from_file=answered_from_file
                ),
                history_messages=history_messages
            )

            # Acknowledge an attached file the question was not about, even when
            # nothing was retrieved for the answer instead.
            if attachment.file_names and not attachment.file_directed and not answered_from_file:
                note = unused_file_note(
                    file_names=attachment.file_names,
                    from_knowledge_base=bool(context_retrieved)
                )
                response = f"{response}\n\n{note}"

            if attachment.missing_file:
                response = f"{response}\n\n{MISSING_FILE_USER_NOTE}"

            if retrieved_sources:
                response = f"{response}\n\n{self._format_sources(retrieved_sources)}"

        # NOTE:
        # Last step to transfer final response with some extra info (I/O
        # token, retrieve score, etc) over to opensi_cosmic.py. This's the
        # 2nd time QA sent user query + selected service prompt (from
        # query_analyser.py). The latter is different based on which condition
        # above get hit

        # Sum up I/O tokens from 1st Ollama call (in `service_info_dict`) with
        # I/O tokens from 2nd Ollama call (in either condition above).
        total_input_token:  int = service_info_dict["input_token"] + input_token
        total_output_token: int = service_info_dict["output_token"] + output_token

        return (
            str(response),
            str(raw_response),
            response_timestamp,
            total_input_token,
            total_output_token,
            retrieve_score
        )
