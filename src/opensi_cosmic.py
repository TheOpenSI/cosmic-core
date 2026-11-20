### Core modules ###
from datetime import (
    datetime,
    timezone
)
import os
from pathlib import Path
from dotenv import load_dotenv
from sys import (
    exit,
    stdout
)
from fastapi import (
    HTTPException,
    status
)
from torch import cuda
from httpx import (
    Client,
    ConnectError,
    ConnectTimeout,
    Response
)
from pprint import pp


### Type hints ###
from typing import Any


### Internal modules ###
from .services.llms import llm as llm_instances
from .maps import LLM_INSTANCE_DICT
from .services.vector_database import VectorDatabase
from .services.qa import QABase
from .services.rag import RAGBase
from .services.system_information_service import SystemInformationService
from .services.fallback_service import FallbackService
from ..modules.chess.chess_qa_puzzle import PuzzleAnalyse
from ..modules.chess.chess_qa_quality import QualityEval
from ..modules.chess.chess_genfen import FENGenerator
from ..modules.chess.chess_gencot import CotGenerator
from ..modules.code_generation.code_generation import CodeGenerator
from ..utils.log_tool import set_color
from ..utils.module import get_instance
from .query_analyser.query_analyser import QueryAnalyser


# Load the project-root .env into os.environ so services can read their config.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


# Shared cache for the services registry
# Every service-derived view is built from this cached raw list
_services_cache: list[dict[str, Any]] = []
_services_cache_timestamp: float = 0
_SERVICES_CACHE_TTL = 30  # seconds


class OpenSICoSMIC:
    def __init__(
        self,
        general_slm:    str         = "",
        qa_slm:         str         = "",
        user:           dict | None = None
    ) -> None:
        """
        Construct OpenSICoSMIC instance. It contains SLMs and services including
        vector database and RAG, where RAG includes context retriever and vector
        database update. Chess services are included in `PuzzleAnalyse()` and
        `QualityEval()` classes (called on demand, not as global instance).

        Args:
            general_slm (str):
                General SLM name used in every chat session (check default list
                of SLMs in `src/maps.py`). If empty, the entry is a combination
                format with value from `provider` & `model` fields (accessible
                through `self.general_config_data`).

            qa_slm (str):
                Query Analyser SLM name (check default list of SLMs in
                `src/maps.py`). If empty, the entry is a combination format with
                value from `provider` & `model` fields (accessible through
                `self.qa_config_data`).

            user (dict, optional):
                user information including ID, name, etc. Default to None.
        """
        # NOTE:
        # Because of how db table works, we could add multiple preset of CoSMIC
        # default configs. While there're no usecase for this feature yet (there
        # aren't any extra ones either beside the only inserted default configs
        # data through Alembic script), it's good to mentioned here so that
        # future devs can work on this once CoSMIC is getting more complexed and
        # in need of this feature. For now, we'll assume to use the first result
        # only.

        # For-loop here would be too expensive so I used this trick instead.
        # Inspired from:
        # https://stackoverflow.com/questions/61105986/how-to-access-elements-in-a-dict-values
        self.config_data:           dict[str, Any] = list((self.get_configs()[0]).values())[0]
        self.general_config_data:   dict[str, Any] = self.config_data["general"]
        self.qa_config_data:        dict[str, Any] = self.config_data["query_analyser"]

        # print(
        #     set_color(
        #         status="info",
        #         information=f"Default General configs: {self.general_config_data}"
        #     )
        # )
        # print(
        #     set_color(
        #         status="info",
        #         information=f"Default QA configs: {self.qa_config_data}"
        #     )
        # )


        # Set user info
        self.user_id = (
            str(user["id"])
            if (
                (user is not None)  and
                ("id" in user)      and
                (user["id"] != "")
            )
            else (None)
        )


        # Set model device.
        self.device: str = (
            "cuda"
            if   (cuda.is_available())
            else ("cpu")
        )


        # Check OpenAI API key.
        self.openai_api_status: str | None = self.check_openai_key()


        # Read SLMs from Configs API endpoint if not specified
        self.general_slm = general_slm

        if self.general_slm == "":
            # Match model detection format in `src/services/llms/LLMBase.py`.
            # For example: "ollama:qwen2.5:7b"
            self.general_slm = f"{self.general_config_data["provider"]}:{self.general_config_data["model"]}"

        # print(
        #     set_color(
        #         status="info",
        #         information=f"Set General SLM: [{self.general_slm}]"
        #     )
        # )

        self.llm = self.get_llm(
            llm_name=self.general_slm,
            seed=self.general_config_data["seed"],
            is_quantised=self.general_config_data["is_quantised"],
            device=self.device
        )


        # Read SLMs from Configs API endpoint if not specified
        self.qa_slm = qa_slm

        if self.qa_slm == "":
            # Match model detection format in `src/services/llms/LLMBase.py`.
            # For example: "ollama:qwen2.5:7b"
            self.qa_slm = f"{self.qa_config_data["provider"]}:{self.qa_config_data["model"]}"

        # print(
        #     set_color(
        #         status="info",
        #         information=f"Set QA SLM: [{self.qa_slm}]"
        #     )
        # )

        self.query_analyser: QueryAnalyser = QueryAnalyser(
            llm_name=self.qa_slm,
            seed=self.qa_config_data["seed"],
            is_quantised=self.qa_config_data["is_quantised"],
            device=self.device
        )


        # Code generation service.
        self.code_generator: CodeGenerator = CodeGenerator()

        # System information service.
        self.system_information_service: SystemInformationService = SystemInformationService(llm=self.llm)

        # Fallback service
        self.fallback_service: FallbackService = FallbackService()

        # Initialise & setup QA instance.
        self.qa: QABase | None = None

        self.set_up_qa(
            user_id=str(self.user_id),
            user_name=None
        )

        return None


    def __call__(
        self,
        question:   str,
        context:    str         = "",
        log_file:   str | None  = None,
        session_id: str | None  = None,
        has_files:  bool        = False,
        user_id:    str | None  = None,
        file_refs:  list | None = None,
    ) -> tuple[str, str, datetime, int, int, float | int]:
        """
        Execute QA.

        Args:
            question (str):
                a question or a .csv containing multiple questions.

            context (str, optional):
                context for this question. Defaults to "".

            log_file (str, optional):
                whether to print the result in a .txt file. Defaults to None.

        Returns:
            response (str):
                (truncated) response.

            raw_response (str):
                raw response from LLM without truncations.

            response_timestamp (datetime):
                timestamp when received raw response from LLM. The format is align
                with Pydantic AwareDateTime format.

            input_token (int):
                amount of input tokens captured from each QA session.

            output_token (int):
                amount of output tokens generated from each QA session.

            retrieve_score (float | int):
                score of context retrieving (if applicable). Default to '-1' for
                non-RAG services.
        """
        # print(
        #     set_color(
        #         status="info",
        #         information=f"General configs used when query received: {self.general_config_data}"
        #     )
        # )
        # print(
        #     set_color(
        #         status="info",
        #         information=f"QA configs used when query received: {self.qa_config_data}"
        #     )
        # )
        # print(
        #     set_color(
        #         status="info",
        #         information=f"General SLM used when query received: [{self.general_slm}]"
        #     )
        # )
        # print(
        #     set_color(
        #         status="info",
        #         information=f"QA SLM used when query received: [{self.qa_slm}]"
        #     )
        # )

        # Set initial output to return.
        response:           str | None  = None
        raw_response:       str | None  = None
        response_timestamp: datetime    = datetime.now(tz=timezone.utc)
        input_token:        int         = 1
        output_token:       int         = 1
        retrieve_score:     float | int = -1


        # Check if OpenAI API key is valid.
        if self.openai_api_status != "":
            return (
                str(response),
                str(raw_response),
                response_timestamp,
                input_token,
                output_token,
                retrieve_score
            )

        # Chat-mode LLM do not need example in the system prompt.
        if self.llm.llm_name in [
            "mistral-7b-instruct-v0.1",
            "gemma-7b-it"
        ]:
            use_example = False

        else:
            use_example = True

        # Check if the question is a string or a .csv file containing multiple sub-questions.
        if question.find(".csv") > -1:
            # Batch process for puzzles (move prediction and analysis) and 4 other quality evaluations.
            if question.find("puzzle") > -1:
                # Do not truncate from the LLM base class but instead using
                # PuzzleAnalyse.is_truncate_response=True.
                self.llm.set_truncate_response(False)

                # Truncation external to the system_prompter does not require keywords, no example
                # in the system prompt is required.
                self.llm.system_prompter.set_use_example(False)

                # Build PuzzleAnalyse QA service.
                puzzle_analyser = PuzzleAnalyse(
                    llm=self.llm,
                    rag=self.rag,
                    is_rag=False,
                    log_file=log_file,
                    is_truncate_response=True,
                    next_move_predict_backend="stockfish",
                )

                # Batch process the question file.
                average_score = puzzle_analyser.batch_process(question)

                # Return how many questions get correct best move prediction
                # using Stockfish or GPT API.
                response = f"Success rate of {question} is {average_score:.2f}."

            elif (
                (question.find("attention") > -1) or
                (question.find("memory") > -1)    or
                (question.find("perception") > -1)
            ):
                # Manually switch on and off RAG for specific questions.
                if (
                    (question.find("attention") > -1) or
                    (question.find("memory_update") > -1)
                ):
                    is_rag = True

                else:
                    is_rag = False

                # Truncate the response according to the system_prompt format.
                self.llm.set_truncate_response(True)

                # Use example's keywords for truncation for non-chat LLM.
                self.llm.system_prompter.set_use_example(use_example)

                # Build quality evaluation service.
                quality_evaluator = QualityEval(
                    llm=self.llm,
                    rag=self.rag,
                    is_rag=is_rag,
                    log_file=log_file
                )

                # Batch process the question file.
                average_score = quality_evaluator.batch_process(question)

                # Return how many questions get correct answer from LLM compared to GT answer.
                response = f"Success rate of {question} is {average_score:.2f}."

            elif question.find("checkmate_moves") > -1:
                # Generate FEN with moves
                fen_generator = FENGenerator(log_file=log_file)

                # Batch process the question file.
                fen_generator.batch_process(question)

            elif question.find("finetune_dataset") > -1:
                # Generate CoT analysis for finetune dataset.
                cot_generator = CotGenerator(
                    log_file=log_file,
                    is_truncate_response=True
                )

                # Batch process the question file.
                cot_generator.batch_process(question)

        else:
            # General question needs truncation according the system prompt to avoid hallucination.
            self.llm.set_truncate_response(True)

            # Truncation needs keywords from the example of system prompt.
            self.llm.system_prompter.set_use_example(True)

            # Resolve which memory scopes are active for retrieval
            _services = self.get_services(raw=True)
            global_service_names = [
                s["name"] for s in _services
                if s.get("status") is True
                and s.get("memory_capability") is True
                and (s.get("name") or "").lower() != "memory"
            ]
            memory_service_active = any(
                (s.get("name") or "").lower() == "memory" and s.get("status") is True
                for s in _services
            )

            # Per-request user (from the chat call) for retrieval scoping
            call_user_id = user_id if user_id is not None else self.user_id

            # NOTE:
            # User query received. Therefore, we can start processing each of
            # them against our QA.
            (
                response,
                raw_response,
                response_timestamp,
                input_token,
                output_token,
                retrieve_score
            ) = self.qa( # pyright: ignore[reportOptionalCall]
                query=question,
                services=self.get_services(), # pyright: ignore[reportArgumentType]
                context=context, # Chat history context (see `backend/routers/cosmic.py`)
                is_rag=True,
                verbose=False,
                user_id=call_user_id,
                session_id=session_id,
                global_service_names=global_service_names,
                memory_service_active=memory_service_active,
                has_files=has_files,
                file_refs=file_refs, # pyright: ignore[reportUnknownArgumentType]
            )

        # NOTE:
        # Each user query has been processed successfully. We can return the
        # final output to our endpont call now.
        return (
            str(response),
            str(raw_response),
            response_timestamp,
            input_token,
            output_token,
            retrieve_score
        )


    def set_up_qa(
        self,
        user_id: str,
        user_name: str | None = None
    ):
        """
        Set up QA instance by user ID.

        Args:
            user_id     (str):              user ID through front-end.
            user_name   (str, optional):    user name. Defaults to None.
        """
        # Invalid user ID.
        if (
            (user_id == "") and
            (not isinstance(user_id, str))
        ):
            self.user_id = None

        # For a default user (no user ID), always use the same QA instance.
        if (
            (user_id is None) and
            (self.qa is not None)
        ):
            return -1

        if (
            (user_id != self.user_id) or
            (self.qa is None)
        ):
            # Change the global user ID.
            self.user_id = user_id

            # Create vector database service, storage is backed by Qdrant (reached via QDRANT_URL)
            # The VectorDatabase service handles the connection, so there is no local on-disk path
            # Embedding, deduplication and reranker settings are read from the
            # environment inside VectorDatabase.
            vector_database = VectorDatabase(device=self.device)

            # RAG tuning is read from the environment inside RAGBase.
            self.rag = RAGBase(vector_database=vector_database)

            # QA module to handle basic types of questions, such __next__move__, __update__store__, and
            # general questions.
            self.qa = QABase(
                query_analyser=self.query_analyser, # pyright: ignore
                llm=self.llm,
                rag=self.rag,
                code_generator=self.code_generator,
                system_information_service = self.system_information_service,
                fallback_service=self.fallback_service,
                config=None
            )


    def check_openai_key(self):
        """
        Check OpenAI API key valid.

        Returns:
            answer: status information.
        """
        llm_name                = f"{self.general_config_data["provider"]}:{self.general_config_data["model"]}"
        query_analyser_llm_name = f"{self.qa_config_data["provider"]}:{self.qa_config_data["model"]}"

        is_llm_name_gpt                 = llm_name.find("gpt") > -1
        is_query_analyser_llm_name_gpt  = query_analyser_llm_name.find("gpt") > -1

        llm_name_list = []

        if is_llm_name_gpt:
            llm_name_list.append(llm_name)

        if (
            (is_query_analyser_llm_name_gpt) and
            (query_analyser_llm_name not in llm_name_list)
        ):
            llm_name_list.append(query_analyser_llm_name)

        count = len(llm_name_list)

        openai_api_key: str | None = self.general_config_data["api_key"]

        if  (count > 0) \
        and (openai_api_key == ""):
            answer = ""

            if count == 1:
                answer = f"Since '{llm_name_list[0]} is' used, please add valid OPENAI_API_KEY in .env."

            elif count == 2:
                answer = f"Since '{llm_name_list[0]} and {llm_name_list[1]} are' used, please add valid OPENAI_API_KEY in .env."

        else:
            answer = ""

        return answer


    def get_llm(
        self,
        llm_name: str,
        seed: int = 0,
        is_quantised: bool = False,
        **kwargs
    ):
        """
        Construct LLM give an LLM name.

        Args:
            llm_name        (str):
                LLM name, check LLM_MODEL_DICT in src/maps.py.

            seed            (int):
                LLM content generation seed. Default to 0.

            is_quantised    (bool):
                use quantised LLM. Default to False.

        Return:
            llm (LLMBase): LLM instance.
        """
        # Build LLM instance from class defined in .py if llm_name is supported.
        if llm_name in LLM_INSTANCE_DICT.keys():
            llm_instance_name = LLM_INSTANCE_DICT[llm_name]

        elif llm_name.find("gpt") > -1:
            llm_instance_name = "GPT"

        elif llm_name.find("ollama") > -1:
            llm_instance_name = "Ollama"

        else:
            print(
                set_color(
                    status="error",
                    information=f"Unsupported LLM: {llm_name}."
                )
            )
            exit(1)

        llm = get_instance(
            instances=llm_instances,
            instance_name=llm_instance_name
        )(
            llm_name=llm_name,
            seed=seed,
            is_quantised=is_quantised,
            **kwargs
        )

        print(
            set_color(
                status="info",
                information=f"LLM instance created: {llm}"
            )
        )

        return llm


    def quit(self):
        """
        Release memory of LLM and vector embedding model in vector_database.
        """
        self.query_analyser.quit()
        self.llm.quit()
        self.rag.vector_database.quit()


    def get_services(
        self,
        # TODO:
        # create a dedicated util to handle valid URL format
        url:        str | None              = None,
        params:     dict[str, bool] | None  = {"active": True},
        lifetime:   float                   = 10.0,
        verbose:    bool                    = False,
        raw:        bool                    = False,
    # NOTE:
    # for legacy purposes. Change to `dict[int, dict[str, str]]` type when update
    # to handle `int` properly
    ) -> dict[str, dict[str, str]] | list[dict[str, Any]]:
        """
        THE single entry point for the services registry.

        One HTTP fetch + one module-level 30s cache; every service-derived view
        is built from this method. On fetch failure the stale cache is returned
        if present, otherwise empty — no default/invented services.

        Args:
            url:
                base URL of the services API endpoint. Defaults to the
                SERVICES_API_URL env var or "http://backend:8000/api/v1/services/".

            params:
                optional query parameter for provided endpoint. Defaults to
                {"active": True} to get active only services.

            lifetime:
                HTTP client timeout in seconds. Defaults to 10.0 seconds.

            verbose:
                Enable pretty-printed debug output of service data.

            raw:
                When True, return the cached raw records
                [{id, name, desc, status, memory_capability}, ...].
                When False (default), return the query-analyser shape with
                0-based string indices: {str(id-1): {"name", "desc"}}.

        Example:
            >>> obj.get_services()["0"]           # analyser shape (1st service)
            {"name": "chess", "desc": "<a very long description>"}
            >>> obj.get_services(raw=True)[0]     # raw registry record
            {"id": 1, "name": "chess", "desc": "...", "status": True, ...}
        """
        global _services_cache, _services_cache_timestamp
        from time import time

        if url is None:
            url = os.getenv("SERVICES_API_URL", "http://backend:8000/api/v1/services/")

        current_time = time()

        # Refresh the shared cache when stale.
        if not (_services_cache and (current_time - _services_cache_timestamp) < _SERVICES_CACHE_TTL):
            try:
                with Client(
                    base_url=url,
                    params=params,
                    timeout=lifetime
                ) as client:
                    response: Response = client.get(url="")
                    response.raise_for_status()
                    _services_cache = response.json().get("result", [])
                    _services_cache_timestamp = current_time
            except Exception as httpx_err:
                print(
                    set_color(
                        status="warning",
                        information=f"Services registry unreachable ({httpx_err}); "
                                    f"using {'stale cache' if _services_cache else 'no services'}."
                    )
                )

        datas: list[dict[str, Any]] = list(_services_cache)

        if verbose:
            print(
                "{head_sep:s}\n{body_msg:s}\n{foot_sep:s}".format(
                    head_sep=f"{'=' * 80}",
                    body_msg="[DEBUG]   SERVICES DATA   [DEBUG]",
                    foot_sep=f"{'=' * 80}"
                )
            )
            pp(
                object=datas,
                stream=stdout,
                indent=4 # Prefer tab over spaces indentation
            )
            print(f"{'=' * 80}")

        if raw:
            return datas

        # Query-analyser shape, derived from the same cached raw list.
        # NOTE:
        # for legacy purposes. Change to normal when update the checking
        # logic to handle `int` properly
        services: dict[str, dict[str, str]] = {}
        for data in datas:
            services.setdefault(
                str(data["id"]),
                {}
            ).update(
                {
                    "name": data["name"],
                    "desc": data["desc"],
                    "memory_capability": data.get("memory_capability", False),
                }
            )

        return dict(sorted(services.items()))


    def get_configs(
        self,
        # TODO:
        # create a dedicated util to handle valid URL format
        url:        str                     = "http://backend:8000/api/v1/configs/",
        params:     dict[str, bool] | None  = None,
        lifetime:   float                   = 10.0,
        verbose:    bool                    = False
    ) -> list[dict[str, dict[str, Any]]]:
        """
        Retrieve presets of configuration from API endpoint.

        This method fetches presets of configuration data from the specified
        endpoint and returns the exact data structure received for further
        CoSMIC usages.

        Args:
            url:
                base URL of the configurations API endpoint. Defaults to
                "http://backend:8000/api/v1/configs/".

            params:
                optional query parameter for provided endpoint. Defaults to None.

            lifetime:
                HTTP client timeout in seconds. Defaults to 10.0 seconds.

            verbose:
                Enable pretty-printed debug output for presets of configuration
                data. When True, prints formatted presets of configuration data
                using 'pprint'.

        Returns:
            Presets of configuration data.

            Example:
            [
                { <first configuration preset> },
                { <second configuration preset> }
            ]

        Raises:
            HTTPException:
                With status code 500 if any connection error occurs
                (ConnectError, ConnectTimeout) or other unexpected exceptions.

        Example:
            >>> configs = obj.get_configs(verbose=True)
            >>> configs[0] # 1st configuration preset
            [{"<configuration preset name>": "<default configurations>"}]
        """
        configs: list[dict[str, dict[str, Any]]] = []

        try:
            with Client(
                base_url=url,
                params=params,
                timeout=lifetime
            ) as client:
                response: Response = client.get(url="")
                response.raise_for_status()
                datas: list[dict[str, Any]] = response.json().get("result", [])

            if len(datas) == 0:
                # No config presets available
                if verbose:
                    print(
                        "{head_sep:s}\n{body_msg:s}\n{foot_sep:s}".format(
                            head_sep=f"{'=' * 80}",
                            body_msg="[DEBUG]   CONFIGURATIONS DATA   [DEBUG]",
                            foot_sep=f"{'=' * 80}"
                        )
                    )
                    print(
                        "{debug_msg:s}\n{foot_sep:s}".format(
                            debug_msg="No configuration presets available...",
                            foot_sep=f"{'=' * 80}"
                        )
                    )
                    return []

                else:
                    return []

            else:
                # There is/are config preset(s) available
                for data in datas:
                    # We only need `name` & `details` field to form minimal
                    # config presets data
                    configs.append({data["name"]: data["details"]})

                if verbose:
                    print(
                        "{head_sep:s}\n{body_msg:s}\n{foot_sep:s}".format(
                            head_sep=f"{'=' * 80}",
                            body_msg="[DEBUG]   CONFIGURATIONS DATA   [DEBUG]",
                            foot_sep=f"{'=' * 80}"
                        )
                    )
                    pp(
                        object=configs,
                        stream=stdout,
                        indent=4 # Prefer tab over spaces indentation
                    )
                    print(f"{'=' * 80}")
                    return configs

                else:
                    return configs


        except ConnectError as httpx_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{httpx_err}"
            )


        except ConnectTimeout as httpx_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{httpx_err}"
            )