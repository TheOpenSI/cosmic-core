### Core modules ###
from datetime import datetime
from sys import exit
from pathlib import Path
from re import search


### Type hints ###


### Internal modules ###
from ..services.llms import llm as llm_instances
from ..query_analyser import user_prompt as query_user_prompt_instances
from ..maps import LLM_INSTANCE_DICT
from ...utils.log_tool import set_color
from ...utils.module import get_instance



class QueryAnalyser:
    def __init__(
        self,
        llm_name:       str     = "qwen2.5:7b",
        seed:           int     = 0,
        is_quantised:   bool    = False,
        device:         str     = "cuda"
    ) -> None:
        """
        Query analyser to select a service.

        Args:
            llm_name (str, optional):
                LLM name for analyser. Defaults to "qwen2.5:7b".

            seed (int, optional):
                response generation seed. Defaults to 0.

            is_quantised (bool, optional):
                use quantized LLM. Defaults to False.

            device (str, optional):
                use cuda or cpu for LLM. Defaults to "cuda".
        """
        # Set config.
        self.root = Path(__file__).resolve(strict=True).parent.parent.parent
        self.device = device

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

        # Build LLM instance from class defined in .py
        self.llm = get_instance(
            instances=llm_instances,
            instance_name=llm_instance_name
        )(
            llm_name=llm_name,
            seed=seed,
            is_quantised=is_quantised,
            use_example=False,
            is_truncate_response=True,
            device=device
        )

        return None


    def __call__(
        self,
        query:      str,
        # NOTE:
        # for legacy purposes. Change to `dict[int, dict[str, str]]` type when
        # update to handle `int` properly
        services:   dict[str, dict[str, str]],
        verbose:    bool = False
    ) -> tuple[str, dict[str, str | datetime | int | bool | None]]:
        """
        Analyse query to get service option.

        Args:
            query (str):
                question.

            services (dict[str, dict[str, str]]):
                all available services for Query Analyser, which then being used
                to re-structure question with its prompting techniques.

            verbose (bool, optional):
                debug mode. Default to False.

        Returns:
            service_option (str):
                service option.

            service_info_dict (dict[str, str | int | bool | None]):
                updated information dictionary.
        """
        # Set a list of services.
        self.services: dict[str, str] = {
            service_id: service_info['desc']
            for (service_id, service_info) in services.items()
        }

        # # There is/are active services from fetched API endpoint
        # if len(self.services) != 0:
        #     pass

        # # No active services found from fetched API endpoint
        # else:
        #     raise HTTPException(
        #         status_code=status.HTTP_404_NOT_FOUND,
        #         detail={
        #             "status": "404 - Not Found",
        #             "message": "{trig:s}: {cond:s}".format(
        #                 trig="EmptyServiceError",
        #                 cond="No active services found from fetched API endpoint. At least 1 service is required to for Query Analyser."
        #             )
        #         }
        #     )

        # Chect if chess is one of the services, if yes, then add chess subservices to full services.
        self.full_services: dict[str, str] = dict(self.services)

        if "1" in self.services:
            self.chess_subservices: dict[str, str] = {
                "1.0": "predict next move given a chess FEN",
                "1.1": "predict next move given a sequence of moves"
            }

            self.full_services.update(self.chess_subservices)


        # print(
        #     set_color(
        #         status="info",
        #         information=f"Full services from Query Analyser: {self.full_services}"
        #     )
        # )

        # Get the number of services.
        self.num_services: int = len(self.services)

        # Set user prompter for service option.
        self.user_prompter_service = get_instance(
            instances=query_user_prompt_instances,
            instance_name="QueryAnalyserService"
        )(
            services=self.services
        )

        # Create an initial information dictionary.
        service_info_dict: dict[str, str | int | bool | None] = {
            "query":        query,
            "input_token":  0,
            "output_token": 0
        }

        # Set the user prompter for service option.
        self.llm.set_user_prompter(self.user_prompter_service)

        # NOTE:
        # This's the 1st time QA sent user query + its combined prompt
        # (user & system prompting). We only use Ollama models at the moment so
        # it's safe to assume that `self.llm` is calling methods from `Ollama.py`.
        (
            service_analysis,
            _service_raw_analysis,
            service_response_timestamp,
            service_input_token,
            service_output_token
        ) = self.llm(query)

        # Get the service option.
        service_option: str = self.mapping(response=service_analysis)

        # print(
        #     set_color(
        #         status="info",
        #         information=f"Selected service from SLM response (user query has been re-prompted by Query Analyser): {service_option}"
        #     )
        # )

        # Analysis information.
        if verbose:
            print(
                set_color(
                    status="info",
                    information="{0:s}\n{1:s}\n{2:s}".format(
                        f"Query: {query}",
                        f"Analysis: {service_analysis}",
                        f"Service: {service_option}"
                    )
                )
            )

        if service_option == "1":
            # Remove last symbol.
            if query[-1] in [
                ",",
                ".",
                "!",
                "?"
            ]:
                query = query[:-1]

            # Predict the next move in chess game.
            (
                service_option,
                service_info_dict
            ) = self.chess_parse(
                query=query,
                service_info_dict=service_info_dict
            )

        elif service_option == "2":
            # Update the vector database.
            (
                service_option,
                service_extra_info
            ) = self.update_vector_database_parse(query=query)

            service_info_dict.update(service_extra_info)

        service_info_dict["response_timestamp"] = service_response_timestamp
        service_info_dict["input_token"]        = service_input_token
        service_info_dict["output_token"]       = service_output_token

        return (
            service_option,
            service_info_dict
        )


    def quit(self):
        """
        Quit by releasing model memory and instance.
        """
        # Release memory of LLM.
        if self.llm:
            self.llm.quit()


    def mapping(
        self,
        response: str
    ):
        """
        Parse response to get service option.

        Args:
            response (str): response from LLM analysis.

        Returns:
            option (int): index of the service option if avaiable; otherwise, None.
        """
        # Set to lower case.
        response = response.lower()

        # Truncate to get the option index.
        option = search(
            pattern=r"service (\d{1,3}\.\d{1,3}|\d{1,3})",
            string=response,
            flags=0
        )

        if option:
            option = option.group(1)

            if option not in self.full_services.keys():
                print(
                    set_color(
                        status="error",
                        information=f"Unknown service '{option}' from '{response}'."
                    )
                )
                return "-1"

            else:
                return option

        else:
            return "-1"


    def get_service(
        self,
        index: int
    ):
        """
        Get the description of the service.

        Args:
            index (int): index of the service.

        Returns:
            service (str): description of the service.
        """
        # Option not found.
        if index not in self.full_services.keys():
            return None

        return self.services[str(object=index)]


    def chess_parse(
        self,
        query: str,
        service_info_dict: dict
    ):
        """
        Parse query to get chess service.

        Args:
            query               (str):  question.
            service_info_dict   (dict): dictionary to contain parsed information.

        Returns:
            service_option      (str):  service option.
            service_info_dict   (dict): updated information dictionary.
        """
        # Default service option.
        service_option = "-1"

        # Parse move string.
        move_match = search(
            pattern="[\[,\:](.*?[,\s].*?)[\.,\]]?$",
            string=query,
            flags=0
        )

        # Parse FEN string.
        fen_match = search(
            pattern="{0:s}{1:s}".format(
                "(((?:[rnbqkpRNBQKP1-8]+\/){7})[rnbqkpRNBQKP1-8]+)",
                "\s([b|w])\s(-|[K|Q|k|q]{1,4})\s(-|[a-h][1-8])\s(\d+\s\d+)$"
            ),
            string=query,
            flags=0
        )

        if fen_match:
            # Given FEN.
            current_fen = fen_match.group()
            service_option = "1.0"
            service_info_dict.update({"fen": current_fen})

        elif move_match:
            # Given a sequence of moves.
            current_moves = move_match.group(1)
            service_option = "1.1"
            service_info_dict.update({"moves": current_moves})

        else:
            # Invalid inputs.
            if query.find("predict") > -1 or query.find("next move") > -1:
                print(
                    set_color(
                        status="hint",
                        information="For chess move prediction, index a sequence of moves or FEN with \":\"."
                    )
                )

        return (service_option, service_info_dict)


    def update_vector_database_parse(
        self,
        query: str,
    ) -> tuple[str, dict[str, bool | str | None]]:
        """
        Parse query to get text or document path to update vector database.

        Args:
            query (str):
                question.

        Returns:
            service_option (str):
                service option.

            service_extra_info (dict[str, bool | str | None]):
                extra information needed to current information dictionary.
        """
        service_option: str = "2"

        # Memory-specific information dictionary
        service_extra_info: dict[str, bool | str | None] = {
            "text": None,
            "document_path": None
        }

        # Check if context is a .pdf.
        if query.find(".pdf") > -1:
            service_extra_info.update({"is_a_document": True})

            # Parse move string
            document_path = search(
                pattern=r"(?<=\:\s)(.*?)+\.pdf",
                string=query,
                flags=0
            )

            if document_path:
                document_path = document_path.group()

            else:
                print(
                    set_color(
                        status="warning",
                        information="Invalid document query [tip: pdf file(s) is required.]"
                    )
                )

                return (
                    service_option,
                    service_extra_info
                )

            # Check if not an absolute path, convert to an absoluate path.
            if not Path(document_path).resolve(strict=True).is_absolute():
                document_path = str(self.root.joinpath(document_path))

            # Update our extra information dictionary
            service_extra_info["document_path"] = document_path

        else:
            service_extra_info.update({"is_a_document": False})

            # Extract text.
            text = search(
                pattern=r"\:((\"|\')?(.*?)[\",\']?$)",
                string=query,
                flags=0
            )

            if text:
                text = text.group(0)
                text = text.replace(": ", "").replace(":", "")

                # Update our extra information dictionary
                service_extra_info["text"] = text

            else:
                print(
                    set_color(
                        status="warning",
                        information="Invalid text query [tip: index the text with :]"
                    )
                )

                return (
                    service_option,
                    service_extra_info
                )

        return (
            service_option,
            service_extra_info
        )