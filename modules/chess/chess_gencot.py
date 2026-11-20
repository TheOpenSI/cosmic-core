### Core modules ###
from sys import exit
from pandas import read_csv


### Type hints ###


### Internal modules ###
from ...utils.log_tool import set_color
from ...src.services.llms.prompts import user_prompt as user_prompt_instances
from ...src.services.llms.llm import GPT
from ...src.services.chess import ChessBase


class CotGenerator(ChessBase):
    def __init__(
        self,
        is_truncate_response: bool = True,
        **kwargs
    ):
        """
        Generate Chain-of-Thought analysis for chess next move prediction.

        Args:
            is_truncate_response (bool, optional): truncate raw reponse from LLM. Defaults to True.
        """
        super().__init__(**kwargs)

        # Use CoT user prompt defined externally.
        self.user_prompter = user_prompt_instances.CoTGeneration()

        # Use GPT 4-o as default because GPT 3.5-turbo performs bad on Chess analysis.
        self.llm = GPT(llm_name="gpt-4o")

        # Set truncate response bool flag globally.
        self.is_truncate_response = is_truncate_response


    def get_player(
        self,
        fen: str
    ):
        """Read the full name of player from the shorten name.

        Args:
            fen (str): chess FEN containing w or b.

        Returns:
            player (str): full name of player for next move.
        """
        # Get player shortname from FEN, w or b.
        color_to_be_checked = fen.split(" ")[1]

        # Ensure only w or b.
        if color_to_be_checked not in ["w", "b"]:
            print(set_color("error", f"Player name is wrong, either w or b, but not {color_to_be_checked}."))
            exit(1)

        # Return the full name of player.
        return self.PLAYER_DICT[color_to_be_checked]


    def truncate_response(
        self,
        response: str
    ):
        """
        Truncate response with key words from system prompt.

        Args:
            response (str): raw response from LLM.

        Returns:
            response: truncated response or raw response if no truncation is required.
        """
        # If truncation is not required, return raw response.
        if not self.is_truncate_response:
            return response

        # Remove the first point of FEN analysis and rename all the indices.
        fen_analysis = response.split("\n1.")[-1].split("\n2.")[0]
        analysis = response.replace("\n1.", "")
        analysis = analysis.replace(fen_analysis, "")

        # Up to 10 CoT steps. If more steps are needed, increase it to more than 10.
        for i in range(10):
            analysis = analysis.replace(f"\n{i}.", f"\n{i-1}.")

        return analysis


    def __call__(
        self,
        fen: str,
        best_move: str
    ):
        """
        Generate CoT analysis for each chess puzzle.

        Args:
            fen         (str): chess FEN.
            best_move   (str): best move is also used as next move given FEN.

        Returns:
            cot_analysis        (str): truncated CoT analysis.
            raw_cot_analysis    (str): original CoT analysis.
            gpt_analysis        (str): analysis from GPT 4-o without CoT.
        """
        # Get player's full name, to be used in the user prompt for CoT analysis.
        player = self.get_player(fen)

        # Get LLM response with and without CoT instruction in the system prompt.
        for with_cot_instruct in [True, False]:
            # Get user prompt for the current chess status.
            user_prompt = self.user_prompter(fen, player, best_move, with_cot_instruct=with_cot_instruct)

            # Original response from LLM.
            response = self.llm(user_prompt)

            if with_cot_instruct is True:
                # Original CoT response.
                raw_cot_analysis = response

                # Truncated CoT response.
                cot_analysis = self.truncate_response(response)
            else:
                # GPT 4-o analysis for best move prediction.
                gpt_analysis = response

        return (cot_analysis, raw_cot_analysis, gpt_analysis)


    def batch_process(
        self,
        query_csv: str
    ):
        """
        Generate CoT analysis for multiple puzzle FENs in .csv.

        Args:
            query_csv (str): .csv path.
        """
        # This is to generate CoT analysis from OpenAI for model finetuning on reasoning.
        df = read_csv(query_csv)
        raw_fen = df["Question"]
        raw_best_move = df["Answer"]
        fens = []
        best_moves = []

        # The move has ["{best_move}"] so extract it from these keywords.
        for fen_per, best_move_per in zip(raw_fen, raw_best_move):
            if isinstance(best_move_per, str) and best_move_per.find("[") > -1 and best_move_per.find("]") > -1:
                fens.append(fen_per)
                best_moves.append(best_move_per.replace("["", "").replace(""]", ""))

        # Write the head for information to be stored in log file accordingly.
        if self.log_file is not None:
            self.log_file.writerow(["FEN", "Best Move", "CoT Analysis", "Raw CoT Analysis", "GPT Analysis"])

        for idx, (fen, best_move) in enumerate(zip(fens, best_moves)):
            # Print the progress.
            if idx % 10 == 0 or idx == len(fens) - 1:
                print(set_color("info", f"Generating finetune dataset for reasoning {idx + 1}/{len(fens)}..."))

            # Run OpenAI API with CoT step-by-step analysis and direct analysis.
            cot_analysis, raw_cot_analysis, gpt_analysis = self.__call__(fen, best_move)

            # Save to log.
            if self.log_file is not None:
                self.log_file.writerow([fen, best_move, cot_analysis, raw_cot_analysis, gpt_analysis])