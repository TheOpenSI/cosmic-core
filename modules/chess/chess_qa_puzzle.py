### Core modules ###
from pandas import read_csv
from numpy import (
    mean,
    array
)


### Type hints ###


### Internal modules ###
from ...src.services import chess as chess_instances
from ...src.services.llms.prompts import user_prompt as user_prompter_instances
from ...src.services.llms.prompts import system_prompt as system_prompter_instances
from ...src.services.qa import QABase
from ...utils.log_tool import set_color
from ...utils.module import get_instance


class PuzzleAnalyse(QABase):
    def __init__(
        self,
        next_move_predict_backend: str = "stockfish",
        is_truncate_response: bool = False,
        is_rag: bool = False,
        binary_path: str = "",
        **kwargs
    ):
        """
        Use LLM to analyse the reason of taking next move based on a given chess FEN.

        Args:
            next_move_predict_backend   (str, optional):    use stockfish or LLM for analysis.
                                                            Defaults to "stockfish".
            is_truncate_response        (bool, optional):   truncate response from LLM. Defaults to False.
            is_rag                      (bool, optional):   use RAG to extract context from system's vector database.
            binary_path                 (str):              path of Stockfish binary file.
                                                            Defaults to False.
        """
        super().__init__(**kwargs)

        # Set config.
        self.llm_name = self.llm.llm_name
        self.is_truncate_response = is_truncate_response
        self.is_rag = is_rag

        # Use GPT or Stockfish to predict next move given a chess FEN.
        if next_move_predict_backend == "gpt":
            self.fen_next_move_predictor = chess_instances.GPTFENNextMove()

        else:
            self.fen_next_move_predictor = chess_instances.StockfishFENNextMove(
                binary_path=binary_path
            )

        # User prompt and system prompt specific for next move analysis.
        if self.llm_name.find("mistral") > -1 and self.llm_name.find("finetune") > -1:
            user_prompt_instance_name = "FENNextMoveAnalyseMistralFinetuned"
            system_prompt_instance_name = "FENNextMoveAnalyseMistralFinetuned"

        else:
            user_prompt_instance_name = "FENNextMoveAnalyse"
            system_prompt_instance_name = "FENNextMoveAnalyse"

        # Find the use and system prompter in src/llms/prompts/user_prompt
        # and src/llms/prompts/system_prompt respectively.
        self.user_prompter = get_instance(user_prompter_instances, user_prompt_instance_name)()
        self.system_prompter = get_instance(system_prompter_instances, system_prompt_instance_name)()


    def parse_puzzle_csv(
        self,
        puzzle_path: str
    ):
        """
        Get chess status from .csv file.

        Args:
            puzzle_path (str): .csv file containing the chess information for next move.

        Returns:
            info (dict): a dictionary containing chess status information.
        """
        # Parse .csv to get all FENs.
        df = read_csv(puzzle_path)
        fens = df["FEN"]
        best_cases = df["best_case"]
        players = df["player"]
        best_move_list = df["moves"]

        # When start from black, the number of moves is across two blobs, thus minus 1.
        best_case_updated = []

        for idx, player in enumerate(players):
            # Convert to the number of moves, not blobs.
            if player == "White":
                num_moves = (best_cases[idx] - 1) * 2

            else:
                num_moves = best_cases[idx] * 2

            best_case_updated.append(num_moves)

        # Parse the ground truth moves given the FEN.
        # Remove namespace and . for each piece of moves.
        best_move_updated = []

        for best_moves in best_move_list:
            # Truncate to best the moves only.
            best_move_updated_per = [v for v in best_moves.split(" ") if v.find(".") <= -1]

            # Always add # to the last move.
            best_move_updated_per[-1] += "#"

            # Assign to the entire question list.
            best_move_updated.append(best_move_updated_per)

        # Write information to a dictionary.
        info = {
            "fen": fens,
            "best_case": best_case_updated,
            "moves": best_move_updated,
            "player": players
        }

        return info


    def truncate_response(
        self,
        response: str
    ):
        """
        Truncate next move analysis using keywords in system prompt.

        Args:
            response (str): original analysis.

        Returns:
            response (str): truncated analysis if applicable.
        """
        try:
            # Parse the answer, corresponding to system prompt and LLM.
            if self.llm_name.find("mistral") > -1:
                if self.llm_name.find("-instruct") > -1:
                    response = response.split("[/INST]")[-1].split("</s>")[0]

                else:
                    if self.llm_name.find("finetune") > -1:
                        response = response.split("\n")[-1]

                    else:
                        # Extract from uncertain keywords.
                        if self.llm.use_example:
                            response = response.split("**Solution:**")[1]

                        else:
                            response = response.split("Answer")[1].split("Comment:")[0]

            elif self.llm_name.find("gemma") > -1:
                if self.llm_name.find("-it") > -1:
                    response = response.split("model\n")[1]

                else:
                    # Extract from uncertain keywords.
                    response = response.split("Answer:\n")[1]

        except:
            response = response

        response = response.replace("\n", " ").strip()

        return response


    def __call__(
        self,
        fen: str,
        player: str,
        gt_moves: str,
        move_mode: str = "",
        idx: int = 0
    ):
        """
        Generate next move and analyse the reason for each puzzle.

        Args:
            fen         (str):              a chess FEN.
            player      (str):              player for next move.
            gt_moves    (str):              expected next move for the current FEN.
            move_mode   (str, optional):    "coordinate" or "algebric". Defaults to "".
            idx         (int, optional):    index of FEN query in the .csv file. Defaults to 0.

        Returns:
            score: if predicted next move is the same as expected.
        """
        # Set FEN to predictor.
        self.fen_next_move_predictor.set_fen(fen)

        # Store every predicted next move for a puzzle.
        next_moves = []

        for idx_move, gt_move in enumerate(gt_moves):
            # Even step is the opponent, odd step is the player.
            # Only push next_move for the player, and gt_move for the opponent.
            if idx_move % 2 == 0:
                next_move = gt_move
            else:
                # Stockfish only returns the best move, so push FEN to get the best move.
                next_move = self.fen_next_move_predictor(current_fen, move_mode=move_mode)[0]

                # Interaction between LLM and chess engine.
                analysis, raw_analysis, _ = self.analyse_puzzle(
                    current_fen,
                    player,
                    next_move
                )

                # Display the analysis.
                print(set_color(
                    "info",
                    f"Puzzle {idx + 1} at step {idx_move + 1}:" \
                    f" player {player} takes {next_move} given FEN '{current_fen}'.\n" \
                    f"Analysis: {analysis}\n")
                )

                # Save to log file.
                if self.log_file is not None:
                    self.log_file.writerow([
                        f"Puzzle {idx + 1} at step {idx_move + 1}: " \
                        f"player {player} takes {next_move} given FEN '{current_fen}'",
                        analysis,
                        "",
                        "",
                        "",
                        raw_analysis
                    ])

            try:
                # Push the predicted move to the chess board.
                status = self.fen_next_move_predictor.push_single(next_move, move_mode=move_mode)

            except:
                # If any errors raised, the next move is invalid.
                print(set_color("fail", f"Move {next_move} for FEN {current_fen} is invalid."))
                status = -1

            # If the next move is illegal, stop processing the current puzzle.
            if status < 0:
                next_moves.append(next_move)
                break

            else:
                # Then update the FEN in chess engine.
                current_fen = self.fen_next_move_predictor.get_fen()

                # Save the actual move to next_moves.
                next_moves.append(next_move)

        # Check if next moves are the same as GT moves, only on the player which is odd.
        comparison_list = \
            [float(next_v == gt_v) for idx_inner, (next_v, gt_v) \
             in enumerate(zip(next_moves, gt_moves)) if idx_inner % 2 == 1]

        # Get the average score over all predicted moves in this puzzle.
        score = mean(array(comparison_list))

        # Save to log file.
        if self.log_file is not None:
            self.log_file.writerow([fen, next_moves, gt_moves, score])

        return score


    def analyse_puzzle(
        self,
        fen: str,
        player: str,
        next_move: str
    ):
        """
        Analyse next move given a chess FEN.

        Args:
            fen         (str): a given chess FEN before taking the next move.
            player      (str): player for the next move.
            next_move   (str): the next move for the given FEN.

        Returns:
            response        (str): truncated response if applicable.
            raw_response    (str): orignial response.
            retrieve_score  (str): similarity score for retrieved context if applicable.
        """
        # Generate user prompt given chess status.
        user_prompt = self.user_prompter(fen, player, next_move)

        # Get response and retrieve score if is_rag=True.
        response, raw_response, retrieve_score = super().__call__(
            user_prompt,
            is_rag=self.is_rag
        )

        if self.is_truncate_response:
            response = self.truncate_response(raw_response)
        else:
            response = raw_response.replace("\n", " ")

        return (response, raw_response, retrieve_score)


    def batch_process(
        self,
        query_csv: str,
        move_mode: str = ""
    ):
        """
        Process multiple puzzles in a .csv file.

        Args:
            query_csv (str): .csv file containing multiple puzzles.
            move_mode (str, optional): "coordinate" or "algebric". Defaults to "".

        Returns:
            average_score (float): average score over all puzzles.
        """
        # Read the csv file to get questions and context.
        puzzle_info = self.parse_puzzle_csv(query_csv)

        # Use context to indicate the move mode.
        if move_mode == "": move_mode = "algebric"

        # Get all puzzle status.
        fens = puzzle_info["fen"]
        gt_move_lists = puzzle_info["moves"]
        players = puzzle_info["player"]

        num_fens = len(fens)
        score_list = []

        # Write information head for log file.
        if self.log_file is not None:
            self.log_file.writerow(["Question", "Answer", "Label", "Score", "Comment", "Raw Answer"])

        for idx, (fen, gt_moves, player) in enumerate(zip(fens, gt_move_lists, players)):
            # Print the progress.
            if idx % 10 == 0 or idx == num_fens - 1:
                print(set_color("info", f"Solving puzzles {idx + 1}/{num_fens}..."))

            # Check if the predicted next moves are correct in a puzzle.
            score_per = self.__call__(fen, player, gt_moves, move_mode=move_mode, idx=idx)

            # Save to score list.
            score_list.append(score_per)

        # Calculate the average score over all puzzles' next move prediction.
        average_score = mean(score_list)

        # Save results to log file.
        if self.log_file is not None:
            self.log_file.writerow([query_csv, "", "", f"{average_score:.4f}"])

        # Print for progress checking.
        print(set_color("info", "Solving puzzles finished."))

        return average_score