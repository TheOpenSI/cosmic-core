### Core modules ###import os, sys
from pandas import read_csv


### Type hints ###


### Internal modules ###
from ...src.services import chess as chess_instances
from ...utils.log_tool import set_color
from ...src.services.chess import ChessBase


class FENGenerator(ChessBase):
    def __init__(
        self,
        **kwargs
    ):
        """
        Generate chess FEN with a sequence of moves.
        """
        super().__init__(**kwargs)

        # Use Stockfish as chess backend.
        self.chess_engine = chess_instances.StockfishFENNextMove()


    def __call__(
        self,
        move_string: str
    ):
        """
        Parse the string containing moves, set the last move as to be predicted.

        Args:
            move_string (str): a string of moves.

        Returns:
            fen         (str): chess FEN after all the moves except for the last move.
            next_move   (str): last move as next move.
        """
        # Remain the last checkmate move with the rest to generate FEN.
        moves = [str(v.replace(".", "")) for v in move_string.split(" ") if v != ""]

        # Last move as next move.
        next_move = moves[-1]

        # Reset chess board.
        self.chess_engine.reset_board()

        # Push a bunch of moves.
        for current_move in moves[:-1]:
            self.chess_engine.push_single(current_move, move_mode="algebric")

        # Get chess FEN.
        fen = self.chess_engine.get_fen()

        return (fen, next_move)


    def batch_process(
        self,
        query_csv: str
    ):
        """
        Generate chess FEN for multiple queries.

        Args:
            query_csv (str): .csv file containing multiple queries of move sequences.
        """
        # This is to generate FEN with a sequence of moves to chess engine for model finetuning on reasoning checkmate.
        df = read_csv(query_csv)
        raw_moves = df["moves"]

        # Write log head for printed information.
        if self.log_file is not None:
            self.log_file.writerow(["FEN", "Next Move"])

        for idx, raw_move_string in enumerate(raw_moves):
            # Print the progress.
            if idx % 10 == 0 or idx == len(raw_moves) - 1:
                print(set_color("info", f"Generating FEN with moves for reasoning on checkmate {idx + 1}/{len(raw_moves)}..."))

            # Generate chess FEN for each query.
            fen, next_move = self.__call__(raw_move_string)

            # Write to .csv file
            if self.log_file is not None:
                self.log_file.writerow([fen, f"['{next_move}']"])