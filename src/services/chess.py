### Core modules ###
from sys import exit
from pathlib import Path
from chess import (
    Board,
    Move
)
from stockfish import Stockfish


### Type hints ###


### Internal modules ###
from ...utils.log_tool import set_color
from .llms.llm import GPT
from .base import ServiceBase


class ChessBase(ServiceBase):
    def __init__(
        self,
        binary_path: str = "",
        **kwargs
    ):
        """
        Base class for all chess questions.

        Args:
            binary_path (str, optional): stockfish executable file path. Defaults to "".
        """
        super().__init__(**kwargs)

        self.root = Path(__file__).resolve(strict=True).parent.parent.parent

        self.valid_move_modes = ["algebric", "coordinate"]

        # Get an initial chess board.
        self.board = Board()

        # To get the full player name from the shortname.
        self.PLAYER_DICT = {"w": "White", "b": "Black"}

        self.binary_path: Path = Path(binary_path).resolve(strict=True)

        # Set initial path for stockfish.
        if self.binary_path == "":
            self.binary_path = self.root.joinpath("third_party/stockfish/stockfish-ubuntu-x86-64-avx2")

        # Kill the entire program.
        if not self.binary_path.exists(follow_symlinks=True):
            print(
                set_color(
                    status="error",
                    information=f"!!!Error, stockfish binary file not exist: {binary_path}."
                )
            )
            exit(1)

        # Set chess engine.
        self.stockfish = Stockfish(
            path=binary_path,
            depth=20,
            parameters={
                "Threads": 2,
                "Minimum Thinking Time": 30
            },
            num_nodes=1000000,
            turn_perspective=True,
            debug_view=False
        )

        # Initialize the engine.
        self.reset_board()


    def reset_board(self):
        """
        Reset stockfish board for each chess game.
        """
        # Reset every in the board.
        self.board.reset()

        # Reset stockfish.
        self.stockfish.reset_engine_parameters()
        self.stockfish.set_fen_position(fen_position="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")


    def get_fen(self):
        """
        Get FEN of current chess board.

        Returns:
            fen (str): FEN of the current chess board.
        """
        return self.board.fen()


    def convert_algebric_to_coordinate(
        self,
        move: str
    ):
        """
        Convert move mode from algebric to coordinate.

        Args:
            move (str): a chess move in algebric mode.

        Returns:
            move (str): a chess move in coodinate mode.
        """
        # Illegal san will raise error and kill the entire program.
        try:
            move_san = self.board.parse_san(move)
        except:
            return None

        return str(move_san)


    def convert_coordinate_to_algebric(
        self,
        move: str
    ):
        """
        Convert move mode from coordinate to algebric.

        Args:
            move (str): a chess move in coodinate mode.

        Returns:
            move (str): a chess move in algebric mode.
        """
        # If move is a string, then convert to chess.Move for the board to parse.
        if not isinstance(move, Move):
            chess_compatible_move = Move.from_uci(move)

            return str(self.board.san(chess_compatible_move))
        else:
            return ""


    def check_is_legal_move(
        self,
        move: str,
        move_mode: str
    ):
        """
        Check if the next move is legal for the current chess board.

        Args:
            move        (str): a chess move.
            move_mode   (str): move mode.

        Returns:
            is_legal (bool): check if the move is legal.
        """
        # self.board requires coordinate format.
        if move_mode == "algebric":
            chess_algebric_move = self.convert_algebric_to_coordinate(move)

            # Move can be illegal with wrong inputs.
            if chess_algebric_move is None:
                is_legal = False

                return is_legal
            else:
                # Check if the move is in the legal moves on the current board.
                is_legal = move in [str(v) for v in self.board.legal_moves]

                return is_legal

        else:
            return False


    def push_single(
        self,
        current_move: str,
        move_mode: str
    ):
        """
        Push a chess move to the current chess board.

        Args:
            current_move    (str): a chess move.
            move_mode       (str): move mode.

        Returns:
            status (int): 0 for legal move and -1 for illegal move.
        """
        # Check if the move is legal.
        is_legal = self.check_is_legal_move(current_move, move_mode)

        if not is_legal:
            print(
                set_color(
                    status="error",
                    information=f"\nMove {current_move} is illegal on FEN {self.board.fen}."
                )
            )

            return -1

        # Convert from algebraic to coordinate.
        if move_mode == "algebric":
            chess_algebric_current_move = self.convert_algebric_to_coordinate(current_move)

            # If the current move is valid, then push to the board.
            self.board.push(Move.from_uci(str(object=chess_algebric_current_move)))

            # Also push the move to stockfish.
            self.stockfish.make_moves_from_current_position([str(object=chess_algebric_current_move)])

            return 0

        else:
            return -1


    def set_fen(
        self,
        fen: str
    ):
        """
        Set FEN to the current chess board.

        Args:
            fen (str): a given chess FEN.
        """
        # Reset the board to set a given FEN.
        self.reset_board()

        # Set the FEN.
        self.board.set_fen(fen)

        # Set FEN to stockfish.
        self.stockfish.set_fen_position(fen)


    def get_next_moves(self):
        """
        Get all legal next moves given the current chess board.

        Returns:
            next_move_list (list): a list of legal moves for the current chess board.
        """
        # All next moves on the current board, unnecessary to be the best move(s)
        next_move_list = [v for v in self.board.generate_legal_moves()]

        return next_move_list


    def _check_move_mode(
        self,
        move_mode: str
    ):
        """
        Internally check if the move mode is legal.

        Args:
            move_mode (str): move mode.
        """
        # Only support algebric mode and coordinate mode.
        if move_mode not in self.valid_move_modes:
            print(
                set_color(
                    status="error",
                    information=f"!!!Error, unknown move mode: {move_mode}, only support {self.valid_move_modes}."
                )
            )
            exit(1)


    def __call__(
        self,
        current_move: str = "",
        move_mode: str = "coordinate",
        topk: int = 1
    ):
        """
        Predict the next move given current_move.

        Args:
            current_move    (str, optional): a move string or a sequence of moves. Defaults to "".
            move_mode       (str, optional): move mode. Defaults to "coordinate".
            topk            (int, optional): up to topk predicted moves returned. Defaults to 1.

        Returns:
            next_move (list): a list of predicted topk next move(s).
        """
        # Check if move_mode is valid.
        self._check_move_mode(move_mode)

        # For moves in coordinate mode, if it is empty, no push but automatic estimate.
        if current_move != "":
            # Parse moves in a string, if it is a "" also put it to a list.
            if isinstance(current_move, str):
                current_moves = [str(v.replace(".", "")) for v in current_move.split(" ") if v != ""]

            # If push a list with multiple moves in advance.
            for current_move_per in current_moves:
                state = self.push_single(current_move_per, move_mode)

                if state < 0:
                    return []

        # Check if game over.
        is_game_over = self.board.is_game_over()

        next_move = []

        # If game over, return empty as next move.
        if is_game_over:
            return next_move

        # For puzzle, to get the best solution(s) with multiple moves; otherwise, just general next moves.
        if topk == 1:
            # Return only one move.
            best_move = self.stockfish.get_best_move()
            best_solution_list = [best_move]
        else:
            # Returns topk moves.
            top_moves = self.stockfish.get_top_moves(topk)
            best_solution_list = [v["Move"] for v in top_moves]

        for best_solution in best_solution_list:
            # Each solution is a move.
            if move_mode == "algebric":
                best_solution = self.convert_coordinate_to_algebric(str(object=best_solution))
                next_move.append(best_solution)
            else:
                # Append directly as it is already coordinate required.
                next_move.append(str(best_solution))

        return next_move


class StockfishSequenceNextMove(ChessBase):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        """
        Predict the next move given a sequence of moves.
        """
        super().__init__(*args, **kwargs)


    def __call__(
        self,
        moves: str,
        move_mode: str = "",
        topk: int = 1
    ):
        """
        Predict the next move for a query.

        Args:
            moves       (str):              a string of moves.
            move_mode   (str, optional):    move mode. Defaults to "".
            topk        (int, optional):    topk move(s) to be predicted. Defaults to 1.

        Returns:
            next_move_list (list): a list of topk next move(s).
        """
        # Clean the current board and restart the game.
        self.reset_board()

        # Estimate the next moves for puzzle.
        next_move_list = super().__call__(
            current_move=moves,
            move_mode=move_mode,
            topk=topk
        )

        return next_move_list


class StockfishFENNextMove(ChessBase):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        """
        Predict the next move given a chess FEN using Stockfish as backend.
        """
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        fen: str,
        move_mode: str = "",
        topk: int = 1
    ):
        """
        Predict the next move for a query.

        Args:
            fen         (str):              a chess FEN.
            move_mode   (str, optional):    move mode. Defaults to "".
            topk        (int, optional):    number of next moves. Default to 1.

        Returns:
            next_move_list (list): a list of topk next move(s).
        """
        # Clean the current board and restart the game
        self.set_fen(fen)

        # Estimate the next moves for puzzle
        next_move_list = super().__call__(
            move_mode=move_mode,
            topk=topk
        )

        return next_move_list


class GPTFENNextMove(ChessBase):
    def __init__(
        self,
        llm_name: str = "gpt-4o",
        is_truncate_response: bool = True
    ):
        """
        Predict the next move given a chess FEN using GPT as backend.

        Args:
            llm_name                (str, optional):    LLM name. Defaults to "gpt-4o".
            is_truncate_response    (bool, optional):   Truncate response for next move prediction.
                                                        Defaults to True.
        """
        # Set config.
        self.llm_name = llm_name
        self.is_truncate_response = is_truncate_response

        # Default to use GPT model.
        if llm_name.find("ollama") > -1:
            self.llm = GPT(llm_name, user_prompt_instance_name="FenNextMovePredict")

        self.llm = GPT(llm_name, user_prompt_instance_name="FenNextMovePredict")


    def quit(self):
        """
        Release LLM memory cached on GPU and LLM instannce.
        """
        self.llm.quit()
        del self.llm


    def get_player(
        self,
        fen: str
    ):
        """
        Parse FEN to get the player for the next move.

        Args:
            fen (str): a given chess FEN.

        Returns:
            player (str): the player's full name, White or Black.
        """
        # Get player shortname from FEN
        color_to_be_checked = fen.split(" ")[1]

        if color_to_be_checked not in ["w", "b"]:
            print(
                set_color(
                    status="error",
                    information=f"Player name is wrong, either w or b, but not {color_to_be_checked}."
                )
            )
            exit(1)

        return self.PLAYER_DICT[color_to_be_checked]


    def truncate_response(
        self,
        response: str
    ):
        """
        Truncate the response from LLM.

        Args:
            response (str): response from LLM.

        Returns:
            response (str): truncated response if applicable.
        """
        # Return full response if no truncation is required.
        if not self.is_truncate_response:
            return response

        # Truncate according to keywords of LLM.
        try:
            if self.llm_name  == "gpt-4o":
                response = response.split("**")[1].replace(".", "").split(" ")[-1]
            elif self.llm_name == "gpt-3.5-turbo":
                response = response.split("is ")[-1].replace(".", "").replace("*", "").split(" ")[-1]
        except:
            response = response.split("** is")[-1]

        return response


    def __call__(
        self,
        fen: str,
        move_mode: str = ""
    ):
        """
        Main entry to predict next move for a query.

        Args:
            fen (str): a chess FEN.
            move_mode (str, optional): move mode. Defaults to "".

        Returns:
            next_move: the next move predicted for the given FEN.
        """
        # Get the player's full name.
        player = self.get_player(fen)

        # Get the next move response.
        response = self.llm(fen, player)

        # Truncate to get the next move and set the next move in a list.
        next_move = self.truncate_response(response)
        next_move = [next_move]

        return next_move