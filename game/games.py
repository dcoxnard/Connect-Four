import time
import numpy as np


BOARD_DIMENSIONS = (6, 7)
PLAYER_1_TOKEN = 1
PLAYER_2_TOKEN = 2
EMPTY_TOKEN = 0
CONNECT_N = 4


def make_board(board_dim):
    """
    Initialize empty board with passed dimensions.
    """
    board = np.zeros(board_dim)
    return board


def make_choice(model, board, empty_token):
    """
    If no prediction model is passed,
    return a random choice of column to place a piece.
    Otherwise, use the model to choose the column with
    the highest predicted reward.
    """
    if model is None:
        choice = random_choice(board, empty_token)
    else:
        choice = predict_choice(model, board, empty_token)
    return choice


def random_choice(board, empty_token):
    """
    Return a random choice, selected uniformly from
    only the remaining legal options.
    """
    n_cols = board.shape[1]
    legal_moves = board[0] == empty_token
    move_probs = legal_moves / sum(legal_moves)
    choice = np.random.choice(np.arange(0, n_cols), p=move_probs)
    return choice


def predict_choice(model, board, empty_token):
    """
    Return predictions of reward for each column.
    Return the choice with the highest predicted reward.
    """
    flat_board = board.flatten()
    pred = model.predict(flat_board)
    legal_moves = board[0] == empty_token
    choice = np.argmax(pred * legal_moves)

    return choice


def place_token(board, choice, player_token, empty_token):
    """
    Simulate the effect of gravity when placing a piece.
    Searches for the bottommost empty slot in the given column,
    and returns the updated board state.
    """
    slot = board.T[choice]
    lands_at = np.argwhere(slot == empty_token).max()  # Throws error if no max
    updated_board = np.copy(board)
    updated_board[lands_at][choice] = player_token

    return updated_board


def check_win(board, player_token, empty_token, connect_n):
    """
    Complete check of the board for win and draw conditions.
    Each of row, column, and diagonal win are delegrated
    to helper functions.
    Checks for draw last by looking whether there are any
    remaining open spaces.
    """
    positions_taken = board != empty_token
    if positions_taken.sum() == board.shape[0] * board.shape[1]:
        result = "DRAW"
    elif check_row_win(board, player_token, connect_n):
        result = f"WIN - {player_token} - ROW"
    elif check_col_win(board, player_token, connect_n):
        result = f"WIN - {player_token} - COLUMN"
    elif check_diag_win(board, player_token, connect_n):
        result = f"WIN - {player_token} - DIAGONAL"
    else:
        result = False
    return result


def check_row_win(board, player_token, connect_n):
    """
    Checks whether any horizontal run of four pieces belongs
    to the same player.
    """
    found = False
    check_against = np.full(4, player_token)
    for row in range(board.shape[0]):
        for i in range(board.shape[1] - (connect_n - 1)):
            row_segment = board[row][i:i+4]
            if np.array_equal(row_segment, check_against):
                found = True
                break
    return found


def check_col_win(board, player_token, connect_n):
    """
    Checks whether any vertical run of four pieces is
    owned by the same player.
    """
    found = False
    board_transposed = np.transpose(board)
    check_against = np.full(4, player_token)
    for row in range(board_transposed.shape[0]):
        for i in range(board_transposed.shape[1] - (connect_n - 1)):
            col_segment = board_transposed[row][i:i+connect_n]
            if np.array_equal(col_segment, check_against):
                found = True
                break
    return found


def check_diag_win(board, player_token, connect_n):
    """
    Checks all possible diagonals to find a run of four
    consecutive pieces.
    First checks left-to-right, then right-to-left.
    """
    n_rows = board.shape[0]
    n_cols = board.shape[1]
    check_against = [player_token] * connect_n

    # Check diag left to right
    found = False
    for row in range(n_rows - (connect_n - 1)):
        for col in range(n_cols - (connect_n - 1)):
            diag = make_rightward_diag(board, row, col, connect_n)
            if diag == check_against:
                found = True
                break

    # Check diag right to left
    for row in range(n_rows - (connect_n - 1)):
        for col in range(n_cols - 1, n_cols - (connect_n + 1), -1):
            diag = make_leftward_diag(board, row, col, connect_n)
            if diag == check_against:
                found = True
                break

    return found


def make_rightward_diag(board, i, j, connect_n):
    """
    Helper function for checking win along diagonal, left-to-right.
    """
    diag = []
    for z in range(connect_n):
        diag.append(board[i+z][j+z])
    return diag


def make_leftward_diag(board, i, j, connect_n):
    """
    Helper function for checking win along diagonal, right-to-left.
    """
    diag = []
    for z in range(connect_n):
        diag.append(board[i+z][j-z])
    return diag


def play_game(p1_model, p2_model=None, board_dimensions=BOARD_DIMENSIONS, p1_token=PLAYER_1_TOKEN,
              p2_token= PLAYER_2_TOKEN, empty_token=EMPTY_TOKEN, connect_n=CONNECT_N):
    """
    Simulate a single game of Connect Four.

    Return 4-tuple of winning condition, winner number,
    history of board states, and history of player 1's
    move choices.

    Takes an optional model argument, which is used to generate
    move choices based on predicted reward.

    If no model is passed, all choices are made at random.
    """

    board = make_board(board_dimensions)
    game_over = False
    winner = None

    states = [board]
    choices = []

    while not game_over:

        p1_choice = make_choice(p1_model, board, empty_token)
        board = place_token(board, p1_choice, p1_token, empty_token)
        choices.append(p1_choice)
        states.append(board)
        game_over = check_win(board, p1_token, empty_token, connect_n)
        if game_over:
            winner = p1_token
            break

        p2_choice = make_choice(p2_model, board, empty_token)
        board = place_token(board, p2_choice, p2_token, empty_token)
        choices.append(p2_choice)
        states.append(board)
        game_over = check_win(board, p2_token, empty_token, connect_n)
        if game_over:
            winner = p2_token
            break

    game = {
        "game_over": game_over,
        "winner": winner,
        "states": states,
        "choices": choices
    }

    return game


def draw_game(game, delay=0.5):
    """
    Draw results of a game to the screen
    """
    states = game["states"]
    choices = game["choices"]
    for i, state in enumerate(states):
        print("\n")
        if i != 0:
            print(" " * 3 + " " * (4 * (choices[i-1])) + "V")
        print(state)
        time.sleep(delay)


def play_n_games(n_games=100, p1_model=None, p2_model=None,
                 board_dimensions=BOARD_DIMENSIONS, p1_token=PLAYER_1_TOKEN,
                 p2_token=PLAYER_2_TOKEN, empty_token=EMPTY_TOKEN,
                 connect_n=CONNECT_N):
    """
    Play many games of Connect 4 and return their histories.
    """
    games = []
    for i in range(n_games):
        game = play_game(p1_model, p2_model, board_dimensions, p1_token,
                         p2_token, empty_token, connect_n)
        games.append(game)
    return games


if __name__ == "__main__":
    model = None
    n_games = 100
    start = time.time()
    games = play_n_games(n_games)
    end = time.time()
    print(f'Time to play {n_games} games: {round(end - start, 3)} seconds')
