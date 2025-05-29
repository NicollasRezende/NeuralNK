def get_valid_actions(board):
    return [i for i, x in enumerate(board) if x == 0]

def apply_move(board, move, player):
    new_board = board.copy()
    new_board[move] = player
    return new_board

def check_winner(board):
    winning_combos = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    for a, b, c in winning_combos:
        if board[a] == board[b] == board[c] and board[a] != 0:
            return board[a]
    return 0 if 0 in board else None

def get_reward(winner, player):
    if winner == player:
        return 10.0
    elif winner == -player:
        return -10.0
    elif winner is None:
        return 1.0
    else:
        return 0.0