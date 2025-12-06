import chess
import chess.svg

board = chess.Board()

uci_moves = []

for move in uci_moves:
    board.push_uci(move)

with open("board.svg", "w") as f:
    f.write(chess.svg.board(board=board))
