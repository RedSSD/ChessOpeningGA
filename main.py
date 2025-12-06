#!/usr/bin/env python3
"""
Genetic Algorithm for optimizing chess opening sequences.
"""

import random
import argparse
import shutil
import logging
import requests

import chess
import chess.engine

from deap import base, creator, tools

# ------------------------------------------------------
# LOGGING CONFIGURATION
# ------------------------------------------------------
logging.basicConfig(
    filename="ga.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# GA parameters
# ------------------------------------------------------
N_PLY = 8            # number of white moves (Stockfish replies automatically)
POP_SIZE = 120
N_GEN = 80
CX_PROB = 0.7
MUT_PROB = 0.2
TOURN_SIZE = 3
GENE_MAX = 255       # genes are integers in [0, GENE_MAX]

# Fitness: maximize (positive evaluation means advantage for White)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("gene", random.randint, 0, GENE_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=N_PLY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ------------------------------------------------------
# Stockfish connection
# ------------------------------------------------------
STOCKFISH_PATH = shutil.which("stockfish")
ENGINE = None
if STOCKFISH_PATH:
    try:
        ENGINE = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        print("Stockfish found:", STOCKFISH_PATH)
        logger.info(f"Stockfish initialized at {STOCKFISH_PATH}")
    except Exception as e:
        print("Could not start Stockfish:", e)
        logger.error(f"Failed to start Stockfish: {e}")
        ENGINE = None
else:
    print("Stockfish not found — fallback to material evaluation.")
    logger.warning("Stockfish not found – using material evaluation.")

# Simple material evaluation if no engine is available
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0
}

TELEGRAM_BOT_TOKEN = "7939757093:AAFNAOdipE_t0tYftgHQX0kkC-jPKp1sxsg"
TELEGRAM_CHAT_ID = "625577497"
def notify_telegram(message: str):
    """Send a message to your Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")



def material_evaluation(board: chess.Board) -> float:
    """Simple material score used if Stockfish is unavailable."""
    score = 0.0
    for piece_type, val in PIECE_VALUES.items():
        score += val * (len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK)))
    return score * 100.0

# ---------------------------
# Decode individual with FEN caching
# ---------------------------
move_cache = {}  # global cache to store decoded moves

def decode_individual(individual):
    """
    Decode genome → sequence of moves.
    After each White move, Stockfish plays Black's best move.
    Returns: moves list, board object, FENs after each half-move
    """
    board = chess.Board()
    moves = []
    fens = []

    for gene in individual:
        legal = list(board.legal_moves)
        if not legal:
            break

        idx = gene % len(legal)
        white_move = legal[idx]
        board.push(white_move)
        moves.append(white_move)
        fens.append(board.fen())

        # Black move via Stockfish
        if ENGINE is not None and not board.is_game_over():
            try:
                result = ENGINE.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)
                moves.append(result.move)
                fens.append(board.fen())
            except Exception as e:
                logger.warning(f"Engine play() failed: {e}")

        if board.is_game_over():
            break

    return moves, board, fens

# ---------------------------
# Convert individual to SAN/UCI/FEN with caching
# ---------------------------
def individual_to_san(individual):
    """
    Convert an individual to SAN, UCI, and FEN strings.
    Uses caching to avoid re-decoding with Stockfish.
    """
    ind_id = id(individual)
    if ind_id in move_cache:
        moves, fens = move_cache[ind_id]
    else:
        moves, _, fens = decode_individual(individual)
        move_cache[ind_id] = (moves, fens)

    b = chess.Board()
    san_list = []
    uci_list = []
    for mv in moves:
        san_list.append(b.san(mv))
        uci_list.append(mv.uci())
        b.push(mv)

    return san_list, uci_list, fens

# ---------------------------
# Fitness evaluation
# ---------------------------
def evaluate(individual):
    try:
        moves, board, fens = decode_individual(individual)

        if ENGINE is not None:
            try:
                info = ENGINE.analyse(board, chess.engine.Limit(depth=4))
                score = info.get("score")

                if score.is_mate():
                    mate = score.white().mate()
                    fitness = 100000.0 if mate and mate > 0 else -100000.0
                else:
                    fitness = float(score.white().score())
            except Exception as e:
                logger.warning(f"Engine analyse() failed: {e}")
                fitness = material_evaluation(board)
        else:
            fitness = material_evaluation(board)

        move_cache[id(individual)] = (moves, fens)  # store decoded moves for later
        return (fitness,)

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return (-99999.0,)

# Register DEAP operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=GENE_MAX, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)

# ---------------------------
# Main GA loop
# ---------------------------
def main(pop_size=POP_SIZE, n_gen=N_GEN, seed=None):
    if seed is not None:
        random.seed(seed)
        logger.info(f"Seed set to {seed}")

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: sum(f[0] for f in fits) / len(fits))
    stats.register("min", lambda fits: min(f[0] for f in fits))
    stats.register("max", lambda fits: max(f[0] for f in fits))
    
    logger.info(f"N_PLY: {N_PLY}, POP_SIZE: {POP_SIZE}, N_GEN: {N_GEN}, depth: 10")

    # Initial evaluation
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(1, n_gen + 1):
        logger.info(f"=== Generation {gen} start ===")

        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring
        hof.update(pop)
        record = stats.compile(pop)

        # Logging
        max_ind = max(pop, key=lambda x: x.fitness.values[0])
        min_ind = min(pop, key=lambda x: x.fitness.values[0])
        max_san, max_uci, max_fen = individual_to_san(max_ind)
        min_san, min_uci, min_fen = individual_to_san(min_ind)

        logger.info(
            f"Gen {gen}: avg={record['avg']:.2f}, max={record['max']:.2f}, "
            f"min={record['min']:.2f}"
        )
        logger.info(
            f"Max moves UCI: [{','.join(max_uci)}]"
        )
        logger.info(
            f"Min moves UCI: [{' '.join(min_uci)}]"
        )

        notify_telegram(f"Gen {gen}: avg={record['avg']:.1f}, max={record['max']:.1f}, min={record['min']:.1f}")

    # Final HOF
    logger.info("=== Final Hall of Fame ===")
    print("\n=== Best opening sequences found (Top 5) ===")
    for i, ind in enumerate(hof):
        san, uci, fens = individual_to_san(ind)
        fit = ind.fitness.values[0]
        logger.info(f"#{i+1} fitness={fit:.2f} UCI=[{','.join(uci)}]")
        print(f"\n#{i+1} fitness={fit:.1f}")
        print("SAN:", " ".join(san))
        print("UCI:", " ".join(uci))

    notify_telegram("GA FINISHED")

    if ENGINE is not None:
        ENGINE.quit()

    return pop, hof


# ------------------------------------------------------
# CLI
# ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA for optimizing chess openings")
    parser.add_argument("--pop", type=int, default=POP_SIZE)
    parser.add_argument("--gen", type=int, default=N_GEN)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    notify_telegram("Starting GA")
    main(pop_size=args.pop, n_gen=args.gen, seed=args.seed)
