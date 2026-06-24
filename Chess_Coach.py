import pickle
import chess

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================================================
# MOVE FORMAT
# ==================================================
"""
Examples:

e2e4
g1f3
e7e8q

Commands:

quit
undo
fen
top10
"""

# ==================================================
# LOAD FILES
# ==================================================

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("move_vocab.pkl", "rb") as f:
    move_vocab = pickle.load(f)

idx_to_move = {
    idx: move
    for move, idx in move_vocab.items()
}

PAD_ID = len(vocab)

VOCAB_SIZE = PAD_ID + 1

NUM_MOVES = len(move_vocab)

# ==================================================
# MODEL
# ==================================================

class ChessAttentionModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        num_moves,
        embedding_dim=128,
        num_heads=8
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_ID
        )

        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                embedding_dim
            )
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(
            embedding_dim
        )

        self.fc = nn.Sequential(

            nn.Linear(
                embedding_dim,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                num_moves
            )
        )

    def forward(self, x):

        batch_size = x.shape[0]

        mask = (
            x == PAD_ID
        )

        x = self.embedding(x)

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [cls, x],
            dim=1
        )

        cls_mask = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=x.device
        )

        mask = torch.cat(
            [cls_mask, mask],
            dim=1
        )

        attn_out, _ = self.attention(
            x,
            x,
            x,
            key_padding_mask=mask
        )

        attn_out = self.norm(attn_out)

        board_vector = attn_out[:, 0]

        return self.fc(board_vector)

# ==================================================
# LOAD MODEL
# ==================================================

print("Loading model...")

model = ChessAttentionModel(
    VOCAB_SIZE,
    NUM_MOVES
)

model.load_state_dict(
    torch.load(
        "v3_attention_model.pt",
        map_location="cpu"
    )
)

model.eval()

print("Model loaded.")

# ==================================================
# BOARD -> TOKENS
# ==================================================

def board_to_tokens(board):

    tokens = []

    for square, piece in board.piece_map().items():

        color = (
            "W"
            if piece.color
            else "B"
        )

        piece_symbol = (
            piece.symbol()
            .upper()
        )

        square_name = chess.square_name(
            square
        )

        token = (
            f"{color}{piece_symbol}_{square_name}"
        )

        if token in vocab:

            tokens.append(
                vocab[token]
            )

    return tokens

# ==================================================
# BOARD -> TENSOR
# ==================================================

def board_to_tensor(board):

    token_ids = board_to_tokens(
        board
    )

    arr = [PAD_ID] * 32

    n = min(
        len(token_ids),
        32
    )

    arr[:n] = token_ids[:n]

    return torch.tensor(
        [arr],
        dtype=torch.long
    )

# ==================================================
# MOVE SUGGESTIONS
# ==================================================

def suggest_moves(
    board,
    top_k=5
):

    x = board_to_tensor(board)

    with torch.no_grad():

        logits = model(x)

    probs = F.softmax(
        logits,
        dim=1
    )[0]

    ranked = torch.argsort(
        probs,
        descending=True
    )

    legal_moves = {
        move.uci()
        for move in board.legal_moves
    }

    suggestions = []

    for idx in ranked:

        move = idx_to_move[
            idx.item()
        ]

        if move in legal_moves:

            suggestions.append(
                (
                    move,
                    probs[idx].item()
                )
            )

        if len(suggestions) >= top_k:
            break

    return suggestions

# ==================================================
# DISPLAY HELPERS
# ==================================================

def show_suggestions(
    board,
    top_k=5
):

    print("\nSuggested moves:\n")

    suggestions = suggest_moves(
        board,
        top_k=top_k
    )

    for i, (move, score) in enumerate(
        suggestions,
        start=1
    ):

        print(
            f"{i}. {move:8s} {score:.4f}"
        )

# ==================================================
# START
# ==================================================

print("\n========================")
print("CHESS COACH")
print("========================\n")

side = input(
    "Play as White or Black? (w/b): "
).lower()

board = chess.Board()

# ==================================================
# MAIN LOOP
# ==================================================

while True:

    print("\n=================================")
    print(board)
    print("=================================")

    if board.is_checkmate():

        print("\nCHECKMATE")
        break

    if board.is_stalemate():

        print("\nSTALEMATE")
        break

    my_turn = (
        board.turn == chess.WHITE
        and side == "w"
    ) or (
        board.turn == chess.BLACK
        and side == "b"
    )

    if my_turn:

        show_suggestions(
            board,
            top_k=5
        )

        cmd = input(
            "\nYour move: "
        ).strip()

    else:

        cmd = input(
            "\nOpponent move: "
        ).strip()

    # ==========================================
    # COMMANDS
    # ==========================================

    if cmd == "quit":

        print("\nGoodbye.")
        break

    if cmd == "fen":

        print("\nFEN:\n")
        print(board.fen())
        continue

    if cmd == "top10":

        show_suggestions(
            board,
            top_k=10
        )
        continue

    if cmd == "undo":

        if len(board.move_stack) > 0:

            board.pop()

            if len(board.move_stack) > 0:
                board.pop()

            print(
                "\nLast full move undone."
            )

        continue

    # ==========================================
    # APPLY MOVE
    # ==========================================

    try:

        board.push_uci(
            cmd
        )

    except:

        print(
            "\nIllegal move."
        )