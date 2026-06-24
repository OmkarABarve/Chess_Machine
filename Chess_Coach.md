This is not a rule-based chess engine.

It learns a vector representation of chess positions from millions of games.

Each piece-square combination is embedded into a learned latent space, similar to word embeddings in language models.

The system can predict moves, retrieve similar historical positions, analyze styles, and eventually support search-based play.

"""

Commands:

quit
undo
fen
top10
====================================================

MOVE INPUT FORMAT (UCI)

Use UCI notation:

Pawn:
e2e4
d7d5

Knight:
g1f3
b8c6

Bishop:
f1c4

Queen:
d1h5

Castling:
e1g1   (white kingside)
e1c1   (white queenside)

e8g8   (black kingside)
e8c8   (black queenside)

Promotion:
e7e8q
e7e8r
e7e8b
e7e8n

Examples:

White:
e2e4

Black:
c7c5

White:
g1f3

Black:
d7d6

====================================================
"""

python CHESS_COACH.py