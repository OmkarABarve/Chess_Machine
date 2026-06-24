# Chess Machine

A self-supervised machine learning system that learns the geometry of chess — not by programming rules, but by reading millions of real games and discovering strategic structure on its own.

---

## The Idea

Traditional chess engines like Stockfish evaluate positions using hand-crafted rules: piece values, pawn structure tables, king safety formulas. Every concept was written by a human expert.

**This project takes the opposite approach.**

The goal is to build a system that reads millions of grandmaster games and learns — entirely on its own — what makes positions similar, what patterns precede winning moves, and how pieces relate to each other across the board.

The inspiration comes directly from language models. In NLP, a model trained on text learns that "king" and "queen" are related, that "Paris" and "France" belong together — without ever being told. The vectors just fall into place because of how those words appear in context.

The hypothesis here is identical:

> If you train on enough chess positions, the model will learn a vector space where **strategically similar positions are close together**, even if the exact piece arrangements differ. Concepts like king safety, initiative, central control, and attacking potential should emerge naturally — without explicit labels.

---

## The Data

Raw PGN files from strong player databases and public game archives are processed through `DataClean.ipynb`.

For every move in every game, the pipeline stores:

| Field | Description |
|---|---|
| FEN | Board state as a string |
| Position hash | For identifying recurring structures |
| Move played | In UCI format (e.g. `e2e4`) |
| Side to move | White or Black |
| Game result | 1-0, 0-1, or 1/2 |
| Move number | Position in the game |
| Game ID | Source game reference |

Duplicate games are removed. Duplicate **positions** are intentionally kept — a position that appears in 10,000 games is important signal, not noise.

The final training dataset reached approximately **2.97 million positions**.

---

## Representation

Each chess piece on each square becomes a token.

```
WhiteKnight_f3
BlackQueen_d8
WhiteKing_e1
```

This gives a vocabulary of roughly **12 piece types × 64 squares = 768 possible tokens**.

Every token gets a learnable embedding vector — exactly like word embeddings in a language model.

---

## Architecture — Three Versions

The model was built in three progressive versions, each adding a new layer of chess intelligence.

### Version 1 — Flat Network

The board is flattened into 64 integers and fed through dense layers.

- Treats each square as an isolated number
- No understanding that two knights share movement properties
- Useful baseline to validate the data pipeline
- Estimated skill: **600–800 Elo**

### Version 2 — Embedding Layer

Raw piece numbers are replaced with learned embedding vectors via `nn.Embedding`.

- Each piece type develops its own identity in vector space
- Knights, bishops, and rooks form natural clusters
- Better generalization to unseen positions
- Estimated skill: **800–1000 Elo**

### Version 3 — Attention (Mini-Transformer)

The embeddings are passed through a **Multi-Head Self-Attention** layer, then pooled into a single board vector via a CLS token.

```
Piece-Square Token
      ↓
 128-D Embedding
      ↓
Multihead Self-Attention  ← pieces "talk" to each other
      ↓
  CLS Board Vector        ← single fixed-length board representation
      ↓
  Move Prediction
```

- A bishop on c1 can attend to a king on g8 across an open diagonal
- Pins, forks, and discovered threats can emerge from attention patterns
- Each position becomes a **256-dimensional vector** that captures its full strategic context
- Estimated skill: **1000–1200+ Elo**

---

## How Attention Works Here

The attention matrix has shape **(batch, 33, 33)**.

- **33** = 1 CLS token + up to 32 piece tokens per position
- **Row i** — how much token i attends to every other token
- **Column j** — how important token j is to token i

The CLS token aggregates information from all pieces simultaneously, producing a single board-level vector used for move prediction and similarity search.

**Current attention behavior (early training):**
Attention weights sit at roughly 0.02–0.05, meaning pieces are attending broadly rather than focusing sharply. This is normal for early training — the model has not yet learned to concentrate attention on the most tactically relevant relationships.

**What focused attention looks like at scale:**
As training continues, the model is expected to learn that a rook should attend strongly to open files, a bishop to its diagonal, and the king to nearby attackers. The attention heads become interpretable chess concepts.

---

## Training Results (V3)

| Metric | Value |
|---|---|
| Training positions | ~2.97 million |
| Unique moves in vocabulary | 1,986 |
| Chess tokens | 736 + 1 PAD |
| Loss (observed) | 5.89 |
| Loss (random baseline) | 7.59 (`-ln(1/1986)`) |
| Improvement over random | ~1.7 loss units |

The model significantly beats random guessing. It has learned real structure. It is not yet a strong player, but the core direction is validated.

**What has been proven:**
- The pipeline works end to end
- Embeddings learn meaningful representations
- Attention runs correctly
- Board positions become fixed-length vectors
- Move probabilities are generated
- The model beats random move selection

**What is not yet proven:**
- Strategic understanding
- Tactical calculation
- Strong chess strength

---

## Chess Coach

`Chess_Coach.py` is an interactive terminal application built on the V3 model.

You play a real game of chess. On your turn, the model suggests the top moves ranked by its learned probability — drawn directly from the attention-based board vector.

**Run it:**
```bash
python Chess_Coach.py
```

**Commands during the game:**

| Command | Action |
|---|---|
| `e2e4` | Make a move (UCI format) |
| `top10` | Show top 10 suggested moves |
| `fen` | Print the current FEN string |
| `undo` | Take back the last full move |
| `quit` | Exit |

**Move format examples:**

```
Pawn:      e2e4   d7d5
Knight:    g1f3   b8c6
Castling:  e1g1   e8c8
Promotion: e7e8q  e7e8r
```

The coach does not play against you. It watches your game and suggests what millions of grandmaster games suggest should come next.

---

## Possible Uses

The board vector produced by the attention encoder is not just for move prediction. It is a universal chess representation that opens a range of applications:

**Similarity Search**
Feed any board position → get its vector → find the most similar historical positions by cosine distance. See what continuations grandmasters played from structurally similar situations.

**Opening Discovery**
Cluster millions of positions by their board vectors to find opening families the model discovered on its own — without ECO codes or named openings.

**Strategic Plan Recommendation**
When a position vector is near a cluster of games that all featured a kingside attack, recommend that plan — even if the exact piece placement is novel.

**Puzzle Generation**
Find positions where the model's top suggestion diverges sharply from what was actually played. These are likely tactical moments where one move is dramatically better than all others.

**Player Style Embeddings**
Average all board vectors from a player's games into a single style vector. Compare players by their vector distance — aggressive attackers cluster together, solid positional players form their own region.

**Position Difficulty Estimation**
Positions where the model's probability is spread evenly across many moves are likely complex or unclear. Positions where probability concentrates on one move are likely forcing or simple.

**Human Blunder Prediction**
Train a classifier on the board vector: given this position, is a human likely to blunder here? The embedding captures the structural features that make positions confusing.

**Foundation Model for Downstream Tasks**
The encoder can be frozen and a lightweight head trained on top for any chess-specific task — result prediction, time pressure analysis, opening classification — without retraining the full model.

---

## Project Files

| File | Description |
|---|---|
| `DataClean.ipynb` | Full data pipeline: PGN parsing, deduplication, tokenization, vocabulary building |
| `Version3.ipynb` | V3 model definition, training loop, attention analysis, embedding visualization |
| `Chess_Coach.py` | Interactive terminal chess coach using the trained V3 model |
| `v3_attention_model.pt` | Trained V3 model weights |
| `v2_model.pt` | Trained V2 model weights |
| `piece_embeddings.pt` | Saved piece-square embedding vectors |
| `vocab.pkl` | Token → index mapping (piece-square vocabulary) |
| `move_vocab.pkl` | Move → index mapping (UCI move vocabulary) |
| `idx_to_token.pkl` | Index → token reverse mapping |
| `Idea.md` | Original project specification |
| `Chess V1 V2 V3.txt` | Architecture comparison across all three versions |
| `V3_analysis.md` | Detailed V3 training and attention analysis notes |
| `PCA Image.png` | PCA visualization of the embedding space |
| `Black White image.png` | Embedding visualization |
| `Data/` | Raw PGN game files |

---

## Dependencies

```bash
pip install torch chess pandas tqdm pyarrow
```

---

## Success Criteria

The project succeeds if embedding distances correspond to meaningful chess relationships — if positions that share strategic DNA end up near each other in vector space, without any human ever defining what "strategic DNA" means.

That is the test. The geometry of chess, discovered from data alone.
