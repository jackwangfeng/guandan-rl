# Mini-Guandan Bot — Integration Guide

> ⚠️ **The model plays a SIMPLIFIED Guandan**.  Caller must convert to/from this representation. Reading this document end-to-end is required before integration.

## What's modelled

- 4 players, 2v2 teams (P0+P2 vs P1+P3)
- 108 cards (2 standard 54-card decks)
- 15 card types (no suits): `3 4 5 6 7 8 9 T J Q K A 2 sj bj` → indices `0..14`
- Combos:
  - `0` PASS · `1` SINGLE · `2` PAIR · `3` TRIPLE · `4` THREE+TWO · `5` BOMB
  - **v4 only**: `6` SEQ5 (single sequence, 5 consecutive ranks) · `7` PSEQ3 (3 consecutive pair ranks) · `8` TSEQ2 (2 consecutive triple ranks)
- Bomb hierarchy: bigger size beats smaller; same size, higher rank beats; joker bomb (4 jokers, encoded as `rank=15`) tops everything
- Round ends when one player empties their hand → that player's TEAM wins

## What's NOT modelled (caller must abstract these away)

- 级牌 wildcard (red-heart current-level card substituting any non-joker card)
- 进贡 / 还贡 (tribute) and multi-round level-up
- Variable-length sequences (only the fixed lengths above)
- Suits

If your real game has these, **strip them before calling, and re-apply on the way out** (e.g. ignore wildcards, treat tribute outside the model).

## Card type encoding

| idx | name | description |
|---|---|---|
| 0..10 | 3 4 5 6 7 8 9 T J Q K | non-honour ranks, in ascending power |
| 11 | A | ace |
| 12 | 2 | "level" rank — highest non-joker (no wildcard logic) |
| 13 | sj | small joker |
| 14 | bj | big joker |
| 15 | (sentinel) | JOKER BOMB rank — only valid as `rank` of a `BOMB` move |

A 2-deck game has `[8] * 13 + [2, 2]` = 108 cards.

## HTTP protocol

Server: `POST /move` with JSON body, returns JSON.

### Request body

```json
{
  "cur": 0,
  "hand":        [1,2,1,0,0,2,0,0,0,1,0,0,0,0,1],
  "hand_sizes":  [27, 20, 27, 22],
  "last": {
    "combo": 1,
    "rank": 5,
    "pair_rank": 0,
    "count": 1
  },
  "last_player": 3,
  "played":      [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
  ]
}
```

Field reference:

| field | type | description |
|---|---|---|
| `cur` | int 0..3 | seat index of the player to move |
| `hand` | int[15] | counts of each card type in `cur`'s hand |
| `hand_sizes` | int[4] | total cards remaining in each player's hand |
| `last` | object or `null` | the play we have to beat. `null` = free play (must play a non-pass move) |
| `last.combo` | int 0..8 | combo type of the last play (`0=PASS` only on free play, normally `1..8`) |
| `last.rank` | int 0..15 | starting/primary rank of the last play (for FULLHOUSE this is the **triple's** rank; for sequences this is the **start**; `15` only for joker bomb) |
| `last.pair_rank` | int 0..14 | only used by FULLHOUSE — the rank of the pair |
| `last.count` | int | number of cards in the last play |
| `last_player` | int 0..3 or `null` | seat of the player who actually made `last` (used by some rule logic; pass when not applicable) |
| `played` | int[4][15] | for each seat, count vector of cards already played that round |

### Response body

```json
{
  "move": {
    "combo": 5,
    "combo_name": "bomb",
    "rank": 8,
    "rank_name": "J",
    "pair_rank": 0,
    "count": 4,
    "human": "BOMB(Jx4)"
  },
  "top_k": [
    {"q": 0.83, "combo": 5, ... "human": "BOMB(Jx4)"},
    {"q": 0.41, "combo": 0, "combo_name": "pass", ... "human": "PASS"},
    ...
  ]
}
```

- `move`: the chosen move (greedy = `argmax Q(state, action)` over all legal moves)
- `top_k`: top-K moves with their Q values, for debugging / human-in-the-loop UI

`PASS` = `{combo: 0, rank: 0, pair_rank: 0, count: 0}`.

### Error handling

- `400` — malformed input (missing fields, wrong shapes)
- `200` with `{"move": null, "reason": "no legal moves"}` — should never happen in a well-formed turn (caller has at least one legal move including PASS in non-free play)

## Translating from your game

Pseudocode for the wrapper your team writes:

```python
def to_model_state(my_hand, last_move, hand_sizes, played, my_seat, last_player):
    # 1. Bucket cards into 15 type indices, ignoring suits.
    hand = np.zeros(15, dtype=np.int8)
    for card in my_hand:
        hand[card_to_type_idx(card)] += 1
    # 2. Convert last_move from your in-game representation.
    last = None if last_move is None else {
        'combo': COMBO_MAP[last_move.combo_type],
        'rank': RANK_MAP[last_move.primary_rank],
        'pair_rank': RANK_MAP.get(last_move.pair_rank, 0),
        'count': last_move.num_cards,
    }
    # 3. Same for played piles, hand sizes.
    return {...}

def apply_move(model_move, my_hand_with_suits):
    # The model returns counts/types; pick concrete cards from your hand to play.
    # For PASS: do nothing. For BOMB(rank=11, count=5): take 5 cards of rank A.
    ...
```

## Caveats / honest expectations

- The model trained on **simplified Guandan** and was verified to score **~91% vs the in-repo rule_agent baseline**, ~83% vs random. **It has NOT been validated against humans or against full-rules engines**.
- If your game has rule features the model can't see (wildcards, tribute, suits-matter), the model will be **strictly weaker** than its training-time score suggests.
- The model is non-stochastic by default (greedy argmax). Optional: pass `epsilon` in body for exploration.
- Latency target: ~5-15 ms per call on a CPU host (single env), negligible on GPU.

## Run the server

```bash
# on the GPU host
cd ~/guandan
python serve.py --ckpt runs/v3/latest.pt --port 8765

# health check
curl http://HOST:8765/health
# ⇒ {"ok": true, "device": "cuda"}
```

Files needed alongside `serve.py`:
- `env.py` `features.py` `model.py` `agent.py` `rule_agent.py`
- the checkpoint `.pt` file
- python ≥3.10, `torch`, `numpy` (no `flask` etc — uses stdlib `http.server`)
