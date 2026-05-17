# Deployment notes

Quick reference for serving a trained model in a web app or via HTTP.

## Web app (1 human vs 3 AI)

```bash
# v6/v7 ckpt (default — current env: level + generic wildcards)
python serve_online.py --ckpt runs/v7/latest.pt --port 8088

# Then open http://<host>:8088/ in a browser.
```

Frontend talks to backend via `/api/{state,new,play}`:

- `GET  /api/state` → snapshot dict (current hand, last play, legal moves, etc.)
- `POST /api/new`   `{"seed": null|int}` → deal new round
- `POST /api/play`  `{"move": {combo, rank, pair_rank, count, n_wild}}` → human plays a move; backend auto-runs AI to next human turn

Each legal move ships with a `consumed[15]` + `consumed_wild` signature so the
frontend can match a user's card selection to a unique move.

## v8 (real-rules) serving — TODO

`serve_online.py` currently uses `env.py` (v6/v7 env). To serve a v8 ckpt:

1. Make a copy `serve_online_v8.py` that imports `env_v8`, `features_v8`,
   `rule_agent_v8`. Replace `import env as ...` lines accordingly.
2. Move tuples become 6-tuples `(combo, rank, pair_rank, count, n_wild, suit)`.
3. Serialise the suit field too.
4. Frontend `app.js` already handles the visual suit assignment; the only
   change needed is to honour `move.suit` for FLUSH_SEQ5 cards (and possibly
   highlight the suit-matched cards in selection).

Estimated effort: 1-2 hours mechanical port.

## What the model expects to see (caller checklist)

For v6/v7 (`env.py`):
- Game state must include `level_rank` (the current level rank index, 0..12).
- Hand is a 15-int count vector. Wildcards are a separate count (must equal `hand[level_rank]` ≥ wildcards, since wildcards live inside hand[level_rank]).
- Action tuple: `(combo, rank, pair_rank, count, n_wild)`.

For v8 (`env_v8.py`):
- Hand is a `(15, 4)` count matrix (rank × suit). Suit indexing: `SPADE=0, HEART=1, DIAMOND=2, CLUB=3`.
- Wildcards = `hand[level_rank, HEART]` exactly. (Real Guandan rule: only red-heart level cards are wildcards.)
- Action: `(combo, rank, pair_rank, count, n_wild, suit)`. `suit` is meaningful only for `FLUSH_SEQ5`.

## Real-game integration (translating from a full-rules engine)

If you have a real Guandan engine and want to use our v8 model:

1. **Suits**: keep suits as-is; v8 understands them.
2. **Wildcards**: identify the red-heart cards of the level rank in each player's hand. Pass them as separate "wildcards" count alongside the hand matrix.
3. **Variable-length sequences**: v8 only supports SEQ5/PSEQ3/TSEQ2. If your engine allows longer, either truncate to length-5 (lose some plays) or extend `env_v8.py` (see V8_DESIGN.md TODO list).
4. **抗贡**: pre-process the tribute step yourself; pass the post-tribute hands to v8 via `GuandanEnvV8(hands=..., wildcards=...)` reset.
5. **Multi-deck variants**: v8 assumes 2 decks (108 cards). 3+ decks would require expanding DECK_COUNTS.

## Compute / Latency

On RTX A4000 (16GB):
- Single Q-net forward (~10M params, batch ~50): ~3 ms (warm)
- Self-play env step (Python): ~1-2 ms
- HTTP roundtrip overhead (stdlib server, localhost): ~5 ms

Typical decision latency for the web app: **~20-50 ms per AI move**. Imperceptible to a human player.
