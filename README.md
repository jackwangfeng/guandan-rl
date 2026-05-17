# Mini-Guandan RL

Self-play DMC for 4-player Guandan (掼蛋), evolving through v4 → v8 as we
add rule fidelity, belief modelling, and multi-round structure.

```
v4/v5  →  v6/v7  →  v8
简化规则    级牌+万能牌   真规则(suit + 红心万能 + 同花顺)
```

Hardware: single **RTX A4000 (16GB)**, ~5h per generation at 600k rounds.

---

## Quick Start

```bash
# Smoke
python test_smoke.py        # v6/v7 env (level + generic wildcards)
python test_smoke_v8.py     # v8 env (suit-aware + flush)

# Train
python train_v6.py --out runs/v6 --total-episodes 300000
python train_v7.py --out runs/v7 --total-rounds 600000 --anchor-ckpt runs/v6/latest.pt
python train_v8.py --out runs/v8 --total-rounds 600000   # v7 anchor incompatible (env changed)

# Eval — single-round
python compare.py     --ckpts runs/v6/latest.pt runs/v7/latest.pt --names v6 v7 --games 400
python compare_v8.py  --ckpts runs/v8/latest.pt --names v8 --games 400

# Eval — full match (multi-round, including tribute + level-up)
python eval_match.py  --ckpt runs/v7/latest.pt --vs rule --matches 50

# Web app (1 human vs 3 AI), default serves v7 ckpt
python serve_online.py --ckpt runs/v7/latest.pt --port 8088
# → open http://localhost:8088/
```

---

## Final Results

### v6 / v7 — same env (simplified wildcards, no suit)

| Model | vs random | vs rule_agent | Head-to-head |
|---|---|---|---|
| v6 (300k eps, MLP 4×1024) | 84.2% | 93.8% | -- |
| **v7** (600k rd, MLP 6×1536 + belief head) | **90.2%** | 90.8% | **vs v6: 61% wins** |
| v7 vs v6 in **full match** (50 matches, avg 11 rounds) | -- | -- | **60% wins** |
| v6/v7 vs random/rule **in full match** | 100% | 100% | -- |

### v8 — real-rules env (suit + heart-only wildcards + flush)

Different env, not directly comparable to v6/v7 numbers.

| Eval | v8 score |
|---|---|
| vs random (single round) | 88.8% |
| vs rule_agent (single round) | 90.2% |
| vs rule_agent (full match, 50 matches) | **100%** |

**Honest expectation**: against amateur human players, v8 probably still loses
— see `What's NOT modelled` below. Against published / commercial Guandan bots,
unknown (we have no public competitor to test against).

---

## Versions & Env Evolution

| Version | Env | Model | Train budget | Notes |
|---|---|---|---|---|
| v4 | simplified (no level/wild) | MLP 4×1024 | 200k single rounds | original DMC |
| v5 | same | same | 200k continued | random-opponent + PIMC-search injection experiments — **both failed** |
| v6 | + level rank + generic wildcards (2/deck) | MLP 4×1024 | 300k single rounds | rule_p=0.5 league_p=0.3 |
| **v7** | same as v6 | MLP **6×1536** + **belief head** | 600k rounds (Match-driven) | v6 anchor, 0.1 belief-loss weight |
| **v8** | **real rules**: 4 suits, red-heart-only wildcards, FLUSH_SEQ5 (同花顺) | same as v7 | 600k rounds | from scratch (env dims changed) |

Negative results worth remembering:

- **PIMC search at inference hurts** (vs rule WR 91.5% → 69-78%) — Q-net already
  near-optimal under self-play assumptions, search adds noise. See `search.py`.
- **Random-opponent injection hurts** — Q-net specializes to random's
  unpredictability at the expense of strong-opponent play. v5 confirmed this.
- **Pure DMC > search-augmented DMC** for this domain, matching DouZero's
  finding on Dou Dizhu.

---

## Combos modelled

```
v4-v7: pass / single / pair / triple / 3+2 / bomb / SEQ5 / PSEQ3 / TSEQ2
v8 也加: FLUSH_SEQ5 (同花顺, bomb-tier)
```

**Bomb hierarchy in v8** (high → low):
1. 王炸 (4 jokers)
2. 6+ 张同点炸弹
3. 同花顺 (5-card flush straight)
4. 5 张同点炸弹
5. 4 张同点炸弹

Same-rank bombs: bigger size wins; same size, higher rank wins. Same-tier
flushes: higher start rank wins.

---

## What's NOT modelled

These have **non-zero strategic impact** but were deferred (effort vs payoff):

- **Wildcards in same-suit sequences** (red-heart flush + wildcard substitution) — v8 allows wildcards in singles/pairs/triples/full-house/bombs but NOT in flush. ~30 lines of env work to add.
- **抗贡** (anti-tribute, big-joker refusal) — rare, ~10 lines of `match_v8.py`.
- **Variable-length sequences** (SEQ6/7, PSEQ4/5, TSEQ3) — region-specific (江苏标准没有,某些北方变体有). Adds ~80 lines + new bomb-tier rules.
- **进贡 tie-breaking** when 4th place is ambiguous (rare).
- **还贡 restrictions** (must be ≤10, not jokers) — minor.

These are NOT major capability gaps; v8 covers ~95% of standard 江苏 rules.

What we don't model that **would matter** for human-level play:

- **Teammate signaling** through play order / timing (the core "tells" in real Guandan / Bridge). No RNN/Transformer on game history.
- **Multi-trick planning** ("控这家、催那家"). Q-net is essentially myopic.
- **Opponent profiling** at runtime. No online adaptation.

These are the gap between v8 and a real Guandan player. They need
architectural changes, not more rule patches.

---

## Files

```
env.py            v6/v7 env (level + generic 2-wildcard)
env_v8.py         v8 env (suit-aware + red-heart wildcards + 同花顺)
features.py       v6/v7 state/action encoding (STATE_DIM=123, ACTION_DIM=45)
features_v8.py    v8 encoding (STATE_DIM=188, ACTION_DIM=50, per-suit hand)
match.py          v6/v7 multi-round (tribute + level-up)
match_v8.py       v8 multi-round
rule_agent.py     v6/v7 heuristic baseline
rule_agent_v8.py  v8 baseline (handles 6-tuple moves)
model.py          QNet — backward-compat layout (`self.net` Sequential) + optional belief_head
agent.py          replay buffer (optional belief_dim) + pick_action
vec_collect.py    v6 vec collector
vec_collect_v7.py v7 (Match-driven + belief capture)
vec_collect_v8.py v8

train_v3.py       v4/v5 (legacy, simplified env)
train_v5.py       v5 experimental (random-opponent + search)
train_v6.py       v6 training
train_v7.py       v7 (bigger model + belief + anchor support)
train_v8.py       v8 (real rules)

eval.py           single-round eval, model vs random/model
eval_match.py     full-match eval (auto-instantiates QNet from ckpt args)
eval_search.py    v4 PIMC inference-time search (negative result)
compare.py        v6/v7 compare + head-to-head
compare_v8.py     v8 env compare

search.py         PIMC implementation (deprecated — doesn't help)

serve.py          v4-era HTTP API (legacy)
serve_online.py   v6/v7 web server — 1 human vs 3 AI

online_guandan/   web frontend (HTML + CSS + JS)
  index.html      QQ-game-style layout
  style.css       poker-card visuals, fanned hand, suits
  app.js          click-to-select cards, match selection to legal move

V8_DESIGN.md      v8 design doc
INTEGRATION.md    v4-era HTTP protocol (outdated; reference only)
test_smoke.py     v6/v7 env smoke tests
test_smoke_v8.py  v8 env smoke tests
```

---

## Web app

A simple 1-human-vs-3-AI interface, served by `serve_online.py`:

- Backend (Python stdlib http.server) loads a `.pt` ckpt and exposes `/api/{state,new,play}`
- Frontend (plain HTML+JS+CSS) shows your hand as visual poker cards,
  click-to-select; legal moves matched automatically by card selection
- Default ckpt: `runs/v7/latest.pt`. Swap to v6 or v8 by `--ckpt` (v8 needs
  the `serve_online.py` upgraded to use `env_v8`/`features_v8` — TODO)

```bash
python serve_online.py --ckpt runs/v7/latest.pt --port 8088
# then visit http://localhost:8088/
```

UI:
- Cards arranged at bottom in a fanned overlap
- Click a card to lift it; click again to deselect
- "出牌" commits when your selection matches a legal combo
- "不要" passes (when allowed); "新游戏" deals fresh

Cosmetic note: v6/v7 env has no suits, so the visual suits are assigned
deterministically per (rank, position) — **purely decorative**. v8's suits
are real and we'll surface them when serve_online switches to v8.

---

## Memory of what worked / what didn't

✅ **Worked**:
- DMC self-play with rule_agent + league + self-play mix (the standard recipe)
- Bigger model (1024→1536, 4→6 layers) — modest gain
- **Belief head as auxiliary loss** — clear gain, esp. vs_random
- Multi-round Match-driven training (post-tribute hands as state diversity)
- Anchor opponent (v6 ckpt as fixed reference) — speeds v7 convergence

❌ **Didn't work** (verified):
- PIMC search at inference time (hurt by 10-20%)
- PIMC injection during self-play (5-10% slowdown, no gain)
- Random opponents in training mix (hurt vs_random somehow)

🤔 **Unverified**:
- Recurrent state encoder for play history
- Belief-conditioned policy (use belief output to bias action selection)
- Adversarial league (a la AlphaStar)
- Self-distillation between successive checkpoints
