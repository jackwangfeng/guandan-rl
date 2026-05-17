"""PIMC (Perfect-Information Monte Carlo) search for Mini-Guandan.

Two variants:
  - `pimc_1ply`     : sample N worlds, apply candidate action, evaluate the next
                      state by Q-net argmax (fast; one batched GPU call per call)
  - `pimc_rollout`  : sample N worlds, full greedy rollout to terminal with Q-net
                      driving all 4 seats (slower but accounts for whole game)

Both convert from acting-player's perspective to my-team perspective.
"""
from __future__ import annotations
import random
from typing import Optional
import numpy as np
import torch

from env import GuandanEnv, NUM_TYPES, DECK_COUNTS
from features import encode_state, encode_action


_DECK = np.array(DECK_COUNTS, dtype=np.int32)


def snapshot_env(env: GuandanEnv) -> dict:
    """Capture mutable state of the env for cheap restoration in rollouts."""
    return {
        'played': [p.copy() for p in env.played],
        'cur': env.cur,
        'last_play': env.last_play,
        'last_player': env.last_player,
        'passes_in_a_row': env.passes_in_a_row,
    }


def restore_env(snapshot: dict, sampled_hands: list) -> GuandanEnv:
    """Build a fresh GuandanEnv with given public state + (sampled) hands.

    `sampled_hands` is a length-4 list of (15,) int8 arrays.
    """
    e = GuandanEnv.__new__(GuandanEnv)
    e.rng = random.Random()
    e.hands = [h.copy() for h in sampled_hands]
    e.played = [p.copy() for p in snapshot['played']]
    e.cur = snapshot['cur']
    e.last_play = snapshot['last_play']
    e.last_player = snapshot['last_player']
    e.passes_in_a_row = snapshot['passes_in_a_row']
    e.done = False
    e.winner_team = None
    e.steps = 0
    return e


def sample_hidden_hands(my_hand: np.ndarray,
                        played: list,
                        hand_sizes: np.ndarray,
                        my_idx: int,
                        rng: random.Random) -> list:
    """Sample a consistent assignment of the unseen cards to the 3 other seats.

    Public constraints:
      - deck composition (DECK_COUNTS) is fixed
      - my own hand and every seat's already-played pile are known
      - each other seat's *count* of remaining cards is known
    What's hidden: the *type* breakdown of each opponent / teammate hand.

    We draw uniformly from the multiset of unseen cards (i.e. multivariate
    hypergeometric), partitioned into the declared sizes.
    """
    total_played = np.zeros(NUM_TYPES, dtype=np.int32)
    for p in played:
        total_played += p.astype(np.int32)
    unseen = _DECK - my_hand.astype(np.int32) - total_played

    # build multiset
    cards: list = []
    for t in range(NUM_TYPES):
        c = int(unseen[t])
        if c > 0:
            cards.extend([t] * c)
    rng.shuffle(cards)

    hands = [None] * 4
    hands[my_idx] = my_hand.astype(np.int8).copy()
    cursor = 0
    for i in range(4):
        if i == my_idx:
            continue
        n = int(hand_sizes[i])
        h = np.zeros(NUM_TYPES, dtype=np.int8)
        for _ in range(n):
            h[cards[cursor]] += 1
            cursor += 1
        hands[i] = h
    return hands


@torch.no_grad()
def _qnet_argmax_batched(qnet, device, state_vecs, legal_lists):
    """For a list of (state, legal_moves), return argmax-Q index per item.

    Batched: concatenate (state, action) pairs across all items into one GPU call.
    """
    flat_s, flat_a, splits = [], [], []
    off = 0
    for s, lg in zip(state_vecs, legal_lists):
        for m in lg:
            flat_s.append(s)
            flat_a.append(encode_action(m))
        splits.append((lg, off, off + len(lg)))
        off += len(lg)
    if not flat_s:
        return []
    s_t = torch.from_numpy(np.stack(flat_s)).float().to(device)
    a_t = torch.from_numpy(np.stack(flat_a)).float().to(device)
    q = qnet(s_t, a_t).cpu().numpy()
    out = []
    for lg, lo, hi in splits:
        out.append((lg[int(np.argmax(q[lo:hi]))], float(q[lo:hi].max())))
    return out


@torch.no_grad()
def pimc_1ply(qnet, device, env: GuandanEnv, my_idx: int,
              n_worlds: int = 16, rng: Optional[random.Random] = None,
              legal: Optional[list] = None) -> tuple:
    """1-ply PIMC: apply each candidate action in each sampled world, then
    take the next-state Q-net leaf value (max over next mover's legal moves)
    as the estimate. Returns (chosen_move, per_action_avg_value).
    """
    rng = rng or random
    if legal is None:
        legal = env.legal()
    n_actions = len(legal)
    if n_actions <= 1:
        return (legal[0] if legal else None), np.zeros(max(n_actions, 1))

    obs = env.obs()
    snapshot = snapshot_env(env)
    my_team = my_idx % 2

    # Direct terminal-after-action cases: (ai, value)
    direct_vals: list = []
    # Pending leaf evaluations: parallel arrays
    pending_ai: list = []  # action index
    pending_state: list = []  # encoded state
    pending_legal: list = []
    pending_team: list = []  # next acting team

    for _w in range(n_worlds):
        sampled = sample_hidden_hands(obs['hand'], obs['played'],
                                      obs['hand_sizes'], my_idx, rng)
        for ai, action in enumerate(legal):
            sim = restore_env(snapshot, sampled)
            sim.step(action)
            if sim.done:
                direct_vals.append((ai, 1.0 if sim.winner_team == my_team else -1.0))
                continue
            nlg = sim.legal()
            if not nlg:
                direct_vals.append((ai, 0.0))
                continue
            pending_ai.append(ai)
            pending_state.append(encode_state(sim.obs()))
            pending_legal.append(nlg)
            pending_team.append(sim.cur % 2)

    # Batched leaf Q
    leaf = _qnet_argmax_batched(qnet, device, pending_state, pending_legal)

    returns = np.zeros(n_actions, dtype=np.float64)
    counts = np.zeros(n_actions, dtype=np.int32)
    for (ai, val) in direct_vals:
        returns[ai] += val
        counts[ai] += 1
    for ai, team, (_, leaf_q) in zip(pending_ai, pending_team, leaf):
        # acting player picks their best move; convert sign for opponent team
        val = leaf_q if team == my_team else -leaf_q
        returns[ai] += val
        counts[ai] += 1

    avg = returns / np.maximum(counts, 1)
    return legal[int(np.argmax(avg))], avg


@torch.no_grad()
def pimc_rollout(qnet, device, env: GuandanEnv, my_idx: int,
                 n_worlds: int = 8, rng: Optional[random.Random] = None,
                 legal: Optional[list] = None, max_steps: int = 80) -> tuple:
    """Full PIMC rollout: for each (world, candidate action), greedy-rollout
    to terminal with Q-net driving all 4 seats; score = +1/-1 from my team's
    point of view, averaged. Returns (chosen_move, per_action_avg).

    Vectorised: at each rollout step, batch Q-net forward across all live rollouts.
    """
    rng = rng or random
    if legal is None:
        legal = env.legal()
    n_actions = len(legal)
    if n_actions <= 1:
        return (legal[0] if legal else None), np.zeros(max(n_actions, 1))

    obs = env.obs()
    snapshot = snapshot_env(env)
    my_team = my_idx % 2

    # Spawn W * n_actions rollouts.
    rollouts: list = []
    rollout_ai: list = []
    for _w in range(n_worlds):
        sampled = sample_hidden_hands(obs['hand'], obs['played'],
                                      obs['hand_sizes'], my_idx, rng)
        for ai, action in enumerate(legal):
            sim = restore_env(snapshot, sampled)
            sim.step(action)
            rollouts.append(sim)
            rollout_ai.append(ai)

    for _step in range(max_steps):
        active_idx: list = []
        states: list = []
        legals: list = []
        for i, r in enumerate(rollouts):
            if r.done:
                continue
            lg = r.legal()
            if not lg:
                r.done = True
                continue
            active_idx.append(i)
            states.append(encode_state(r.obs()))
            legals.append(lg)
        if not active_idx:
            break
        picks = _qnet_argmax_batched(qnet, device, states, legals)
        for i, (mv, _) in zip(active_idx, picks):
            rollouts[i].step(mv)

    returns = np.zeros(n_actions, dtype=np.float64)
    counts = np.zeros(n_actions, dtype=np.int32)
    for r, ai in zip(rollouts, rollout_ai):
        counts[ai] += 1
        if r.winner_team is None:
            continue  # max_steps cutoff — count as draw (0)
        returns[ai] += 1.0 if r.winner_team == my_team else -1.0

    avg = returns / np.maximum(counts, 1)
    return legal[int(np.argmax(avg))], avg


def pimc_choose(qnet, device, env: GuandanEnv, my_idx: int,
                mode: str = '1ply', n_worlds: int = 16,
                rng: Optional[random.Random] = None,
                legal: Optional[list] = None):
    """Convenience wrapper. mode in {'1ply', 'rollout'}."""
    if mode == 'rollout':
        mv, _ = pimc_rollout(qnet, device, env, my_idx, n_worlds=n_worlds,
                             rng=rng, legal=legal)
    else:
        mv, _ = pimc_1ply(qnet, device, env, my_idx, n_worlds=n_worlds,
                          rng=rng, legal=legal)
    return mv
