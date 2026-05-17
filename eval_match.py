"""Multi-round match evaluation.

Plays N full Guandan matches (start at level '2', play until someone reaches A).
Reports per-match win rate. This is the most "real-game" eval we can do.

Usage:
  python eval_match.py --ckpt runs/v6/latest.pt --vs rule --matches 50
  python eval_match.py --ckpt runs/v6/latest.pt --vs runs/v6/league/snap_ep0010016.pt --matches 50
"""
from __future__ import annotations
import argparse
import random
import time
import torch
import numpy as np

from model import QNet, build_from_ckpt_args
from agent import pick_action
from env import GuandanEnv
from features import encode_state
from rule_agent import rule_choose
from match import GuandanMatch


def make_actor(kind, qnet=None, device=None):
    """Returns a fn(env) -> move."""
    if kind == 'random':
        def f(env):
            legal = env.legal()
            return random.choice(legal) if legal else None
        return f
    if kind == 'rule':
        def f(env):
            obs = env.obs()
            m = rule_choose(env.hands[env.cur], env.wildcards[env.cur],
                            env.level_rank, env.last_play, env.last_player,
                            obs['hand_sizes'], env.cur)
            if m is None:
                legal = env.legal()
                m = random.choice(legal) if legal else None
            return m
        return f
    if kind == 'net':
        assert qnet is not None and device is not None
        def f(env):
            obs = env.obs()
            legal = env.legal()
            if not legal:
                return None
            s = encode_state(obs)
            return pick_action(qnet, s, legal, device, 0.0)
        return f
    raise ValueError(kind)


def play_match(actor_t0, actor_t1, seed):
    """Play one full match. team0 uses actor_t0, team1 uses actor_t1.
    Returns (winner_team, rounds_played, jump_history)."""
    m = GuandanMatch(seed=seed)
    safety = 0
    while not m.match_done:
        safety += 1
        if safety > 200:
            break
        env = m.start_round()
        inner_safety = 0
        while not env.done:
            inner_safety += 1
            if inner_safety > 5000:
                break
            actor = actor_t0 if env.cur % 2 == 0 else actor_t1
            move = actor(env)
            if move is None:
                break
            env.step(move)
        m.end_round(env)
    return m.winner_team, len(m.history), [h['jump'] for h in m.history]


def match_winrate(actor_t0, actor_t1, n_matches, seed_base=2024):
    """Play `n_matches` matches; team_0 uses actor_t0, team_1 uses actor_t1.
    Halves get swapped (alternate which side actor_t0 plays) for fairness.
    Returns the WR of actor_t0."""
    wins_a = 0
    total = 0
    rounds = []
    for i in range(n_matches // 2):
        # actor_t0 on team 0
        w, r, _ = play_match(actor_t0, actor_t1, seed=seed_base + i)
        wins_a += (w == 0)
        total += 1
        rounds.append(r)
    for i in range(n_matches // 2):
        # swap: actor_t0 on team 1
        w, r, _ = play_match(actor_t1, actor_t0, seed=seed_base + 10**6 + i)
        wins_a += (w == 1)
        total += 1
        rounds.append(r)
    return wins_a / max(total, 1), float(np.mean(rounds)) if rounds else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--vs', type=str, default='rule',
                   help='"random", "rule", or path to another ckpt')
    p.add_argument('--matches', type=int, default=50)
    p.add_argument('--seed', type=int, default=2024)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck_a = torch.load(args.ckpt, map_location=device)
    qnet_a = build_from_ckpt_args(ck_a.get('args')).to(device)
    qnet_a.load_state_dict(ck_a['state_dict'])
    qnet_a.eval()
    actor_a = make_actor('net', qnet_a, device)

    if args.vs == 'random':
        actor_b = make_actor('random')
        label = 'random'
    elif args.vs == 'rule':
        actor_b = make_actor('rule')
        label = 'rule'
    else:
        ck_b = torch.load(args.vs, map_location=device)
        qnet_b = build_from_ckpt_args(ck_b.get('args')).to(device)
        qnet_b.load_state_dict(ck_b['state_dict'])
        qnet_b.eval()
        actor_b = make_actor('net', qnet_b, device)
        label = args.vs

    t0 = time.time()
    wr, avg_rounds = match_winrate(actor_a, actor_b, args.matches, args.seed)
    print(f"[match-eval] {args.ckpt} vs {label}: "
          f"WR={wr*100:.1f}% over {args.matches} matches "
          f"(avg {avg_rounds:.1f} rounds/match, {time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
