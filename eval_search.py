"""Eval a checkpoint with optional PIMC search. Compares to random / rule baselines.

Usage:
  python eval_search.py --ckpt runs/v4/latest.pt --mode rollout --n-worlds 8 --games 200
"""
from __future__ import annotations
import argparse
import random
import time
import torch
import numpy as np

from model import QNet
from agent import pick_action
from env import GuandanEnv
from features import encode_state
from rule_agent import rule_choose
from search import pimc_choose


def play_one(env, qnet, device, opponent: str, learner_seat_team: int,
             search_mode: str, n_worlds: int, rng: random.Random):
    """Run one game; return 1 if learner team wins, 0 otherwise."""
    safety = 0
    obs = env.obs()
    while not env.done:
        safety += 1
        if safety > 10_000:
            break
        legal = env.legal()
        if not legal:
            break
        if env.cur % 2 == learner_seat_team:
            if search_mode == 'none':
                s = encode_state(obs)
                m = pick_action(qnet, s, legal, device, epsilon=0.0)
            else:
                m = pimc_choose(qnet, device, env, env.cur,
                                mode=search_mode, n_worlds=n_worlds,
                                rng=rng, legal=legal)
        else:
            if opponent == 'random':
                m = random.choice(legal)
            elif opponent == 'rule':
                m = rule_choose(env.hands[env.cur], env.last_play, env.last_player,
                                obs['hand_sizes'], env.cur)
                if m is None:
                    m = random.choice(legal)
            else:
                raise ValueError(opponent)
        obs, _, _, _ = env.step(m)
    return 1 if env.winner_team == learner_seat_team else 0


def winrate(qnet, device, opponent, n_games, search_mode, n_worlds, seed_base):
    rng = random.Random(seed_base + 12345)
    wins = 0
    total = 0
    for seat in (0, 1):
        for i in range(n_games // 2):
            env = GuandanEnv(seed=seed_base + seat * 10**6 + i)
            wins += play_one(env, qnet, device, opponent, seat,
                             search_mode, n_worlds, rng)
            total += 1
    return wins / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--mode', type=str, default='none',
                   choices=['none', '1ply', 'rollout'])
    p.add_argument('--n-worlds', type=int, default=8)
    p.add_argument('--games', type=int, default=200)
    p.add_argument('--opponents', type=str, default='random,rule',
                   help='comma-separated list of opponents')
    p.add_argument('--seed', type=int, default=10**7)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qnet = QNet().to(device)
    sd = torch.load(args.ckpt, map_location=device)['state_dict']
    qnet.load_state_dict(sd)
    qnet.eval()
    print(f"[eval] ckpt={args.ckpt} mode={args.mode} n_worlds={args.n_worlds} games={args.games}",
          flush=True)

    for opp in args.opponents.split(','):
        opp = opp.strip()
        t0 = time.time()
        wr = winrate(qnet, device, opp, args.games, args.mode, args.n_worlds, args.seed)
        dt = time.time() - t0
        print(f"  vs {opp:8s}: WR={wr:.3f}  ({args.games} games in {dt:.1f}s, "
              f"{dt/args.games*1000:.1f} ms/game)", flush=True)


if __name__ == '__main__':
    main()
