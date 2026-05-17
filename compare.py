"""Head-to-head + vs-baselines comparison across multiple checkpoints.

Usage:
  python compare.py --ckpts runs/v4/latest.pt runs/v5_baseline/latest.pt runs/v5_search/latest.pt
                    --names v4 baseline search --games 400
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


def load_qnet(path, device):
    ck = torch.load(path, map_location=device)
    qnet = build_from_ckpt_args(ck.get('args')).to(device)
    qnet.load_state_dict(ck['state_dict'])
    qnet.eval()
    return qnet


def play_one(env, net_a, net_b, device, opponent, learner_team, seed):
    """net_a controls `learner_team`, net_b (or `opponent` string) the other."""
    rng = random.Random(seed)
    safety = 0
    obs = env.obs()
    while not env.done:
        safety += 1
        if safety > 10_000:
            break
        legal = env.legal()
        if not legal:
            break
        team = env.cur % 2
        if team == learner_team:
            s = encode_state(obs)
            m = pick_action(net_a, s, legal, device, 0.0)
        else:
            if net_b is not None:
                s = encode_state(obs)
                m = pick_action(net_b, s, legal, device, 0.0)
            elif opponent == 'random':
                m = rng.choice(legal)
            elif opponent == 'rule':
                m = rule_choose(env.hands[env.cur], env.wildcards[env.cur],
                                env.level_rank, env.last_play, env.last_player,
                                obs['hand_sizes'], env.cur)
                if m is None:
                    m = rng.choice(legal)
            else:
                raise ValueError(opponent)
        obs, _, _, _ = env.step(m)
    return 1 if env.winner_team == learner_team else 0


def winrate(net_a, opponent_or_net, device, n_games, opponent_name=None, seed_base=2024):
    """net_a plays both teams alternately; returns WR averaged."""
    wins = 0
    total = 0
    net_b = opponent_or_net if isinstance(opponent_or_net, torch.nn.Module) else None
    opp_name = opponent_name if net_b is None else None
    for seat in (0, 1):
        for i in range(n_games // 2):
            env = GuandanEnv(seed=seed_base + seat * 10**6 + i)
            wins += play_one(env, net_a, net_b, device, opp_name, seat,
                             seed=seed_base + 50000 + i)
            total += 1
    return wins / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', nargs='+', required=True)
    p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--games', type=int, default=400)
    p.add_argument('--seed', type=int, default=2024)
    args = p.parse_args()
    assert len(args.ckpts) == len(args.names)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nets = [(n, load_qnet(c, device)) for n, c in zip(args.names, args.ckpts)]

    print(f"\n=== vs baselines ({args.games} games each) ===")
    print(f"{'model':>12} | {'vs random':>10} | {'vs rule':>10}")
    print('-' * 40)
    for name, net in nets:
        t0 = time.time()
        wr_r = winrate(net, 'random', device, args.games, 'random', args.seed)
        wr_u = winrate(net, 'rule', device, args.games, 'rule', args.seed + 1)
        print(f"{name:>12} | {wr_r*100:>9.1f}% | {wr_u*100:>9.1f}%  ({time.time()-t0:.1f}s)")

    if len(nets) >= 2:
        print(f"\n=== head-to-head (row beats column, {args.games} games) ===")
        # Header
        hdr = '            ' + ' '.join(f'{n:>10}' for n, _ in nets)
        print(hdr)
        for n_a, net_a in nets:
            row = [f"{n_a:>10}"]
            for n_b, net_b in nets:
                if n_a == n_b:
                    row.append(f"{'--':>10}")
                else:
                    wr = winrate(net_a, net_b, device, args.games, seed_base=args.seed + 100)
                    row.append(f"{wr*100:>9.1f}%")
            print('  ' + ' '.join(row))


if __name__ == '__main__':
    main()
