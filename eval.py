"""Standalone eval: load a checkpoint and report winrate vs random / vs another checkpoint."""
from __future__ import annotations
import argparse
import random
import torch

from model import QNet
from agent import pick_action
from env import GuandanEnv
from features import encode_state


def play_match(qnet_a, qnet_b, device, n_games=400, seed_base=2 * 10**7):
    """Team 0 (P0+P2) uses qnet_a; Team 1 (P1+P3) uses qnet_b (or random if None)."""
    wins_a = 0
    for i in range(n_games):
        env = GuandanEnv(seed=seed_base + i)
        obs = env.obs()
        safety = 0
        while not env.done:
            safety += 1
            if safety > 10_000:
                break
            legal = env.legal()
            if not legal:
                break
            net = qnet_a if env.cur % 2 == 0 else qnet_b
            if net is None:
                m = random.choice(legal)
            else:
                s = encode_state(obs)
                m = pick_action(net, s, legal, device, epsilon=0.0)
            obs, _, _, _ = env.step(m)
        if env.winner_team == 0:
            wins_a += 1
    return wins_a / n_games


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--vs', type=str, default='random', help='"random" or path to another ckpt')
    p.add_argument('--games', type=int, default=400)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qnet_a = QNet().to(device)
    qnet_a.load_state_dict(torch.load(args.ckpt, map_location=device)['state_dict'])
    qnet_a.eval()

    if args.vs == 'random':
        qnet_b = None
        label = 'random'
    else:
        qnet_b = QNet().to(device)
        qnet_b.load_state_dict(torch.load(args.vs, map_location=device)['state_dict'])
        qnet_b.eval()
        label = args.vs

    wr = play_match(qnet_a, qnet_b, device, n_games=args.games)
    print(f"team0 ({args.ckpt}) vs team1 ({label}): winrate={wr:.3f} over {args.games} games")


if __name__ == '__main__':
    main()
