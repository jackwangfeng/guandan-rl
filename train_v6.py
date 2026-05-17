"""v6 training: same DMC loop as v3, but with the level/wildcard env (and the
v5 finding that random opponents *hurt* — so we drop them and stick with the
v3 mix: rule + league + self).

State/action dims change because of new env, so we train from scratch.
"""
from __future__ import annotations
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from model import QNet
from agent import Buffer, pick_action
from env import GuandanEnv
from features import encode_state, encode_action
from rule_agent import rule_choose
from vec_collect import collect_batch


def eval_vs(qnet, device, opponent: str, n_games: int = 400, seed_base: int = 10**7):
    qnet.eval()
    wins = 0
    total = 0
    for seat in (0, 1):
        for i in range(n_games // 2):
            env = GuandanEnv(seed=seed_base + seat * 10**6 + i)
            obs = env.obs()
            safety = 0
            while not env.done:
                safety += 1
                if safety > 10_000:
                    break
                legal = env.legal()
                if not legal:
                    break
                if env.cur % 2 == seat:
                    s = encode_state(obs)
                    m = pick_action(qnet, s, legal, device, 0.0)
                else:
                    if opponent == 'random':
                        m = random.choice(legal)
                    else:
                        m = rule_choose(env.hands[env.cur], env.wildcards[env.cur],
                                        env.level_rank, env.last_play, env.last_player,
                                        obs['hand_sizes'], env.cur)
                        if m is None:
                            m = random.choice(legal)
                obs, _, _, _ = env.step(m)
            total += 1
            if env.winner_team == seat:
                wins += 1
    qnet.train()
    return wins / total


def assign_seats(n_envs, rule_p, league_p, league_keys, rng):
    out = {}
    for i in range(n_envs):
        learner_seat = rng.randrange(2)
        r = rng.random()
        if r < rule_p:
            opp = 'rule'
        elif r < rule_p + league_p and league_keys:
            opp = rng.choice(league_keys)
        else:
            opp = 'learner'
        out[i] = {learner_seat: 'learner', 1 - learner_seat: opp}
    return out


def make_mode_fn(seat_assignment: dict):
    def fn(env_idx, player_idx):
        return seat_assignment[env_idx][player_idx % 2]
    return fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--total-episodes', type=int, default=300_000)
    p.add_argument('--n-envs', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--train-steps-per-batch', type=int, default=8)
    p.add_argument('--epsilon-start', type=float, default=0.4)
    p.add_argument('--epsilon-end', type=float, default=0.03)
    p.add_argument('--epsilon-decay-eps', type=int, default=60_000)
    p.add_argument('--buffer-cap', type=int, default=600_000)
    p.add_argument('--rule-p', type=float, default=0.5)
    p.add_argument('--league-p', type=float, default=0.3)
    p.add_argument('--league-add-every', type=int, default=10_000)
    p.add_argument('--league-max', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=5_000)
    p.add_argument('--eval-games', type=int, default=400)
    p.add_argument('--save-every', type=int, default=10_000)
    p.add_argument('--out', type=str, default='runs/v6')
    p.add_argument('--init', type=str, default='', help='warm-start from this checkpoint (must be v6-shape)')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    league_dir = out / 'league'
    league_dir.mkdir(exist_ok=True)
    log_f = open(out / 'train.log.csv', 'w', newline='', buffering=1)
    log = csv.writer(log_f)
    log.writerow(['episodes', 'wallclock_s', 'epsilon', 'recent_loss',
                  'wr_vs_random', 'wr_vs_rule', 'buffer_size', 'league_size', 'eps_per_sec'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[init] device={device} torch={torch.__version__} args={vars(args)}", flush=True)

    qnet = QNet().to(device)
    if args.init and Path(args.init).exists():
        sd = torch.load(args.init, map_location=device)['state_dict']
        try:
            qnet.load_state_dict(sd, strict=False)
            print(f"[init] warm-started from {args.init}", flush=True)
        except Exception as e:
            print(f"[init] WARN: {e}", flush=True)

    opt = torch.optim.Adam(qnet.parameters(), lr=args.lr)
    buf = Buffer(capacity=args.buffer_cap)

    league_nets: dict = {}

    def add_league_snapshot(ep_marker: int):
        if len(league_nets) >= args.league_max:
            oldest = sorted(league_nets.keys())[0]
            del league_nets[oldest]
        key = f"snap_ep{ep_marker:07d}"
        snap = QNet().to(device)
        snap.load_state_dict(qnet.state_dict())
        snap.eval()
        for p_ in snap.parameters():
            p_.requires_grad_(False)
        league_nets[key] = snap
        torch.save({'state_dict': snap.state_dict(), 'episode': ep_marker},
                   league_dir / f"{key}.pt")

    t0 = time.time()
    losses: list = []
    total_eps = 0
    last_league_add = 0
    last_eval = 0
    last_save = 0

    while total_eps < args.total_episodes:
        frac = min(1.0, total_eps / max(1, args.epsilon_decay_eps))
        eps = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * frac

        league_keys = list(league_nets.keys())
        seat_assignment = assign_seats(args.n_envs, args.rule_p, args.league_p, league_keys, rng)
        mode_fn = make_mode_fn(seat_assignment)

        results = collect_batch(qnet, device, args.n_envs, eps, mode_fn,
                                qnet_for_mode=league_nets,
                                seed_base=args.seed * 999 + total_eps)
        for traj, winner, steps, lseat in results:
            total_eps += 1
            if winner is None:
                continue
            for player, s, a in traj:
                r = 1.0 if (player % 2) == winner else -1.0
                buf.add(s, a, r)

        if len(buf) >= args.batch:
            for _ in range(args.train_steps_per_batch):
                s_b, a_b, g_b = buf.sample(args.batch)
                s_t = torch.from_numpy(s_b).float().to(device)
                a_t = torch.from_numpy(a_b).float().to(device)
                g_t = torch.from_numpy(g_b).float().to(device)
                pred = qnet(s_t, a_t)
                loss = ((pred - g_t) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

        if total_eps - last_league_add >= args.league_add_every:
            add_league_snapshot(total_eps)
            last_league_add = total_eps
            print(f"[league] add snapshot @ep={total_eps}, league_size={len(league_nets)}",
                  flush=True)

        if total_eps - last_eval >= args.eval_every or last_eval == 0:
            last_eval = total_eps
            wr_random = eval_vs(qnet, device, 'random', args.eval_games)
            wr_rule = eval_vs(qnet, device, 'rule', args.eval_games)
            elapsed = time.time() - t0
            recent_loss = float(np.mean(losses[-2000:])) if losses else float('nan')
            eps_per_sec = total_eps / max(1e-9, elapsed)
            print(
                f"[ep {total_eps:7d}] {elapsed:7.1f}s eps={eps:.3f} "
                f"loss={recent_loss:.4f} "
                f"vs_random={wr_random:.3f} vs_rule={wr_rule:.3f} "
                f"buf={len(buf)} league={len(league_nets)} {eps_per_sec:.1f}ep/s",
                flush=True,
            )
            log.writerow([total_eps, f"{elapsed:.2f}", f"{eps:.4f}",
                          f"{recent_loss:.6f}",
                          f"{wr_random:.4f}", f"{wr_rule:.4f}",
                          len(buf), len(league_nets), f"{eps_per_sec:.2f}"])

        if total_eps - last_save >= args.save_every:
            last_save = total_eps
            torch.save({'state_dict': qnet.state_dict(), 'episode': total_eps, 'args': vars(args)},
                       out / f"ckpt_ep{total_eps}.pt")
            torch.save({'state_dict': qnet.state_dict(), 'episode': total_eps, 'args': vars(args)},
                       out / 'latest.pt')

    log_f.close()
    print(f"[done] total {time.time() - t0:.1f}s, episodes={total_eps}", flush=True)
    torch.save({'state_dict': qnet.state_dict(), 'episode': total_eps, 'args': vars(args)},
               out / 'final.pt')


if __name__ == '__main__':
    main()
