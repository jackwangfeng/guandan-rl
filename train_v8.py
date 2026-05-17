"""v8 training: real-rules env (suit-aware + flush) + bigger model + belief head.

Uses STATE_DIM / ACTION_DIM from features_v8 (different from v6/v7 because of
new suit features + flush combo).
"""
from __future__ import annotations
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from model import QNet, build_from_ckpt_args
from agent import Buffer, pick_action
from env_v8 import GuandanEnvV8, NUM_TYPES
from features_v8 import encode_state, encode_action, STATE_DIM, ACTION_DIM
from rule_agent_v8 import rule_choose
from vec_collect_v8 import collect_matches


BELIEF_DIM = 3 * NUM_TYPES  # 45


def make_qnet(args, device):
    return QNet(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                hidden=args.hidden, num_layers=args.num_layers,
                belief_dim=args.belief_dim).to(device)


def eval_vs(qnet, device, opponent: str, n_games: int = 400, seed_base: int = 10**7):
    qnet.eval()
    wins = 0
    total = 0
    for seat in (0, 1):
        for i in range(n_games // 2):
            env = GuandanEnvV8(seed=seed_base + seat * 10**6 + i)
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
                    m = pick_action_v8(qnet, s, legal, device, 0.0)
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


@torch.no_grad()
def pick_action_v8(qnet, state_vec, legal_actions, device, epsilon):
    """v8-specific pick_action that uses features_v8.encode_action."""
    if (epsilon > 0 and random.random() < epsilon) or len(legal_actions) == 1:
        return random.choice(legal_actions)
    s = torch.from_numpy(state_vec).float().to(device)
    s = s.unsqueeze(0).expand(len(legal_actions), -1)
    a = torch.from_numpy(np.stack([encode_action(m) for m in legal_actions])).float().to(device)
    q = qnet(s, a)
    return legal_actions[int(q.argmax().item())]


def assign_seats(n_envs, rule_p, anchor_p, league_p, league_keys, anchor_key, rng):
    out = {}
    for i in range(n_envs):
        learner_seat = rng.randrange(2)
        r = rng.random()
        if r < rule_p:
            opp = 'rule'
        elif r < rule_p + anchor_p and anchor_key:
            opp = anchor_key
        elif r < rule_p + anchor_p + league_p and league_keys:
            opp = rng.choice(league_keys)
        else:
            opp = 'learner'
        out[i] = {learner_seat: 'learner', 1 - learner_seat: opp}
    return out


def make_mode_fn(seat_assignment):
    def fn(env_idx, player_idx):
        return seat_assignment[env_idx][player_idx % 2]
    return fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--total-rounds', type=int, default=600_000)
    p.add_argument('--n-envs', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--train-steps-per-batch', type=int, default=8)
    p.add_argument('--epsilon-start', type=float, default=0.4)
    p.add_argument('--epsilon-end', type=float, default=0.03)
    p.add_argument('--epsilon-decay-rounds', type=int, default=120_000)
    p.add_argument('--buffer-cap', type=int, default=800_000)
    p.add_argument('--rule-p', type=float, default=0.35)
    p.add_argument('--league-p', type=float, default=0.25)
    p.add_argument('--anchor-ckpt', type=str, default='')
    p.add_argument('--anchor-p', type=float, default=0.20)
    p.add_argument('--league-add-every', type=int, default=20_000)
    p.add_argument('--league-max', type=int, default=8)
    p.add_argument('--hidden', type=int, default=1536)
    p.add_argument('--num-layers', type=int, default=6)
    p.add_argument('--belief-dim', type=int, default=BELIEF_DIM)
    p.add_argument('--belief-weight', type=float, default=0.1)
    p.add_argument('--collect-rounds-per-step', type=int, default=64)
    p.add_argument('--eval-every', type=int, default=10_000)
    p.add_argument('--eval-games', type=int, default=400)
    p.add_argument('--save-every', type=int, default=20_000)
    p.add_argument('--out', type=str, default='runs/v8')
    p.add_argument('--init', type=str, default='')
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
    log.writerow(['rounds', 'wallclock_s', 'epsilon', 'q_loss', 'belief_loss',
                  'wr_vs_random', 'wr_vs_rule', 'buffer_size', 'league_size',
                  'rounds_per_sec'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[init] device={device} torch={torch.__version__} "
          f"STATE_DIM={STATE_DIM} ACTION_DIM={ACTION_DIM} args={vars(args)}",
          flush=True)

    qnet = make_qnet(args, device)
    if args.init and Path(args.init).exists():
        ck = torch.load(args.init, map_location=device)
        try:
            qnet.load_state_dict(ck['state_dict'], strict=False)
            print(f"[init] warm-started from {args.init}", flush=True)
        except Exception as e:
            print(f"[init] WARN: {e}", flush=True)

    opt = torch.optim.Adam(qnet.parameters(), lr=args.lr)
    buf = Buffer(capacity=args.buffer_cap, belief_dim=args.belief_dim)

    league_nets: dict = {}
    anchor_key = None
    if args.anchor_ckpt and Path(args.anchor_ckpt).exists() and args.anchor_p > 0:
        ck_a = torch.load(args.anchor_ckpt, map_location=device)
        # The anchor may have different STATE/ACTION dims (v6/v7 vs v8) — won't load
        # unless dims match. We require anchor to be v8-shaped.
        anchor = QNet(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                       hidden=int(ck_a.get('args', {}).get('hidden', args.hidden)),
                       num_layers=int(ck_a.get('args', {}).get('num_layers', args.num_layers)),
                       belief_dim=int(ck_a.get('args', {}).get('belief_dim', 0))).to(device)
        try:
            anchor.load_state_dict(ck_a['state_dict'], strict=False)
            anchor.eval()
            for p_ in anchor.parameters():
                p_.requires_grad_(False)
            anchor_key = '_anchor'
            league_nets[anchor_key] = anchor
            print(f"[init] loaded anchor opponent from {args.anchor_ckpt}", flush=True)
        except Exception as e:
            print(f"[init] anchor load failed: {e}; continuing without anchor", flush=True)

    def add_league_snapshot(rd_marker):
        if len(league_nets) >= args.league_max:
            for k in sorted(league_nets.keys()):
                if k != anchor_key:
                    del league_nets[k]
                    break
        key = f"snap_rd{rd_marker:08d}"
        snap = make_qnet(args, device)
        snap.load_state_dict(qnet.state_dict())
        snap.eval()
        for p_ in snap.parameters():
            p_.requires_grad_(False)
        league_nets[key] = snap
        torch.save({'state_dict': snap.state_dict(), 'rounds': rd_marker,
                    'args': vars(args)},
                   league_dir / f"{key}.pt")

    t0 = time.time()
    q_losses: list = []
    b_losses: list = []
    total_rounds = 0
    last_league_add = 0
    last_eval = 0
    last_save = 0

    while total_rounds < args.total_rounds:
        frac = min(1.0, total_rounds / max(1, args.epsilon_decay_rounds))
        eps = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * frac
        league_keys = [k for k in league_nets.keys() if k != anchor_key]
        seat_assignment = assign_seats(args.n_envs, args.rule_p, args.anchor_p,
                                       args.league_p, league_keys, anchor_key, rng)
        mode_fn = make_mode_fn(seat_assignment)

        completed, _ = collect_matches(qnet, device, args.n_envs,
                                       args.collect_rounds_per_step, eps, mode_fn,
                                       qnet_for_mode=league_nets,
                                       seed_base=args.seed * 999 + total_rounds)
        for traj, winner, steps, lseat in completed:
            total_rounds += 1
            if winner is None:
                continue
            for player, s, a, b in traj:
                r = 1.0 if (player % 2) == winner else -1.0
                buf.add(s, a, r, b=b)

        if len(buf) >= args.batch:
            for _ in range(args.train_steps_per_batch):
                if args.belief_dim > 0:
                    s_b, a_b, g_b, b_b = buf.sample(args.batch)
                    s_t = torch.from_numpy(s_b).float().to(device)
                    a_t = torch.from_numpy(a_b).float().to(device)
                    g_t = torch.from_numpy(g_b).float().to(device)
                    b_t = torch.from_numpy(b_b).float().to(device)
                    pred_q, pred_b = qnet(s_t, a_t, return_belief=True)
                    q_loss = ((pred_q - g_t) ** 2).mean()
                    b_loss = ((pred_b - b_t) ** 2).mean()
                    loss = q_loss + args.belief_weight * b_loss
                    q_losses.append(float(q_loss.item()))
                    b_losses.append(float(b_loss.item()))
                else:
                    s_b, a_b, g_b = buf.sample(args.batch)
                    s_t = torch.from_numpy(s_b).float().to(device)
                    a_t = torch.from_numpy(a_b).float().to(device)
                    g_t = torch.from_numpy(g_b).float().to(device)
                    pred_q = qnet(s_t, a_t)
                    q_loss = ((pred_q - g_t) ** 2).mean()
                    loss = q_loss
                    q_losses.append(float(q_loss.item()))
                opt.zero_grad()
                loss.backward()
                opt.step()

        if total_rounds - last_league_add >= args.league_add_every:
            add_league_snapshot(total_rounds)
            last_league_add = total_rounds
            print(f"[league] add snapshot @rd={total_rounds}, league_size={len(league_nets)}",
                  flush=True)

        if total_rounds - last_eval >= args.eval_every or last_eval == 0:
            last_eval = total_rounds
            wr_random = eval_vs(qnet, device, 'random', args.eval_games)
            wr_rule = eval_vs(qnet, device, 'rule', args.eval_games)
            elapsed = time.time() - t0
            recent_q = float(np.mean(q_losses[-2000:])) if q_losses else float('nan')
            recent_b = float(np.mean(b_losses[-2000:])) if b_losses else float('nan')
            rps = total_rounds / max(1e-9, elapsed)
            print(
                f"[rd {total_rounds:7d}] {elapsed:7.1f}s eps={eps:.3f} "
                f"q={recent_q:.4f} b={recent_b:.4f} "
                f"vs_random={wr_random:.3f} vs_rule={wr_rule:.3f} "
                f"buf={len(buf)} league={len(league_nets)} {rps:.1f}rd/s",
                flush=True,
            )
            log.writerow([total_rounds, f"{elapsed:.2f}", f"{eps:.4f}",
                          f"{recent_q:.6f}", f"{recent_b:.6f}",
                          f"{wr_random:.4f}", f"{wr_rule:.4f}",
                          len(buf), len(league_nets), f"{rps:.2f}"])

        if total_rounds - last_save >= args.save_every:
            last_save = total_rounds
            torch.save({'state_dict': qnet.state_dict(), 'rounds': total_rounds,
                        'args': vars(args)},
                       out / f"ckpt_rd{total_rounds}.pt")
            torch.save({'state_dict': qnet.state_dict(), 'rounds': total_rounds,
                        'args': vars(args)},
                       out / 'latest.pt')

    log_f.close()
    print(f"[done] total {time.time() - t0:.1f}s, rounds={total_rounds}", flush=True)
    torch.save({'state_dict': qnet.state_dict(), 'rounds': total_rounds, 'args': vars(args)},
               out / 'final.pt')


if __name__ == '__main__':
    main()
