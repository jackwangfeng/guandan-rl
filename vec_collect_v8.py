"""v8 collector: Match-driven self-play with belief targets, using v8 env."""
from __future__ import annotations
import random
from typing import Callable, Optional
import numpy as np
import torch

from env_v8 import GuandanEnvV8, NUM_TYPES, rank_totals
from features_v8 import encode_state, encode_action
from rule_agent_v8 import rule_choose
from match_v8 import GuandanMatchV8


def _belief_target(env: GuandanEnvV8, me: int) -> np.ndarray:
    teammate = (me + 2) % 4
    opp_l = (me + 1) % 4
    opp_r = (me + 3) % 4
    h_t = rank_totals(env.hands[teammate]).astype(np.float32) / 8.0
    h_l = rank_totals(env.hands[opp_l]).astype(np.float32) / 8.0
    h_r = rank_totals(env.hands[opp_r]).astype(np.float32) / 8.0
    return np.concatenate([h_t, h_l, h_r])


def _start_match(slot_seed: int):
    m = GuandanMatchV8(seed=slot_seed)
    env = m.start_round()
    return m, env


def collect_matches(qnet, device, n_envs: int, target_rounds: int, epsilon: float,
                    mode_fn: Callable[[int, int], str],
                    qnet_for_mode: Optional[dict] = None, seed_base: int = 0):
    qnet_for_mode = qnet_for_mode or {}
    matches, envs = [], []
    for i in range(n_envs):
        m, e = _start_match(seed_base + i)
        matches.append(m); envs.append(e)
    obs_list = [e.obs() for e in envs]
    trajs: list[list] = [[] for _ in range(n_envs)]
    learner_seats: list[set] = [set() for _ in range(n_envs)]
    completed: list = []
    matches_completed = 0
    fresh = 0
    safety = 0
    while len(completed) < target_rounds:
        safety += 1
        if safety > 200_000:
            break
        learner_items = []
        net_groups: dict[str, list] = {}
        for i, env in enumerate(envs):
            if env.done:
                continue
            mode = mode_fn(i, env.cur)
            legal = env.legal()
            if not legal:
                env.done = True
                continue
            if mode == 'learner':
                s = encode_state(obs_list[i])
                b = _belief_target(env, env.cur)
                learner_items.append((i, s, legal, b))
                learner_seats[i].add(env.cur % 2)
            elif mode == 'rule':
                m = rule_choose(env.hands[env.cur], env.wildcards[env.cur],
                                env.level_rank, env.last_play, env.last_player,
                                obs_list[i]['hand_sizes'], env.cur)
                if m is None:
                    m = random.choice(legal)
                obs_list[i], _, _, _ = env.step(m)
            elif mode == 'random':
                obs_list[i], _, _, _ = env.step(random.choice(legal))
            else:
                s = encode_state(obs_list[i])
                net_groups.setdefault(mode, []).append((i, s, legal))

        _apply_learner(qnet, device, learner_items, envs, obs_list, trajs, epsilon)
        for key, items in net_groups.items():
            net = qnet_for_mode.get(key)
            if net is None:
                for env_i, s, legal in items:
                    obs_list[env_i], _, _, _ = envs[env_i].step(random.choice(legal))
                continue
            _apply_other(net, device, items, envs, obs_list)

        for i in range(n_envs):
            if envs[i].done:
                completed.append((trajs[i], envs[i].winner_team, envs[i].steps,
                                  set(learner_seats[i])))
                trajs[i] = []
                learner_seats[i] = set()
                matches[i].end_round(envs[i])
                if matches[i].match_done:
                    matches_completed += 1
                    fresh += 1
                    matches[i], envs[i] = _start_match(seed_base + n_envs * 1000 + fresh)
                else:
                    envs[i] = matches[i].start_round()
                obs_list[i] = envs[i].obs()
    return completed, matches_completed


def _apply_learner(net, device, items, envs, obs_list, trajs, epsilon):
    if not items:
        return
    flat_s, flat_a, splits = [], [], []
    off = 0
    for env_i, s, legal, b in items:
        for m in legal:
            flat_s.append(s); flat_a.append(encode_action(m))
        splits.append((env_i, s, legal, b, off, off + len(legal)))
        off += len(legal)
    s_t = torch.from_numpy(np.stack(flat_s)).float().to(device)
    a_t = torch.from_numpy(np.stack(flat_a)).float().to(device)
    with torch.no_grad():
        q = net(s_t, a_t).cpu().numpy()
    for env_i, s, legal, b, lo, hi in splits:
        env = envs[env_i]
        if env.done:
            continue
        if epsilon > 0 and random.random() < epsilon:
            m = random.choice(legal)
        else:
            m = legal[int(np.argmax(q[lo:hi]))]
        trajs[env_i].append((env.cur, s, encode_action(m), b))
        obs_list[env_i], _, _, _ = env.step(m)


def _apply_other(net, device, items, envs, obs_list):
    if not items:
        return
    flat_s, flat_a, splits = [], [], []
    off = 0
    for env_i, s, legal in items:
        for m in legal:
            flat_s.append(s); flat_a.append(encode_action(m))
        splits.append((env_i, s, legal, off, off + len(legal)))
        off += len(legal)
    s_t = torch.from_numpy(np.stack(flat_s)).float().to(device)
    a_t = torch.from_numpy(np.stack(flat_a)).float().to(device)
    with torch.no_grad():
        q = net(s_t, a_t).cpu().numpy()
    for env_i, s, legal, lo, hi in splits:
        env = envs[env_i]
        if env.done:
            continue
        m = legal[int(np.argmax(q[lo:hi]))]
        obs_list[env_i], _, _, _ = env.step(m)
