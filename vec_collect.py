"""Vectorised self-play / mixed-opponent rollout for v6 (level/wildcard rules)."""
from __future__ import annotations
import random
from typing import Callable, Optional
import numpy as np
import torch

from env import GuandanEnv
from features import encode_state, encode_action
from rule_agent import rule_choose


def collect_batch(qnet,
                  device,
                  n_envs: int,
                  epsilon: float,
                  mode_fn: Callable[[int, int], str],
                  qnet_for_mode: Optional[dict] = None,
                  seed_base: int = 0):
    qnet_for_mode = qnet_for_mode or {}
    envs = [GuandanEnv(seed=seed_base + i) for i in range(n_envs)]
    obs_list = [e.obs() for e in envs]
    trajs: list[list] = [[] for _ in range(n_envs)]
    learner_seats: list[set] = [set() for _ in range(n_envs)]

    safety = 0
    while any(not e.done for e in envs):
        safety += 1
        if safety > 20_000:
            break

        learner_envs = []
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
                learner_envs.append((i, s, legal))
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

        _apply_net(qnet, device, learner_envs, envs, obs_list, trajs,
                   epsilon, collect=True)
        for key, items in net_groups.items():
            net = qnet_for_mode.get(key)
            if net is None:
                for env_i, s, legal in items:
                    obs_list[env_i], _, _, _ = envs[env_i].step(random.choice(legal))
                continue
            _apply_net(net, device, items, envs, obs_list, trajs,
                       epsilon=0.0, collect=False)

    return [(trajs[i], envs[i].winner_team, envs[i].steps, learner_seats[i])
            for i in range(n_envs)]


def _apply_net(net, device, items, envs, obs_list, trajs, epsilon: float, collect: bool):
    if not items:
        return
    flat_states, flat_actions, splits = [], [], []
    offset = 0
    for env_i, s, legal in items:
        for m in legal:
            flat_states.append(s)
            flat_actions.append(encode_action(m))
        splits.append((env_i, s, legal, offset, offset + len(legal)))
        offset += len(legal)

    s_t = torch.from_numpy(np.stack(flat_states)).float().to(device)
    a_t = torch.from_numpy(np.stack(flat_actions)).float().to(device)
    with torch.no_grad():
        q = net(s_t, a_t).cpu().numpy()

    for env_i, s, legal, lo, hi in splits:
        env = envs[env_i]
        if env.done:
            continue
        if epsilon > 0 and random.random() < epsilon:
            m = random.choice(legal)
        else:
            m = legal[int(np.argmax(q[lo:hi]))]
        if collect:
            trajs[env_i].append((env.cur, s, encode_action(m)))
        obs_list[env_i], _, _, _ = env.step(m)
