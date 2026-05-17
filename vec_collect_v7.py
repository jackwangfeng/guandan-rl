"""v7 collector: Match-driven self-play with belief-target capture.

Differences from v6 collector:
  - Each "slot" runs an entire match (GuandanMatch). When the current round ends,
    the slot transitions to the next round of the same match; when the match
    ends, the slot starts a fresh match. This produces post-tribute hands and
    varying level_rank values across slots — richer state distribution.
  - Each learner-decision captures a 45-dim belief target: the actual current
    hand counts (each /8) of the 3 other players, in canonical order:
    [teammate, opp_l, opp_r].
  - Per-round reward ±1 (same as v6). Match-level reward could be added behind
    a flag in train_v7.

Returns the list of completed rounds (each (traj, winner_team, steps, learner_seats)
where each traj entry is (player, state, action_feat, belief_target)). The training
loop assigns ±1 reward by winner_team and stores into the (s, a, g, belief) buffer.
"""
from __future__ import annotations
import random
from typing import Callable, Optional
import numpy as np
import torch

from env import GuandanEnv, NUM_TYPES
from features import encode_state, encode_action
from rule_agent import rule_choose
from match import GuandanMatch


def _belief_target(env: GuandanEnv, me: int) -> np.ndarray:
    """45-dim target: [teammate_hand, opp_l_hand, opp_r_hand] each (15,)/8."""
    teammate = (me + 2) % 4
    opp_l = (me + 1) % 4
    opp_r = (me + 3) % 4
    h_t = env.hands[teammate].astype(np.float32) / 8.0
    h_l = env.hands[opp_l].astype(np.float32) / 8.0
    h_r = env.hands[opp_r].astype(np.float32) / 8.0
    return np.concatenate([h_t, h_l, h_r])


def _start_match(slot_seed: int) -> tuple:
    m = GuandanMatch(seed=slot_seed)
    env = m.start_round()
    return m, env


def collect_matches(qnet,
                    device,
                    n_envs: int,
                    target_rounds: int,
                    epsilon: float,
                    mode_fn: Callable[[int, int], str],
                    qnet_for_mode: Optional[dict] = None,
                    seed_base: int = 0):
    """Run parallel matches; harvest finished ROUNDS until `target_rounds` completed.

    Returns:
      rounds: list of (traj, winner_team, steps, learner_seats)
              traj is list of (player, state_vec, action_feat, belief_target)
      matches_completed: how many matches were finished in this call
    """
    qnet_for_mode = qnet_for_mode or {}
    matches = []
    envs = []
    for i in range(n_envs):
        m, e = _start_match(seed_base + i)
        matches.append(m)
        envs.append(e)
    obs_list = [e.obs() for e in envs]
    trajs: list[list] = [[] for _ in range(n_envs)]  # per-slot current-round trajectory
    learner_seats: list[set] = [set() for _ in range(n_envs)]

    completed_rounds: list = []
    matches_completed = 0
    fresh_match_counter = 0

    safety = 0
    while len(completed_rounds) < target_rounds:
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

        _apply_net_learner(qnet, device, learner_items, envs, obs_list, trajs, epsilon)
        for key, items in net_groups.items():
            net = qnet_for_mode.get(key)
            if net is None:
                for env_i, s, legal in items:
                    obs_list[env_i], _, _, _ = envs[env_i].step(random.choice(legal))
                continue
            _apply_net_other(net, device, items, envs, obs_list)

        # Transition slots whose round just ended
        for i in range(n_envs):
            if envs[i].done:
                completed_rounds.append((trajs[i], envs[i].winner_team, envs[i].steps,
                                          set(learner_seats[i])))
                trajs[i] = []
                learner_seats[i] = set()
                matches[i].end_round(envs[i])
                if matches[i].match_done:
                    matches_completed += 1
                    fresh_match_counter += 1
                    matches[i], envs[i] = _start_match(seed_base + n_envs * 1000 +
                                                        fresh_match_counter)
                else:
                    envs[i] = matches[i].start_round()
                obs_list[i] = envs[i].obs()

    return completed_rounds, matches_completed


def _apply_net_learner(net, device, items, envs, obs_list, trajs, epsilon):
    if not items:
        return
    flat_s, flat_a, splits = [], [], []
    off = 0
    for env_i, s, legal, b in items:
        for m in legal:
            flat_s.append(s)
            flat_a.append(encode_action(m))
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
        a_feat = encode_action(m)
        trajs[env_i].append((env.cur, s, a_feat, b))
        obs_list[env_i], _, _, _ = env.step(m)


def _apply_net_other(net, device, items, envs, obs_list):
    if not items:
        return
    flat_s, flat_a, splits = [], [], []
    off = 0
    for env_i, s, legal in items:
        for m in legal:
            flat_s.append(s)
            flat_a.append(encode_action(m))
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
