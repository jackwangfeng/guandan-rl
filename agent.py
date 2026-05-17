"""DMC-style replay buffer (with optional belief targets for v7) + pick_action."""
from __future__ import annotations
import random
import numpy as np
import torch

from env import GuandanEnv
from features import encode_state, encode_action


class Buffer:
    """Uniform replacement buffer. Stores (s, a, g) and optionally a belief target b."""

    def __init__(self, capacity: int = 200_000, belief_dim: int = 0):
        self.cap = capacity
        self.belief_dim = belief_dim
        self.s = np.zeros((capacity, 0), dtype=np.float32)
        self.a = np.zeros((capacity, 0), dtype=np.float32)
        self.g = np.zeros(capacity, dtype=np.float32)
        self.b = np.zeros((capacity, belief_dim), dtype=np.float32) if belief_dim > 0 else None
        self.n = 0
        self.idx = 0

    def _ensure(self, sd, ad):
        if self.s.shape[1] == 0:
            self.s = np.zeros((self.cap, sd), dtype=np.float32)
            self.a = np.zeros((self.cap, ad), dtype=np.float32)

    def add(self, s, a, g, b=None):
        self._ensure(s.shape[0], a.shape[0])
        i = self.idx
        self.s[i] = s
        self.a[i] = a
        self.g[i] = g
        if self.b is not None and b is not None:
            self.b[i] = b
        self.idx = (self.idx + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int):
        assert self.n >= batch
        idx = np.random.randint(0, self.n, size=batch)
        if self.b is None:
            return self.s[idx], self.a[idx], self.g[idx]
        return self.s[idx], self.a[idx], self.g[idx], self.b[idx]

    def __len__(self):
        return self.n


@torch.no_grad()
def pick_action(qnet, state_vec, legal_actions, device, epsilon: float):
    if (epsilon > 0 and random.random() < epsilon) or len(legal_actions) == 1:
        return random.choice(legal_actions)
    s = torch.from_numpy(state_vec).float().to(device)
    s = s.unsqueeze(0).expand(len(legal_actions), -1)
    a = torch.from_numpy(np.stack([encode_action(m) for m in legal_actions])).float().to(device)
    q = qnet(s, a)
    return legal_actions[int(q.argmax().item())]


def play_episode(qnet, device, epsilon: float, seed: int | None = None):
    """Self-play one round (single-round env). Returns (traj, winner_team, steps)."""
    env = GuandanEnv(seed=seed)
    obs = env.obs()
    traj = []
    safety = 0
    while not env.done:
        safety += 1
        if safety > 10_000:
            break
        legal = env.legal()
        if not legal:
            break
        s = encode_state(obs)
        m = pick_action(qnet, s, legal, device, epsilon)
        a = encode_action(m)
        traj.append((env.cur, s, a))
        obs, _, done, info = env.step(m)
    return traj, env.winner_team, env.steps
