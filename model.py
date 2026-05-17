"""Q-network with optional belief head.

Structurally compatible with the v4/v5/v6 single-Sequential layout (keys
prefixed `net.`) so older checkpoints load cleanly. When `belief_dim > 0`, an
auxiliary `belief_head` is attached to the trunk output and trained with an
MSE auxiliary loss in v7.
"""
import torch
import torch.nn as nn

from features import STATE_DIM, ACTION_DIM


class QNet(nn.Module):
    def __init__(self,
                 state_dim: int = STATE_DIM,
                 action_dim: int = ACTION_DIM,
                 hidden: int = 1024,
                 num_layers: int = 4,
                 belief_dim: int = 0):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            in_dim = hidden
        # Final Linear is the Q head — keep inside self.net for back-compat with v6 ckpts.
        self.net = nn.Sequential(*layers, nn.Linear(in_dim, 1))
        self.belief_dim = belief_dim
        if belief_dim > 0:
            self.belief_head = nn.Linear(in_dim, belief_dim)
        else:
            self.belief_head = None

    def forward(self, s: torch.Tensor, a: torch.Tensor, return_belief: bool = False):
        x = torch.cat([s, a], dim=-1)
        if return_belief and self.belief_head is not None:
            # Trunk = all but final Linear in self.net
            trunk_x = x
            for layer in list(self.net)[:-1]:
                trunk_x = layer(trunk_x)
            q = self.net[-1](trunk_x).squeeze(-1)
            b = self.belief_head(trunk_x)
            return q, b
        return self.net(x).squeeze(-1)


def build_from_ckpt_args(ckpt_args: dict | None, default_hidden: int = 1024,
                         default_layers: int = 4, default_belief: int = 0) -> QNet:
    """Instantiate a QNet sized for a checkpoint based on its saved args dict."""
    if ckpt_args is None:
        return QNet(hidden=default_hidden, num_layers=default_layers,
                    belief_dim=default_belief)
    return QNet(
        hidden=int(ckpt_args.get('hidden', default_hidden)),
        num_layers=int(ckpt_args.get('num_layers', default_layers)),
        belief_dim=int(ckpt_args.get('belief_dim', default_belief)),
    )
