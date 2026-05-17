"""High-level Python API for the bot.

Usage:
    from bot import GuandanBot
    bot = GuandanBot('runs/v3/latest.pt')
    move = bot.choose_move(
        cur=0,
        hand=[0,0,2,1,0,0,1,2,4,0,2,1,0,1,0],
        hand_sizes=[14,18,22,8],
        last={'combo': 1, 'rank': 5, 'pair_rank': 0, 'count': 1},
        last_player=3,
        played=[[...]*15]*4,
    )
    # move is dict: {'combo', 'rank', 'pair_rank', 'count', 'human'}
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch

from model import QNet
from features import encode_state, encode_action
from env import legal_moves, NUM_TYPES, COMBO_NAMES, RANK_NAMES, move_str


class GuandanBot:
    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.qnet = QNet().to(self.device)
        sd = torch.load(ckpt_path, map_location=self.device)
        self.qnet.load_state_dict(sd['state_dict'] if 'state_dict' in sd else sd)
        self.qnet.eval()

    @torch.no_grad()
    def choose_move(self,
                    cur: int,
                    hand,
                    hand_sizes,
                    played,
                    last: Optional[dict] = None,
                    last_player: Optional[int] = None,
                    return_top_k: int = 0) -> dict:
        hand_v = np.asarray(hand, dtype=np.int8)
        if hand_v.shape != (NUM_TYPES,):
            raise ValueError(f"hand must be length {NUM_TYPES}; got {hand_v.shape}")
        last_tuple = None
        if last is not None:
            last_tuple = (
                int(last['combo']), int(last['rank']),
                int(last.get('pair_rank', 0)), int(last['count']),
            )
        obs = {
            'cur': int(cur),
            'hand': hand_v,
            'last': last_tuple,
            'last_player': last_player,
            'hand_sizes': np.asarray(hand_sizes, dtype=np.int32),
            'played': [np.asarray(p, dtype=np.int32) for p in played],
        }
        legal = legal_moves(obs['hand'], obs['last'])
        if not legal:
            return {'move': None, 'top_k': [], 'reason': 'no legal moves'}

        s = encode_state(obs)
        s_t = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
        s_t = s_t.expand(len(legal), -1)
        a_t = torch.from_numpy(np.stack([encode_action(m) for m in legal])).float().to(self.device)
        q = self.qnet(s_t, a_t).cpu().numpy()
        order = np.argsort(-q)
        best = legal[int(order[0])]

        def to_dict(m):
            return {
                'combo': int(m[0]), 'combo_name': COMBO_NAMES[m[0]],
                'rank': int(m[1]), 'rank_name': RANK_NAMES[m[1]] if m[1] < len(RANK_NAMES) else 'JOKER_BOMB',
                'pair_rank': int(m[2]), 'count': int(m[3]),
                'human': move_str(m),
            }

        result = {'move': to_dict(best)}
        if return_top_k:
            result['top_k'] = [{'q': float(q[i]), **to_dict(legal[int(i)])}
                               for i in order[:return_top_k]]
        return result
