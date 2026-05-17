"""Online Guandan: 1 human vs 3 AI (v6 ckpt). HTTP server + static frontend.

Run:
    python serve_online.py --ckpt runs/v6/latest.pt --port 8088

Then open http://localhost:8088/ in browser.

Game design:
    - Human is seat 0; teammate seat 2 is AI; opponents seats 1, 3 are AI.
    - Single round (v6 env). After someone empties their hand, round ends.
    - "New game" deals fresh hand.
"""
from __future__ import annotations
import argparse
import json
import random
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch

from env import (GuandanEnv, NUM_TYPES, RANK_NAMES, JOKER_BOMB_RANK,
                 COMBO_NAMES, move_str, PASS, _apply_move_to_hand)
from features import encode_state, encode_action
from model import QNet, build_from_ckpt_args
from agent import pick_action
from rule_agent import rule_choose


HUMAN_SEAT = 0


def serialize_move(m: tuple, env: GuandanEnv | None = None) -> dict:
    combo, rank, pair_rank, count, n_wild = m
    out = {
        'combo': int(combo), 'rank': int(rank), 'pair_rank': int(pair_rank),
        'count': int(count), 'n_wild': int(n_wild),
        'combo_name': COMBO_NAMES.get(combo, '?'),
        'human': move_str(m),
    }
    if env is not None and combo != PASS:
        # Compute exact card consumption from the current hand for matching against user selection.
        hand = env.hands[env.cur]
        wild = env.wildcards[env.cur]
        try:
            new_hand, new_wild = _apply_move_to_hand(hand, wild, env.level_rank, m)
            consumed = (hand.astype(int) - new_hand.astype(int)).tolist()
            consumed_wild = int(wild - new_wild)
            out['consumed'] = consumed
            out['consumed_wild'] = consumed_wild
        except Exception:
            pass
    elif combo == PASS:
        out['consumed'] = [0] * NUM_TYPES
        out['consumed_wild'] = 0
    return out


def deserialize_move(d: dict) -> tuple:
    return (int(d['combo']), int(d['rank']), int(d['pair_rank']),
            int(d['count']), int(d['n_wild']))


def snapshot_state(env: GuandanEnv, human_seat: int, last_log=None):
    teammate = (human_seat + 2) % 4
    opp_l = (human_seat + 1) % 4
    opp_r = (human_seat + 3) % 4
    legal = env.legal() if (not env.done and env.cur == human_seat) else []
    return {
        'done': bool(env.done),
        'winner_team': int(env.winner_team) if env.winner_team is not None else None,
        'cur': int(env.cur),
        'human_seat': human_seat,
        'human_team': human_seat % 2,
        'is_human_turn': (not env.done and env.cur == human_seat),
        'level_rank': int(env.level_rank),
        'level_rank_name': RANK_NAMES[env.level_rank],
        # Hand info only for human
        'hand': env.hands[human_seat].astype(int).tolist(),
        'wildcards': int(env.wildcards[human_seat]),
        # Public info
        'hand_sizes': [int(h.sum()) for h in env.hands],
        'wildcards_left': 2 - sum(env.played_wild),
        'last': serialize_move(env.last_play) if env.last_play else None,
        'last_player': env.last_player,
        'played': [p.astype(int).tolist() for p in env.played],
        'played_wild': list(env.played_wild),
        'legal_moves': [serialize_move(m, env=env) for m in legal],
        'seat_labels': {
            str(human_seat): '你',
            str(teammate): '队友',
            str(opp_l): '左对手',
            str(opp_r): '右对手',
        },
        'rank_names': RANK_NAMES,
        'combo_names': COMBO_NAMES,
        'log': last_log or [],
    }


class GameSession:
    """One game's state, mutated under a lock."""
    def __init__(self, qnet, device, seed=None):
        self.qnet = qnet
        self.device = device
        self.lock = threading.Lock()
        self.env: GuandanEnv | None = None
        self.log: list = []
        self.new_game(seed=seed)

    def new_game(self, seed=None):
        with self.lock:
            self.env = GuandanEnv(seed=seed)
            self.log = []
            self._auto_play_ai()

    def _auto_play_ai(self):
        """Run AI players until env is done or it's human's turn."""
        env = self.env
        safety = 0
        while not env.done and env.cur != HUMAN_SEAT:
            safety += 1
            if safety > 1000:
                self.log.append({'type': 'error', 'text': 'AI loop guard tripped'})
                break
            legal = env.legal()
            if not legal:
                env.done = True
                break
            # AI policy: Q-net argmax (rule-agent style fallback if any error).
            m = self._ai_pick(env)
            self._step_and_log(m, ai=True)

    def _ai_pick(self, env):
        try:
            s = encode_state(env.obs())
            return pick_action(self.qnet, s, env.legal(), self.device, epsilon=0.0)
        except Exception:
            # fallback rule agent
            obs = env.obs()
            m = rule_choose(env.hands[env.cur], env.wildcards[env.cur],
                            env.level_rank, env.last_play, env.last_player,
                            obs['hand_sizes'], env.cur)
            return m if m is not None else random.choice(env.legal())

    def _step_and_log(self, m: tuple, ai: bool):
        env = self.env
        player = env.cur
        seat_tag = ('AI' if player != HUMAN_SEAT else '你')
        self.log.append({
            'type': 'play',
            'player': int(player),
            'seat_tag': seat_tag,
            'move': serialize_move(m),
            'after_hand_size': int(env.hands[player].sum()) - sum(m[3] if m[0] != 0 else 0 for _ in [0]),
        })
        env.step(m)
        if env.done:
            self.log.append({
                'type': 'end',
                'winner_team': int(env.winner_team) if env.winner_team is not None else None,
                'finish_order': list(env.finish_order),
            })

    def play_human(self, move: tuple):
        with self.lock:
            env = self.env
            if env.done:
                return self.snapshot('game already over')
            if env.cur != HUMAN_SEAT:
                return self.snapshot('not your turn')
            legal = env.legal()
            if move not in legal:
                return self.snapshot(f'illegal move {move_str(move)}')
            self._step_and_log(move, ai=False)
            self._auto_play_ai()
            return self.snapshot()

    def snapshot(self, error=None):
        s = snapshot_state(self.env, HUMAN_SEAT, last_log=self.log)
        if error:
            s['error'] = error
        return s


_session: GameSession | None = None


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False, default=_json_default).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, mime: str):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        root = Path(__file__).parent / 'online_guandan'
        if path == '/' or path == '/index.html':
            return self._send_file(root / 'index.html', 'text/html; charset=utf-8')
        if path == '/app.js':
            return self._send_file(root / 'app.js', 'application/javascript; charset=utf-8')
        if path == '/style.css':
            return self._send_file(root / 'style.css', 'text/css; charset=utf-8')
        if path == '/api/state':
            return self._send_json(_session.snapshot())
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        content_len = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(content_len) if content_len else b'{}'
        try:
            body = json.loads(raw)
        except Exception:
            body = {}

        if path == '/api/new':
            seed = body.get('seed')
            _session.new_game(seed=seed)
            return self._send_json(_session.snapshot())
        if path == '/api/play':
            try:
                mv = deserialize_move(body['move'])
            except Exception as e:
                return self._send_json({'error': f'bad move: {e}'}, 400)
            return self._send_json(_session.play_human(mv))
        self.send_error(404)

    def log_message(self, fmt, *args):
        pass  # quieter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='runs/v6/latest.pt')
    p.add_argument('--port', type=int, default=8088)
    p.add_argument('--host', type=str, default='0.0.0.0')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.ckpt, map_location=device)
    qnet = build_from_ckpt_args(ck.get('args')).to(device)
    qnet.load_state_dict(ck['state_dict'])
    qnet.eval()
    print(f"[serve] device={device} ckpt={args.ckpt}", flush=True)

    global _session
    _session = GameSession(qnet, device)

    srv = HTTPServer((args.host, args.port), Handler)
    print(f"[serve] listening on http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
