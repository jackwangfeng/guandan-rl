"""HTTP inference service. POST /move with a JSON game state, get back the chosen move.

Run:
    python serve.py --ckpt runs/v3/latest.pt --port 8765

See INTEGRATION.md for protocol.
"""
from __future__ import annotations
import argparse
import hmac
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

API_TOKEN = os.environ.get('BOT_API_TOKEN', '').strip()

# Files exposed without auth so callers' AIs can self-onboard from a single URL.
DOCS_WHITELIST = {
    'INTEGRATION.md': 'text/markdown; charset=utf-8',
    'README.md': 'text/markdown; charset=utf-8',
    'env.py': 'text/x-python; charset=utf-8',
    'features.py': 'text/x-python; charset=utf-8',
    'model.py': 'text/x-python; charset=utf-8',
    'agent.py': 'text/x-python; charset=utf-8',
    'rule_agent.py': 'text/x-python; charset=utf-8',
    'bot.py': 'text/x-python; charset=utf-8',
    'client_example.py': 'text/x-python; charset=utf-8',
    'serve.py': 'text/x-python; charset=utf-8',
}
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import torch

from model import QNet
from agent import pick_action
from features import encode_state, encode_action
from env import legal_moves, NUM_TYPES, NUM_COMBO_TYPES, COMBO_NAMES, RANK_NAMES, move_str


def parse_state(body: dict) -> tuple[dict, list]:
    """Convert request JSON into our (obs, legal_moves) representation."""
    hand = np.asarray(body['hand'], dtype=np.int8)
    if hand.shape != (NUM_TYPES,):
        raise ValueError(f"hand must be length {NUM_TYPES}; got {hand.shape}")
    last = body.get('last')
    last_tuple = None
    if last is not None:
        last_tuple = (
            int(last['combo']),
            int(last['rank']),
            int(last.get('pair_rank', 0)),
            int(last['count']),
        )
    obs = {
        'cur': int(body['cur']),
        'hand': hand,
        'last': last_tuple,
        'last_player': body.get('last_player'),
        'hand_sizes': np.asarray(body['hand_sizes'], dtype=np.int32),
        'played': [np.asarray(p, dtype=np.int32) for p in body['played']],
    }
    legal = legal_moves(obs['hand'], obs['last'])
    return obs, legal


def serialize_move(m: tuple) -> dict:
    combo, rank, pair_rank, count = m
    return {
        'combo': int(combo),
        'combo_name': COMBO_NAMES[combo],
        'rank': int(rank),
        'rank_name': RANK_NAMES[rank] if rank < len(RANK_NAMES) else 'JOKER_BOMB',
        'pair_rank': int(pair_rank),
        'count': int(count),
        'human': move_str(m),
    }


def make_handler(qnet, device, top_k_n: int):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, payload: Any):
            buf = json.dumps(payload).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(buf)))
            self.end_headers()
            self.wfile.write(buf)

        def _authed(self) -> bool:
            if not API_TOKEN:
                return True
            hdr = self.headers.get('Authorization', '')
            if not hdr.startswith('Bearer '):
                return False
            return hmac.compare_digest(hdr[7:].strip(), API_TOKEN)

        def do_GET(self):
            if self.path == '/health':
                # /health stays open so monitoring works
                return self._json(200, {'ok': True, 'device': str(device),
                                        'auth': bool(API_TOKEN)})
            if self.path in ('/', '/index'):
                lines = ["Mini-Guandan Bot service", "",
                         "Docs (open, no auth):"]
                lines += [f"  /{n}" for n in sorted(DOCS_WHITELIST)]
                lines += ["",
                          "Endpoints:",
                          "  GET  /health   (open)",
                          "  POST /move     (Authorization: Bearer <token> required)",
                          ""]
                body = "\n".join(lines).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            # Doc / source file passthrough (open).
            if self.path.startswith('/') and self.path[1:] in DOCS_WHITELIST:
                name = self.path[1:]
                path = os.path.join(DOCS_DIR, name)
                if os.path.isfile(path):
                    with open(path, 'rb') as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', DOCS_WHITELIST[name])
                    self.send_header('Content-Length', str(len(data)))
                    self.send_header('Cache-Control', 'public, max-age=300')
                    self.end_headers()
                    self.wfile.write(data)
                    return
            if not self._authed():
                return self._json(401, {'error': 'unauthorized'})
            return self._json(404, {'error': 'not found'})

        def do_POST(self):
            if not self._authed():
                self.send_response(401)
                self.send_header('WWW-Authenticate', 'Bearer')
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"error":"unauthorized"}')
                return
            if self.path != '/move':
                return self._json(404, {'error': 'not found'})
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(length) or b'{}')
                obs, legal = parse_state(body)
            except Exception as e:
                return self._json(400, {'error': f'bad request: {e}'})

            if not legal:
                return self._json(200, {'move': None, 'top_k': [], 'reason': 'no legal moves'})

            s = encode_state(obs)
            s_t = torch.from_numpy(s).float().unsqueeze(0).to(device)
            s_t = s_t.expand(len(legal), -1)
            a_t = torch.from_numpy(np.stack([encode_action(m) for m in legal])).float().to(device)
            with torch.no_grad():
                q = qnet(s_t, a_t).cpu().numpy()
            order = np.argsort(-q)
            best = legal[int(order[0])]
            top = []
            for idx in order[:top_k_n]:
                top.append({'q': float(q[idx]), **serialize_move(legal[int(idx)])})
            return self._json(200, {'move': serialize_move(best), 'top_k': top})

        def log_message(self, fmt, *args):
            sys.stderr.write(f"[serve] {self.address_string()} - {fmt % args}\n")

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--top-k', type=int, default=5)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qnet = QNet().to(device)
    sd = torch.load(args.ckpt, map_location=device)
    qnet.load_state_dict(sd['state_dict'] if 'state_dict' in sd else sd)
    qnet.eval()
    print(f"[serve] loaded {args.ckpt} on {device}; NUM_COMBO_TYPES={NUM_COMBO_TYPES}", flush=True)

    Handler = make_handler(qnet, device, args.top_k)
    httpd = HTTPServer((args.host, args.port), Handler)
    print(f"[serve] listening on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == '__main__':
    main()
