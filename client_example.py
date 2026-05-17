"""Minimal example caller for the bot HTTP service.

Run server first:
    python serve.py --ckpt runs/v3/latest.pt --port 8765

Then run this:
    python client_example.py
"""
import json
import urllib.request

# Example: I'm player 0. Last play was P3 (right opponent) playing a single 8.
# My hand: a few singles, some pairs, one bomb of Js.
state = {
    "cur": 0,
    "hand": [
        0,  # 3
        0,  # 4
        2,  # 5x2
        1,  # 6
        0,  # 7
        0,  # 8
        1,  # 9
        2,  # T x2
        4,  # J x4  (a bomb!)
        0,  # Q
        2,  # K x2
        1,  # A
        0,  # 2
        1,  # sj
        0,  # bj
    ],
    "hand_sizes": [14, 18, 22, 8],   # right opp (P3) close to winning
    "last": {                         # P3 just played a single 8
        "combo": 1, "rank": 5, "pair_rank": 0, "count": 1
    },
    "last_player": 3,
    "played": [
        [1,1,0,1,2,0,1,0,0,2,0,1,0,0,0],   # me already played a few
        [0,0,1,1,0,1,1,1,0,0,1,0,0,0,0],   # P1 (teammate)
        [1,2,0,0,1,1,0,0,1,1,0,0,0,0,0],   # P2 (left opp)
        [0,0,0,0,1,1,1,0,0,1,2,0,0,0,1],   # P3 (right opp, low cards)
    ],
}

req = urllib.request.Request(
    "http://localhost:18765/move",
    data=json.dumps(state).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=10) as resp:
    out = json.loads(resp.read())

print("chosen move:", out["move"]["human"])
print("\ntop-K considered (Q values):")
for cand in out["top_k"]:
    print(f"  Q={cand['q']:+.3f}  {cand['human']}")
