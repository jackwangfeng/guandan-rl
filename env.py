"""Guandan environment with level/wildcard rules (v6 rule set).

Rules added over v4/v5:
  - Each round has a `level_rank` (the level the dealer team is climbing on).
  - Exactly 2 wildcards exist per round: the two "red-heart suit" cards of the
    `level_rank`. We don't track suits; we keep a per-player counter `wildcards`
    of how many wildcards each player currently holds. They live inside
    `hand[level_rank]` (a wildcard is physically a level-rank card).
  - Wildcards can SUBSTITUTE for any non-joker rank in singles/pairs/triples/
    fullhouse/bombs. Sequences (SEQ5 / PSEQ3 / TSEQ2) do NOT accept wildcards
    in this implementation (simplification — real Guandan allows it but rarely
    used). Combos using substitution must include at least one real card.
  - When `rank == level_rank`, the move auto-prefers regular level-rank cards;
    wildcards are consumed only if regular cards run out. The agent does not
    expose a choice between "spend wildcard" vs "save for substitution" on
    level-rank combos. (This is a deliberate simplification of the action space.)
  - Joker bomb (4 jokers) is unchanged and does not accept wildcards.

Move encoding: (combo, rank, pair_rank, count, n_wild)
  - n_wild = number of wildcards consumed BY SUBSTITUTION (i.e., only when
    `rank != level_rank`). For level-rank combos, n_wild is always 0 in the
    move tuple even though wildcards may still be physically consumed.
"""
from __future__ import annotations
import random
from typing import Optional
import numpy as np

NUM_TYPES = 15
RANK_NAMES = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', '2', 'sj', 'bj']
DECK_COUNTS = [8] * 13 + [2, 2]  # 13 non-honour ranks of 8 each + 2 small jokers + 2 big jokers
SJ, BJ = 13, 14
JOKER_BOMB_RANK = 15

# Combo type ids
PASS, SINGLE, PAIR, TRIPLE, FULLHOUSE, BOMB, SEQ5, PSEQ3, TSEQ2 = 0, 1, 2, 3, 4, 5, 6, 7, 8
NUM_COMBO_TYPES = 9
COMBO_NAMES = {
    0: 'pass', 1: 'single', 2: 'pair', 3: 'triple', 4: 'three+two', 5: 'bomb',
    6: 'seq5', 7: 'pair_seq3', 8: 'triple_seq2',
}

SEQ5_LEN = 5
PSEQ3_LEN = 3
TSEQ2_LEN = 2
SEQ_HIGH_RANK = 11  # A; ranks 12 ("2") and jokers cannot be in sequences

MAX_WILDCARDS = 2  # exactly two red-heart level-rank cards in a 2-deck game

# Move = (combo, rank, pair_rank, count, n_wild)
PASS_MOVE = (PASS, 0, 0, 0, 0)


def deal(rng: random.Random, level_rank: int):
    """Deal 27 cards to each of 4 players. Mark 2 random level-rank cards as wildcards.

    Returns (hands, wildcards). hands is list of 4 (15,) int8 arrays; wildcards is
    list of 4 ints (count of wildcards per player).
    """
    # Build the deck as a list of (rank, is_wild) tokens
    cards: list = []
    for r, c in enumerate(DECK_COUNTS):
        for _ in range(c):
            cards.append([r, False])
    # Mark 2 level-rank cards as wildcards
    lr_idx = [i for i, (r, _) in enumerate(cards) if r == level_rank]
    if len(lr_idx) >= MAX_WILDCARDS:
        for wi in rng.sample(lr_idx, MAX_WILDCARDS):
            cards[wi][1] = True
    rng.shuffle(cards)

    hands = [np.zeros(NUM_TYPES, dtype=np.int8) for _ in range(4)]
    wildcards = [0, 0, 0, 0]
    for j, (rank, is_wild) in enumerate(cards):
        player = j % 4
        hands[player][rank] += 1
        if is_wild:
            wildcards[player] += 1
    return hands, wildcards


def hand_str(h, w=0, level_rank=-1):
    parts = [f"{RANK_NAMES[i]}x{c}" for i, c in enumerate(h) if c]
    s = " ".join(parts) if parts else "(empty)"
    if w > 0 and 0 <= level_rank < NUM_TYPES:
        s += f"  [wild={w} of {RANK_NAMES[level_rank]}]"
    return s


def move_str(m, level_rank=-1):
    combo, rank, pair_rank, count, n_wild = m
    if combo == PASS:
        return "PASS"
    if combo == BOMB and rank == JOKER_BOMB_RANK:
        return "BOMB(JOKER)"
    tag = f"+{n_wild}w" if n_wild > 0 else ""
    if combo == FULLHOUSE:
        return f"3+2({RANK_NAMES[rank]}x3+{RANK_NAMES[pair_rank]}x2){tag}"
    if combo == BOMB:
        return f"BOMB({RANK_NAMES[rank]}x{count}){tag}"
    if combo == SEQ5:
        return f"SEQ5({RANK_NAMES[rank]}-{RANK_NAMES[rank + SEQ5_LEN - 1]})"
    if combo == PSEQ3:
        return f"PSEQ3({RANK_NAMES[rank]}-{RANK_NAMES[rank + PSEQ3_LEN - 1]})"
    if combo == TSEQ2:
        return f"TSEQ2({RANK_NAMES[rank]}-{RANK_NAMES[rank + TSEQ2_LEN - 1]})"
    return f"{COMBO_NAMES[combo].upper()}({RANK_NAMES[rank]}x{count}){tag}"


def _can_take(hand, wildcards, level_rank, rank, count_of_rank, n_wild):
    """Can we form `count_of_rank` copies of `rank` using `n_wild` wildcards?

    Returns True if hand has enough regular cards of `rank` AND enough wildcards.
    The (count_of_rank - n_wild) regular cards must exist; n_wild wildcards must
    exist; if rank == level_rank, no substitution makes sense — caller should
    pass n_wild=0 and check hand[level_rank] >= count_of_rank.
    """
    if rank == level_rank:
        # No substitution; need full count from hand[level_rank]
        return n_wild == 0 and hand[level_rank] >= count_of_rank
    # rank != level_rank: substitution
    real_needed = count_of_rank - n_wild
    if real_needed < 1:  # must have >= 1 real card in the substituted combo
        return False
    if hand[rank] < real_needed:
        return False
    if wildcards < n_wild:
        return False
    # Also: the wildcards live in hand[level_rank]; can't reuse them as level-rank
    # cards simultaneously — but since this combo doesn't touch level_rank,
    # we're fine.
    return True


def _add_singles(hand, wildcards, level_rank, last, moves):
    free = last is None or last[0] == PASS
    last_rank = -1 if free else last[1]
    last_combo = -1 if free else last[0]
    if not (free or last_combo == SINGLE):
        return
    for r in range(NUM_TYPES):
        if hand[r] >= 1 and (free or r > last_rank):
            moves.append((SINGLE, r, 0, 1, 0))
    # Wildcard played as level-rank (single). Only valid if there's a wildcard
    # AND hand[level_rank] - wildcards == 0 (no regular level-rank card to play
    # as the n_wild=0 variant — otherwise that variant exists via the loop).
    # We do NOT emit a duplicate "wildcard-as-level-rank single" if regular
    # level-rank singles are also available; the auto-prefer-regular rule applies.
    if wildcards >= 1 and 0 <= level_rank < NUM_TYPES:
        if hand[level_rank] - wildcards == 0:
            if free or level_rank > last_rank:
                # hand[level_rank] >= 1 already implied (since wildcards >= 1)
                moves.append((SINGLE, level_rank, 0, 1, 0))


def _add_pairs(hand, wildcards, level_rank, last, moves):
    free = last is None or last[0] == PASS
    last_rank = -1 if free else last[1]
    last_combo = -1 if free else last[0]
    if not (free or last_combo == PAIR):
        return
    for r in range(NUM_TYPES):
        if not (free or r > last_rank):
            continue
        if r == level_rank:
            if hand[r] >= 2:
                moves.append((PAIR, r, 0, 2, 0))
        else:
            # n_wild from 0 to 1 (need >= 1 real card)
            for nw in range(0, min(wildcards, 1) + 1):
                if _can_take(hand, wildcards, level_rank, r, 2, nw):
                    moves.append((PAIR, r, 0, 2, nw))


def _add_triples(hand, wildcards, level_rank, last, moves):
    free = last is None or last[0] == PASS
    last_rank = -1 if free else last[1]
    last_combo = -1 if free else last[0]
    if not (free or last_combo == TRIPLE):
        return
    for r in range(13):  # ranks 0..12 (no jokers)
        if not (free or r > last_rank):
            continue
        if r == level_rank:
            if hand[r] >= 3:
                moves.append((TRIPLE, r, 0, 3, 0))
        else:
            for nw in range(0, min(wildcards, 2) + 1):
                if _can_take(hand, wildcards, level_rank, r, 3, nw):
                    moves.append((TRIPLE, r, 0, 3, nw))


def _fh_split(hand, wildcards, level_rank, tr, pr, total_nw):
    """Find a feasible (nw_t, nw_p) split for FULLHOUSE.

    nw_t / nw_p = wildcards used FOR SUBSTITUTION (not when rank == level_rank).

    Returns (nw_t, nw_p) or None. Convention: maximise nw_t (wildcards on triple).

    Bookkeeping: physical wildcards may also get consumed when a part has
    rank == level_rank and there aren't enough regular level-rank cards. Those
    consumptions are accounted for here so the total wildcards used never
    exceeds `wildcards`, and the total level-rank cards used never exceeds
    `hand[level_rank]`.
    """
    H_L = int(hand[level_rank])
    W = int(wildcards)
    regular_L = H_L - W  # regular (non-wild) level-rank cards

    # Range of nw_t.
    if tr == level_rank:
        if H_L < 3:
            return None
        nw_t_options = [0]
    else:
        nw_t_min = max(0, 3 - int(hand[tr]))
        nw_t_max = min(2, total_nw)
        if nw_t_min > nw_t_max:
            return None
        nw_t_options = list(range(nw_t_max, nw_t_min - 1, -1))

    for nw_t in nw_t_options:
        nw_p = total_nw - nw_t
        if nw_p < 0:
            continue

        # Compute wildcards & level-rank cards consumed by the triple part.
        if tr == level_rank:
            wild_t = max(0, 3 - regular_L)  # forced if not enough regular L
            L_used_t = 3
        else:
            wild_t = nw_t
            L_used_t = nw_t  # nw_t wildcards live in hand[level_rank]

        # Pair part
        if pr == level_rank:
            if nw_p != 0:
                continue
            # pair uses 2 level-rank cards from what's left after the triple.
            remaining_L = H_L - L_used_t
            if remaining_L < 2:
                continue
            remaining_wild = W - wild_t
            remaining_regular_L = remaining_L - remaining_wild
            wild_p = max(0, 2 - remaining_regular_L)
            L_used_p = 2
        else:
            if nw_p > 1:
                continue
            real_p = 2 - nw_p
            if int(hand[pr]) < real_p:
                continue
            wild_p = nw_p
            L_used_p = nw_p

        if wild_t + wild_p > W:
            continue
        if L_used_t + L_used_p > H_L:
            continue
        return (nw_t, nw_p)
    return None


def _add_fullhouse(hand, wildcards, level_rank, last, moves):
    free = last is None or last[0] == PASS
    last_rank = -1 if free else last[1]
    last_combo = -1 if free else last[0]
    if not (free or last_combo == FULLHOUSE):
        return
    # triple-rank tr (0..12), pair-rank pr (0..14, != tr).
    for tr in range(13):
        if not (free or tr > last_rank):
            continue
        for pr in range(NUM_TYPES):
            if pr == tr:
                continue
            for total_nw in range(0, min(wildcards, 3) + 1):
                if _fh_split(hand, wildcards, level_rank, tr, pr, total_nw) is not None:
                    moves.append((FULLHOUSE, tr, pr, 5, total_nw))


def _add_bombs(hand, wildcards, level_rank, last, moves):
    """Bombs: ≥4 of one rank, wildcards can substitute (need ≥1 real card).
    Joker bomb (4 jokers) is unchanged, no wildcards.
    """
    free = last is None or last[0] == PASS
    last_is_bomb = (not free) and last[0] == BOMB
    last_rank = last[1] if not free else -1
    last_count = last[3] if not free else 0

    have_joker_bomb = hand[SJ] >= 2 and hand[BJ] >= 2

    for r in range(13):  # ranks 0..12
        max_real = int(hand[r] if r != level_rank else hand[r])  # both same
        # Total bomb size c = real + n_wild, where:
        #  - if r == level_rank: n_wild=0, c = hand[level_rank]
        #  - if r != level_rank: real = hand[r] - 0 (regular cards of r), n_wild ∈ [0, wildcards]
        if r == level_rank:
            c_options = [hand[r]] if hand[r] >= 4 else []
        else:
            real = hand[r]
            c_options = []
            for nw in range(0, wildcards + 1):
                c = real + nw
                if c >= 4 and (nw == 0 or real >= 1):  # need ≥1 real
                    c_options.append((c, nw))
        for opt in c_options:
            if r == level_rank:
                c = opt
                nw = 0
            else:
                c, nw = opt
            if last_is_bomb:
                if last_rank == JOKER_BOMB_RANK:
                    continue
                if c > last_count or (c == last_count and r > last_rank):
                    moves.append((BOMB, r, 0, c, nw))
            else:
                moves.append((BOMB, r, 0, c, nw))

    if have_joker_bomb:
        if not (last_is_bomb and last_rank == JOKER_BOMB_RANK):
            moves.append((BOMB, JOKER_BOMB_RANK, 0, 4, 0))


def _add_sequences(hand, last, moves):
    """SEQ5 / PSEQ3 / TSEQ2 — no wildcards allowed (simplification)."""
    free = last is None or last[0] == PASS
    last_rank = -1 if free else last[1]
    last_combo = -1 if free else last[0]

    if free or last_combo == SEQ5:
        for s in range(0, SEQ_HIGH_RANK - SEQ5_LEN + 2):
            if not (free or s > last_rank):
                continue
            if all(hand[s + k] >= 1 for k in range(SEQ5_LEN)):
                moves.append((SEQ5, s, 0, SEQ5_LEN, 0))

    if free or last_combo == PSEQ3:
        for s in range(0, SEQ_HIGH_RANK - PSEQ3_LEN + 2):
            if not (free or s > last_rank):
                continue
            if all(hand[s + k] >= 2 for k in range(PSEQ3_LEN)):
                moves.append((PSEQ3, s, 0, PSEQ3_LEN * 2, 0))

    if free or last_combo == TSEQ2:
        for s in range(0, SEQ_HIGH_RANK - TSEQ2_LEN + 2):
            if not (free or s > last_rank):
                continue
            if all(hand[s + k] >= 3 for k in range(TSEQ2_LEN)):
                moves.append((TSEQ2, s, 0, TSEQ2_LEN * 3, 0))


def legal_moves(hand: np.ndarray, wildcards: int, level_rank: int,
                last: Optional[tuple]) -> list:
    """All legal moves given current hand + wildcards + level + last play."""
    moves: list = []
    free = (last is None) or (last[0] == PASS)
    if not free:
        moves.append(PASS_MOVE)

    _add_singles(hand, wildcards, level_rank, last, moves)
    _add_pairs(hand, wildcards, level_rank, last, moves)
    _add_triples(hand, wildcards, level_rank, last, moves)
    _add_fullhouse(hand, wildcards, level_rank, last, moves)
    _add_sequences(hand, last, moves)
    _add_bombs(hand, wildcards, level_rank, last, moves)
    return moves


def _apply_move_to_hand(hand: np.ndarray, wildcards: int, level_rank: int,
                       move: tuple):
    """Compute hand/wildcards after applying move. Returns (new_hand, new_wildcards).

    Raises ValueError if the move can't be applied.
    """
    combo, rank, pair_rank, count, n_wild = move
    new_hand = hand.astype(np.int32).copy()
    new_wild = int(wildcards)

    def _consume(r, c, nw):
        """Consume c cards of rank r, using nw wildcards (which physically come
        from hand[level_rank])."""
        nonlocal new_wild
        if r == level_rank:
            # nw is forced 0 in this branch; level-rank combo uses regular first
            assert nw == 0
            new_hand[r] -= c
            # If we used more cards than regular available, the excess came from wildcards.
            regular = (hand[r] - wildcards)
            if c > regular:
                new_wild -= (c - regular)
        else:
            real = c - nw
            new_hand[r] -= real
            # nw wildcards come from hand[level_rank]
            new_hand[level_rank] -= nw
            new_wild -= nw

    if combo == PASS:
        return new_hand.astype(np.int8), new_wild

    if combo == SINGLE:
        _consume(rank, 1, n_wild)
    elif combo == PAIR:
        _consume(rank, 2, n_wild)
    elif combo == TRIPLE:
        _consume(rank, 3, n_wild)
    elif combo == FULLHOUSE:
        split = _fh_split(hand, wildcards, level_rank, rank, pair_rank, n_wild)
        if split is None:
            raise ValueError(f"no feasible fullhouse split for {move}")
        nw_t, nw_p = split
        _consume(rank, 3, nw_t)
        _consume(pair_rank, 2, nw_p)
    elif combo == BOMB:
        if rank == JOKER_BOMB_RANK:
            new_hand[SJ] -= 2
            new_hand[BJ] -= 2
        else:
            _consume(rank, count, n_wild)
    elif combo == SEQ5:
        for k in range(SEQ5_LEN):
            _consume(rank + k, 1, 0)
    elif combo == PSEQ3:
        for k in range(PSEQ3_LEN):
            _consume(rank + k, 2, 0)
    elif combo == TSEQ2:
        for k in range(TSEQ2_LEN):
            _consume(rank + k, 3, 0)
    else:
        raise ValueError(f"unknown combo {combo}")

    if (new_hand < 0).any() or new_wild < 0:
        raise ValueError(
            f"illegal move {move_str(move, level_rank)} for hand "
            f"{hand_str(hand, wildcards, level_rank)}: post-state {new_hand}, "
            f"wild={new_wild}"
        )
    return new_hand.astype(np.int8), new_wild


class GuandanEnv:
    """One round of Guandan with level/wildcard rules.

    Default behavior (full_round=False): episode ends as soon as ONE player empties
    their hand. Winner = that player's team. Matches DMC-training style.

    With full_round=True: episode continues until 3 of 4 players have emptied
    (the 4th is implicit). `finish_order` contains all finishers in order. Used
    by the Match wrapper to determine 双下 / 单下 / 平 for level-up.
    """

    def __init__(self, seed: Optional[int] = None, level_rank: Optional[int] = None,
                 full_round: bool = False):
        self.rng = random.Random(seed)
        self._init_level_rank = level_rank
        self.full_round = full_round
        self.reset()

    def reset(self, seed: Optional[int] = None, level_rank: Optional[int] = None,
              hands: Optional[list] = None, wildcards: Optional[list] = None,
              starter: Optional[int] = None):
        """Reset the env. Optional `hands` + `wildcards` override the dealt cards
        (used by Match to inject post-tribute starting hands). `starter` overrides
        who plays first."""
        if seed is not None:
            self.rng = random.Random(seed)
        if level_rank is None:
            level_rank = self._init_level_rank
        if level_rank is None:
            level_rank = self.rng.randrange(13)
        self.level_rank = int(level_rank)
        if hands is not None and wildcards is not None:
            self.hands = [h.astype(np.int8).copy() for h in hands]
            self.wildcards = list(wildcards)
        else:
            self.hands, self.wildcards = deal(self.rng, self.level_rank)
        self.played = [np.zeros(NUM_TYPES, dtype=np.int32) for _ in range(4)]
        self.played_wild = [0, 0, 0, 0]
        self.cur = self.rng.randrange(4) if starter is None else int(starter)
        self.last_play: Optional[tuple] = None
        self.last_player: Optional[int] = None
        self.passes_in_a_row = 0
        self.done = False
        self.winner_team: Optional[int] = None
        self.finish_order: list = []
        self.steps = 0
        return self.obs()

    def obs(self):
        return {
            'cur': self.cur,
            'hand': self.hands[self.cur].copy(),
            'wildcards': int(self.wildcards[self.cur]),
            'level_rank': self.level_rank,
            'last': self.last_play,
            'last_player': self.last_player,
            'hand_sizes': np.array([h.sum() for h in self.hands], dtype=np.int32),
            'wildcards_left': MAX_WILDCARDS - sum(self.played_wild),
            'played': [p.copy() for p in self.played],
            'played_wild': list(self.played_wild),
        }

    def legal(self):
        return legal_moves(self.hands[self.cur], int(self.wildcards[self.cur]),
                           self.level_rank, self.last_play)

    def _next_active(self, start: int) -> int:
        """In full_round mode, advance past finished (empty-hand) players."""
        if not self.full_round:
            return start
        c = start
        for _ in range(4):
            if self.hands[c].sum() > 0:
                return c
            c = (c + 1) % 4
        return start  # all out (shouldn't reach here while running)

    def step(self, move):
        assert not self.done, "step on done env"
        self.steps += 1
        combo = move[0]

        if combo == PASS:
            self.passes_in_a_row += 1
            # Triple pass → trick reset → control returns to last_player (or next)
            if self.passes_in_a_row >= 3:
                self.last_play = None
                self.passes_in_a_row = 0
                if self.last_player is not None and self.hands[self.last_player].sum() > 0:
                    self.cur = self.last_player
                else:
                    self.cur = self._next_active((self.cur + 1) % 4)
                return self.obs(), 0.0, False, {}
            self.cur = self._next_active((self.cur + 1) % 4)
            return self.obs(), 0.0, False, {}

        old_hand = self.hands[self.cur]
        old_wild = self.wildcards[self.cur]
        new_hand, new_wild = _apply_move_to_hand(old_hand, old_wild,
                                                  self.level_rank, move)
        diff = (old_hand.astype(np.int32) - new_hand.astype(np.int32))
        self.played[self.cur] += diff
        self.played_wild[self.cur] += (old_wild - new_wild)
        self.hands[self.cur] = new_hand
        self.wildcards[self.cur] = new_wild
        self.last_play = move
        self.last_player = self.cur
        self.passes_in_a_row = 0

        if self.hands[self.cur].sum() == 0:
            self.finish_order.append(self.cur)
            if not self.full_round:
                # Single-round: stop on first finisher.
                self.done = True
                self.winner_team = self.cur % 2
                self.cur = (self.cur + 1) % 4
                return self.obs(), 0.0, True, {'winner_team': self.winner_team,
                                                'finish_order': list(self.finish_order)}
            # Full round: keep playing. End when 3 of 4 are out (4th is implicit).
            if len(self.finish_order) >= 3:
                # The 4th-place player is the one not yet in finish_order
                remaining = [p for p in range(4) if p not in self.finish_order]
                if remaining:
                    self.finish_order.append(remaining[0])
                self.done = True
                self.winner_team = self.finish_order[0] % 2
                self.cur = self._next_active((self.cur + 1) % 4)
                return self.obs(), 0.0, True, {'winner_team': self.winner_team,
                                                'finish_order': list(self.finish_order)}

        self.cur = self._next_active((self.cur + 1) % 4)
        return self.obs(), 0.0, False, {}
