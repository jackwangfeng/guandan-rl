(function () {
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  async function jsonPost(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    return r.json();
  }
  async function jsonGet(url) {
    const r = await fetch(url);
    return r.json();
  }

  function render(state) {
    if (state.error) {
      console.warn(state.error);
    }
    const HS = state.human_seat;
    const TM = (HS + 2) % 4;
    const OPL = (HS + 1) % 4;
    const OPR = (HS + 3) % 4;

    // Status header
    $('#level-info').textContent =
      `级牌:${state.level_rank_name}  |  万能牌剩 ${state.wildcards_left ?? '-'}`;
    if (state.done) {
      const won = state.winner_team === state.human_team;
      $('#status').textContent = won ? '🎉 你方获胜' : '😢 你方失败';
      $('#status').style.color = won ? '#4ade80' : '#f87171';
    } else {
      const turnLabel = state.seat_labels[String(state.cur)];
      $('#status').textContent = `轮到:${turnLabel}`;
      $('#status').style.color = '#f0f0e0';
    }

    // Opponents
    $('#seat-teammate .cards-count').textContent = `${state.hand_sizes[TM]} 张`;
    $('#seat-opp-l .cards-count').textContent = `${state.hand_sizes[OPL]} 张`;
    $('#seat-opp-r .cards-count').textContent = `${state.hand_sizes[OPR]} 张`;
    $$('.seat').forEach(el => el.classList.remove('active-turn'));
    if (!state.done) {
      const seatMap = { [TM]: 'seat-teammate', [OPL]: 'seat-opp-l', [OPR]: 'seat-opp-r' };
      const cls = seatMap[state.cur];
      if (cls) $('#' + cls).classList.add('active-turn');
    }

    // Last play
    if (state.last) {
      $('#last-play').textContent = state.last.human;
      const by = state.last_player == null ? '' :
        `由 ${state.seat_labels[String(state.last_player)] || ('seat ' + state.last_player)} 出`;
      $('#last-by').textContent = by;
    } else {
      $('#last-play').textContent = '自由出';
      $('#last-by').textContent = '';
    }

    // Hand
    const totalHand = state.hand.reduce((a, b) => a + b, 0);
    $('#hand-count').textContent = totalHand;
    $('#hand-wild').textContent = state.wildcards;
    const handDiv = $('#hand');
    handDiv.innerHTML = '';
    state.hand.forEach((cnt, rankIdx) => {
      if (cnt === 0) return;
      const isWild = (rankIdx === state.level_rank && state.wildcards > 0);
      const isJoker = (state.rank_names[rankIdx] === 'sj' || state.rank_names[rankIdx] === 'bj');
      const wildCount = isWild ? state.wildcards : 0;
      const regular = cnt - wildCount;
      if (regular > 0) {
        const div = document.createElement('div');
        div.className = 'card' + (isJoker ? ' joker' : '');
        div.innerHTML = state.rank_names[rankIdx] + (regular > 1 ? `<span class="count">${regular}</span>` : '');
        handDiv.appendChild(div);
      }
      if (wildCount > 0) {
        const div = document.createElement('div');
        div.className = 'card wildcard';
        div.innerHTML = state.rank_names[rankIdx] + '<sup style="font-size:10px;">★</sup>' +
                        (wildCount > 1 ? `<span class="count">${wildCount}</span>` : '');
        handDiv.appendChild(div);
      }
    });
    if (totalHand === 0) handDiv.innerHTML = '<span style="color:#888;">(空)</span>';

    // Legal moves
    const movesDiv = $('#moves');
    movesDiv.innerHTML = '';
    if (state.is_human_turn && state.legal_moves.length > 0) {
      // Group/sort: pass first, then by combo, then by rank
      const sorted = state.legal_moves.slice().sort((a, b) => {
        if (a.combo !== b.combo) return a.combo - b.combo;
        if (a.rank !== b.rank) return a.rank - b.rank;
        return a.n_wild - b.n_wild;
      });
      sorted.forEach((m) => {
        const btn = document.createElement('button');
        btn.className = 'move-btn';
        if (m.combo === 0) btn.classList.add('pass');
        if (m.combo === 5) btn.classList.add('bomb');
        btn.textContent = m.human;
        btn.addEventListener('click', () => playMove(m));
        movesDiv.appendChild(btn);
      });
    } else if (state.done) {
      movesDiv.innerHTML = '<span style="color:#888;">游戏结束,点新游戏</span>';
    } else {
      movesDiv.innerHTML = '<span style="color:#888;">等待 AI 出牌中...</span>';
    }

    // Log
    const logList = $('#log-list');
    logList.innerHTML = '';
    (state.log || []).slice(-200).forEach((entry) => {
      const li = document.createElement('li');
      if (entry.type === 'play') {
        const tag = entry.seat_tag;
        const seatLabel = state.seat_labels[String(entry.player)] || ('seat ' + entry.player);
        li.textContent = `${seatLabel}: ${entry.move.human}`;
        if (entry.player === state.human_seat) li.classList.add('me');
      } else if (entry.type === 'end') {
        const won = entry.winner_team === state.human_team;
        li.textContent = won ? `本局你方获胜 (头名 seat ${entry.finish_order[0]})`
                              : `本局对方获胜 (头名 seat ${entry.finish_order[0]})`;
        li.classList.add(won ? 'win' : 'lose');
      } else if (entry.type === 'error') {
        li.textContent = `[err] ${entry.text}`;
        li.style.color = '#f87171';
      }
      logList.appendChild(li);
    });
    logList.scrollTop = logList.scrollHeight;
  }

  async function playMove(move) {
    const state = await jsonPost('/api/play', { move });
    render(state);
  }

  async function newGame() {
    const state = await jsonPost('/api/new', { seed: null });
    render(state);
  }

  async function init() {
    $('#new-game').addEventListener('click', newGame);
    const state = await jsonGet('/api/state');
    render(state);
  }

  init();
})();
