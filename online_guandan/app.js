(function () {
  const $ = (s) => document.querySelector(s);

  let state = null;
  // Card instances in hand (each is a separate clickable card)
  // each item: { id, rank, isWild, selected }
  let handCards = [];

  async function jsonPost(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    return r.json();
  }
  async function jsonGet(url) {
    return (await fetch(url)).json();
  }

  function rankLabel(rankIdx, rankNames) {
    const name = rankNames[rankIdx];
    if (name === 'sj') return 'sJ';
    if (name === 'bj') return 'BJ';
    return name;
  }

  function isRedRank(rankIdx, rankNames) {
    // Decorative only — we don't really have suits in v6
    return rankNames[rankIdx] === 'bj';
  }

  function buildHandCards(s) {
    // Build a list of individual card instances from the rank counts.
    // Wildcards are level-rank cards but rendered as a distinct visual.
    const cards = [];
    const lr = s.level_rank;
    let idCounter = 0;
    for (let r = 0; r < 15; r++) {
      const totalAtRank = s.hand[r];
      if (totalAtRank === 0) continue;
      let regular = totalAtRank;
      let wild = 0;
      if (r === lr) {
        wild = s.wildcards;
        regular = totalAtRank - wild;
      }
      for (let i = 0; i < regular; i++) {
        cards.push({ id: `r${r}-i${idCounter++}`, rank: r, isWild: false, selected: false });
      }
      for (let i = 0; i < wild; i++) {
        cards.push({ id: `w${r}-i${idCounter++}`, rank: r, isWild: true, selected: false });
      }
    }
    return cards;
  }

  function makeCardDOM(card, rankNames) {
    const div = document.createElement('div');
    div.className = 'card';
    if (card.isWild) div.classList.add('wildcard');
    if (rankNames[card.rank] === 'sj') div.classList.add('joker-small');
    if (rankNames[card.rank] === 'bj') div.classList.add('joker-big');
    if (isRedRank(card.rank, rankNames)) div.classList.add('red');
    if (card.selected) div.classList.add('selected');
    div.dataset.id = card.id;

    const top = document.createElement('div');
    top.className = 'suit';
    top.textContent = card.isWild ? '万' : '';
    const center = document.createElement('div');
    center.className = 'rank';
    center.textContent = rankLabel(card.rank, rankNames);
    div.appendChild(top);
    div.appendChild(center);

    div.addEventListener('click', () => {
      card.selected = !card.selected;
      renderHand();
      checkMatch();
    });
    return div;
  }

  function makeStaticCardDOM(rank, count, rankNames, level_rank, wildcards_used, cls = 'played') {
    // For displaying played cards (last play) — render `count` cards of the rank.
    // For wildcards, render first `wildcards_used` as wild.
    const frag = document.createDocumentFragment();
    for (let i = 0; i < count; i++) {
      const div = document.createElement('div');
      div.className = `card ${cls}`;
      const isWild = (rank === level_rank && i < wildcards_used);
      if (isWild) div.classList.add('wildcard');
      if (rankNames[rank] === 'sj') div.classList.add('joker-small');
      if (rankNames[rank] === 'bj') div.classList.add('joker-big');
      const center = document.createElement('div');
      center.className = 'rank';
      center.textContent = rankLabel(rank, rankNames);
      div.appendChild(center);
      frag.appendChild(div);
    }
    return frag;
  }

  function renderHand() {
    const handDiv = $('#hand');
    handDiv.innerHTML = '';
    handCards.forEach(c => handDiv.appendChild(makeCardDOM(c, state.rank_names)));
  }

  function selectionSignature() {
    // Returns { regular: [15 ints], wild: int }
    const regular = new Array(15).fill(0);
    let wild = 0;
    handCards.forEach(c => {
      if (!c.selected) return;
      if (c.isWild) wild += 1;
      else regular[c.rank] += 1;
    });
    return { regular, wild };
  }

  function moveExpectedSignature(move, level_rank) {
    // The backend gave us `consumed` (total per rank) and `consumed_wild`.
    // The total consumed at level_rank = regular_level + wildcards.
    const regular = move.consumed.slice();
    regular[level_rank] = (move.consumed[level_rank] || 0) - move.consumed_wild;
    return { regular, wild: move.consumed_wild };
  }

  function sigEq(a, b) {
    if (a.wild !== b.wild) return false;
    for (let i = 0; i < 15; i++) if (a.regular[i] !== b.regular[i]) return false;
    return true;
  }

  function findMatchingMove() {
    if (!state || !state.is_human_turn) return null;
    const sel = selectionSignature();
    const total = sel.regular.reduce((a, b) => a + b, 0) + sel.wild;
    if (total === 0) return null;
    const lr = state.level_rank;
    const matches = state.legal_moves.filter(m => {
      if (m.combo === 0) return false; // skip PASS
      if (!m.consumed) return false;
      return sigEq(sel, moveExpectedSignature(m, lr));
    });
    return matches;
  }

  function checkMatch() {
    const msg = $('#match-msg');
    const playBtn = $('#btn-play');
    if (!state || !state.is_human_turn) {
      msg.textContent = '';
      playBtn.disabled = true;
      return;
    }
    const matches = findMatchingMove();
    if (!matches || matches.length === 0) {
      const sel = selectionSignature();
      const total = sel.regular.reduce((a, b) => a + b, 0) + sel.wild;
      msg.textContent = total === 0 ? '请选择要打的牌' : '当前选择不是合法组合';
      msg.className = total === 0 ? 'match-msg' : 'match-msg';
      playBtn.disabled = true;
      playBtn.dataset.move = '';
    } else if (matches.length === 1) {
      msg.textContent = `→ ${matches[0].human}`;
      msg.className = 'match-msg ok';
      playBtn.disabled = false;
      playBtn.dataset.move = JSON.stringify(matches[0]);
    } else {
      // Show first match; let user click play to choose, or show pick UI
      msg.textContent = `多种解释:${matches.map(m => m.human).join(' / ')} — 按"出牌"用第一种`;
      msg.className = 'match-msg ok';
      playBtn.disabled = false;
      playBtn.dataset.move = JSON.stringify(matches[0]);
    }
  }

  function renderLastPlay(s) {
    // Each player's last-play-box. Clear all first.
    document.querySelectorAll('.last-play-box').forEach(el => el.innerHTML = '');
    if (!s.last || s.last_player == null) return;
    const seatToBox = {
      [(s.human_seat + 2) % 4]: '#player-top .last-play-box',
      [(s.human_seat + 1) % 4]: '#player-left .last-play-box',
      [(s.human_seat + 3) % 4]: '#player-right .last-play-box',
    };
    const sel = seatToBox[s.last_player];
    if (!sel) return;
    const box = document.querySelector(sel);
    const m = s.last;
    if (m.combo === 0) {
      const tag = document.createElement('div');
      tag.style.color = '#aaa';
      tag.textContent = '不要';
      box.appendChild(tag);
      return;
    }
    // Render consumed cards as visuals
    if (!m.consumed) {
      // best-effort fallback
      const tag = document.createElement('div');
      tag.textContent = m.human;
      box.appendChild(tag);
      return;
    }
    const lr = s.level_rank;
    const totalConsumed = m.consumed;
    const wildConsumed = m.consumed_wild || 0;
    for (let r = 0; r < 15; r++) {
      let c = totalConsumed[r];
      if (c === 0) continue;
      let wildAtR = (r === lr) ? wildConsumed : 0;
      box.appendChild(makeStaticCardDOM(r, c, s.rank_names, lr, wildAtR, 'played'));
    }
  }

  function render(s) {
    state = s;
    // Status / level
    $('#level-banner').textContent = s.level_rank_name;
    const statusEl = $('#status-banner');
    if (s.done) {
      const won = s.winner_team === s.human_team;
      statusEl.textContent = won ? '🎉 你方获胜!' : '😢 你方失败';
      statusEl.style.color = won ? '#4ade80' : '#f87171';
    } else {
      const labels = s.seat_labels;
      statusEl.textContent = `轮到:${labels[String(s.cur)] || ('seat ' + s.cur)}`;
      statusEl.style.color = '#f0f0e0';
    }

    // Player meta + active highlight
    const TM = (s.human_seat + 2) % 4;
    const OPL = (s.human_seat + 1) % 4;
    const OPR = (s.human_seat + 3) % 4;
    document.querySelectorAll('.player').forEach(el => el.classList.remove('active'));
    if (!s.done) {
      const seatMap = { [TM]: '#player-top', [OPL]: '#player-left',
                         [OPR]: '#player-right', [s.human_seat]: '#player-bottom' };
      const cls = seatMap[s.cur];
      if (cls) document.querySelector(cls).classList.add('active');
    }
    $('#player-top .num').textContent = s.hand_sizes[TM];
    $('#player-left .num').textContent = s.hand_sizes[OPL];
    $('#player-right .num').textContent = s.hand_sizes[OPR];

    // Last play
    renderLastPlay(s);

    // Hand
    const handTotal = s.hand.reduce((a, b) => a + b, 0);
    $('#hand-count').textContent = handTotal;
    $('#hand-wild-info').textContent = s.wildcards > 0
      ? `(其中 ${s.wildcards} 张万能-红心${s.level_rank_name})` : '';

    // Preserve selection where possible
    const oldSel = new Set(handCards.filter(c => c.selected).map(c => c.id));
    handCards = buildHandCards(s);
    handCards.forEach(c => { if (oldSel.has(c.id)) c.selected = true; });
    renderHand();

    // Match-check
    checkMatch();

    // Pass button enable check
    const hasPass = s.is_human_turn && s.legal_moves.some(m => m.combo === 0);
    $('#btn-pass').disabled = !hasPass;

    // Log
    const logList = $('#log-list');
    logList.innerHTML = '';
    (s.log || []).slice(-200).forEach(e => {
      const li = document.createElement('li');
      if (e.type === 'play') {
        const label = s.seat_labels[String(e.player)] || ('seat ' + e.player);
        li.textContent = `${label}: ${e.move.human}`;
        if (e.player === s.human_seat) li.classList.add('me');
      } else if (e.type === 'end') {
        const won = e.winner_team === s.human_team;
        li.textContent = won ? `本局你方获胜 (头名 seat ${e.finish_order[0]})`
                              : `本局对方获胜 (头名 seat ${e.finish_order[0]})`;
        li.classList.add(won ? 'win' : 'lose');
      }
      logList.appendChild(li);
    });
    logList.scrollTop = logList.scrollHeight;
  }

  async function doPlay() {
    const btn = $('#btn-play');
    if (!btn.dataset.move) return;
    const move = JSON.parse(btn.dataset.move);
    const r = await jsonPost('/api/play', { move });
    render(r);
  }

  async function doPass() {
    if (!state || !state.is_human_turn) return;
    const passMove = state.legal_moves.find(m => m.combo === 0);
    if (!passMove) return;
    const r = await jsonPost('/api/play', { move: passMove });
    render(r);
  }

  async function doNew() {
    handCards = [];
    const r = await jsonPost('/api/new', {});
    render(r);
  }

  function doClear() {
    handCards.forEach(c => c.selected = false);
    renderHand();
    checkMatch();
  }

  async function init() {
    $('#btn-play').addEventListener('click', doPlay);
    $('#btn-pass').addEventListener('click', doPass);
    $('#btn-new').addEventListener('click', doNew);
    $('#btn-clear').addEventListener('click', doClear);
    $('#btn-toggle-log').addEventListener('click', () => {
      const panel = document.querySelector('.log-panel');
      panel.classList.toggle('collapsed');
      $('#btn-toggle-log').textContent = panel.classList.contains('collapsed') ? '展开' : '收起';
    });
    const s = await jsonGet('/api/state');
    render(s);
  }

  init();
})();
