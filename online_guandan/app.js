(function () {
  const $ = (s) => document.querySelector(s);
  const SUITS = ['♠', '♥', '♦', '♣'];  // visual only — v6 env has no real suits
  const RED_SUITS = new Set(['♥', '♦']);

  let state = null;
  // Card instances in hand. Each: { id, rank, suit, isWild, selected }
  let handCards = [];

  async function jsonPost(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    return r.json();
  }
  async function jsonGet(url) { return (await fetch(url)).json(); }

  function rankLabel(rankIdx, rankNames) {
    const name = rankNames[rankIdx];
    if (name === 'sj') return 'JOKER';
    if (name === 'bj') return 'JOKER';
    return name;
  }

  // Deterministic suit cycling: card #i of rank r gets suit (i + r) % 4
  function suitForCard(rank, idxInRank) {
    return SUITS[(idxInRank + rank) % 4];
  }

  function buildHandCards(s) {
    const cards = [];
    const lr = s.level_rank;
    let counter = 0;
    for (let r = 0; r < 15; r++) {
      const totalAtRank = s.hand[r];
      if (totalAtRank === 0) continue;
      let regular = totalAtRank;
      let wild = 0;
      if (r === lr) {
        wild = s.wildcards;
        regular = totalAtRank - wild;
      }
      // jokers
      const isJoker = (s.rank_names[r] === 'sj' || s.rank_names[r] === 'bj');
      // regular cards
      for (let i = 0; i < regular; i++) {
        const suit = isJoker ? '' : suitForCard(r, counter);
        cards.push({
          id: `r${r}-i${counter}`, rank: r,
          suit, isWild: false, isJoker, selected: false
        });
        counter++;
      }
      // wildcards (always red hearts)
      for (let i = 0; i < wild; i++) {
        cards.push({
          id: `w${r}-i${counter}`, rank: r,
          suit: '♥', isWild: true, isJoker: false, selected: false
        });
        counter++;
      }
    }
    return cards;
  }

  function classForCard(card, rankNames) {
    const classes = ['card'];
    if (card.isJoker) {
      classes.push('joker');
      if (rankNames[card.rank] === 'sj') classes.push('joker-small');
      else classes.push('joker-big');
    } else if (card.isWild) {
      classes.push('wildcard');
      classes.push('suit-red');
    } else {
      classes.push(RED_SUITS.has(card.suit) ? 'suit-red' : 'suit-black');
    }
    if (card.selected) classes.push('selected');
    return classes.join(' ');
  }

  function makeCardDOM(card, rankNames, opts) {
    opts = opts || {};
    const div = document.createElement('div');
    div.className = classForCard(card, rankNames);
    if (opts.small) div.classList.add('played');
    div.dataset.id = card.id;

    if (card.isJoker) {
      // big "JOKER" text with small/big distinction
      const center = document.createElement('div');
      center.className = 'center';
      center.textContent = rankNames[card.rank] === 'bj' ? 'BIG\nJOKER' : 'SMALL\nJOKER';
      center.style.whiteSpace = 'pre';
      div.appendChild(center);
    } else {
      const label = rankLabel(card.rank, rankNames);
      const tl = document.createElement('div');
      tl.className = 'corner-tl';
      tl.innerHTML = `<span>${label}</span><span>${card.suit}</span>`;
      const br = document.createElement('div');
      br.className = 'corner-br';
      br.innerHTML = `<span>${label}</span><span>${card.suit}</span>`;
      const c = document.createElement('div');
      c.className = 'center';
      c.textContent = card.suit;
      div.appendChild(tl);
      div.appendChild(br);
      div.appendChild(c);
    }

    if (opts.onClick) {
      div.addEventListener('click', opts.onClick);
    }
    return div;
  }

  function renderHand() {
    const handDiv = $('#hand');
    handDiv.innerHTML = '';
    handCards.forEach(c => {
      const dom = makeCardDOM(c, state.rank_names, {
        onClick: () => { c.selected = !c.selected; renderHand(); checkMatch(); }
      });
      handDiv.appendChild(dom);
    });
  }

  function renderPlayedCards(rank, count, wildAtR, rankNames, lr) {
    // Render `count` cards of rank, of which `wildAtR` are wildcards.
    const row = document.createElement('div');
    row.className = 'played-row';
    let counter = 0;
    const isJoker = (rankNames[rank] === 'sj' || rankNames[rank] === 'bj');
    // regular first
    const regular = count - wildAtR;
    for (let i = 0; i < regular; i++) {
      const c = {
        rank, isWild: false, isJoker,
        suit: isJoker ? '' : suitForCard(rank, counter),
        selected: false,
      };
      row.appendChild(makeCardDOM(c, rankNames, { small: true }));
      counter++;
    }
    for (let i = 0; i < wildAtR; i++) {
      const c = { rank, isWild: true, isJoker: false, suit: '♥', selected: false };
      row.appendChild(makeCardDOM(c, rankNames, { small: true }));
      counter++;
    }
    return row;
  }

  function selectionSignature() {
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
    const regular = move.consumed.slice();
    regular[level_rank] = (move.consumed[level_rank] || 0) - move.consumed_wild;
    return { regular, wild: move.consumed_wild };
  }

  function sigEq(a, b) {
    if (a.wild !== b.wild) return false;
    for (let i = 0; i < 15; i++) if (a.regular[i] !== b.regular[i]) return false;
    return true;
  }

  function findMatches() {
    if (!state || !state.is_human_turn) return [];
    const sel = selectionSignature();
    const total = sel.regular.reduce((a, b) => a + b, 0) + sel.wild;
    if (total === 0) return [];
    const lr = state.level_rank;
    return state.legal_moves.filter(m => {
      if (m.combo === 0 || !m.consumed) return false;
      return sigEq(sel, moveExpectedSignature(m, lr));
    });
  }

  function checkMatch() {
    const msg = $('#match-msg');
    const playBtn = $('#btn-play');
    if (!state || !state.is_human_turn) {
      msg.textContent = '';
      playBtn.disabled = true;
      return;
    }
    const matches = findMatches();
    const sel = selectionSignature();
    const total = sel.regular.reduce((a, b) => a + b, 0) + sel.wild;
    if (matches.length === 0) {
      msg.textContent = total === 0 ? '请选择要打的牌' : '当前选择不是合法组合';
      msg.className = 'match-msg';
      playBtn.disabled = true;
      playBtn.dataset.move = '';
    } else if (matches.length === 1) {
      msg.textContent = `→ ${matches[0].human}`;
      msg.className = 'match-msg ok';
      playBtn.disabled = false;
      playBtn.dataset.move = JSON.stringify(matches[0]);
    } else {
      msg.textContent = `多种解释 (用第一种):${matches.map(m => m.human).join(' / ')}`;
      msg.className = 'match-msg ok';
      playBtn.disabled = false;
      playBtn.dataset.move = JSON.stringify(matches[0]);
    }
  }

  function renderLastPlay(s) {
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
      tag.className = 'pass-tag';
      tag.textContent = '不要';
      box.appendChild(tag);
      return;
    }
    if (!m.consumed) {
      box.textContent = m.human;
      return;
    }
    // Render consumed cards as visual cards (sorted by rank ascending)
    const lr = s.level_rank;
    const totalConsumed = m.consumed;
    const wildConsumed = m.consumed_wild || 0;
    // We render one big row for the play
    const row = document.createElement('div');
    row.className = 'played-row';
    for (let r = 0; r < 15; r++) {
      const c = totalConsumed[r];
      if (c === 0) continue;
      const wildAtR = (r === lr) ? wildConsumed : 0;
      const sub = renderPlayedCards(r, c, wildAtR, s.rank_names, lr);
      while (sub.firstChild) row.appendChild(sub.firstChild);
    }
    box.appendChild(row);
  }

  function render(s) {
    state = s;
    $('#level-banner').textContent = s.level_rank_name;
    const statusEl = $('#status-banner');
    if (s.done) {
      const won = s.winner_team === s.human_team;
      statusEl.textContent = won ? '🎉 你方获胜!' : '😢 你方失败';
      statusEl.style.color = won ? '#4ade80' : '#f87171';
    } else {
      statusEl.textContent = `轮到:${s.seat_labels[String(s.cur)] || ('seat ' + s.cur)}`;
      statusEl.style.color = '#f0f0e0';
    }

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

    renderLastPlay(s);

    const handTotal = s.hand.reduce((a, b) => a + b, 0);
    $('#hand-count').textContent = handTotal;
    $('#hand-wild-info').textContent = s.wildcards > 0
      ? `(含 ${s.wildcards} 张红心${s.level_rank_name} 万能牌)` : '';

    const oldSel = new Set(handCards.filter(c => c.selected).map(c => c.id));
    handCards = buildHandCards(s);
    handCards.forEach(c => { if (oldSel.has(c.id)) c.selected = true; });
    renderHand();
    checkMatch();

    const hasPass = s.is_human_turn && s.legal_moves.some(m => m.combo === 0);
    $('#btn-pass').disabled = !hasPass;

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
