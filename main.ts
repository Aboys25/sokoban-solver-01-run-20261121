// Push-optimal Sokoban solver, room by room.
//
// Approach: the generator lays a hallway across the board with one to four
// rooms off it, each room holding 3-4 boxes and as many goals, and no box
// ever leaves its room (a box pushed through a door into the hallway can
// never be pushed back). So a level's push-optimal count is the sum of its
// rooms' optima, and each room is solved on its own:
//   1. Cut the dead (non-live) floor squares of the hallway row out of the
//      floor; each remaining part holding a box is a room.
//   2. Solve each room with push-optimal A* over pushes (a state is the
//      boxes plus the player's reachable region, named by its lowest
//      square), with an admissible consistent heuristic (sum of each box's
//      shortest-path distance to its nearest goal), live-square pruning and
//      frozen 2x2 pruning. The goal state must also let the player reach
//      the hallway, so the next room can be entered.
//   3. Stitch the rooms' move strings together with walks between them.
//
// Pure and deterministic: no clock reads, no randomness, budgets are
// expansion counts, all tie-breaks are fixed. Two runs write the same bytes.
// Node built-in modules only; no network, no WASM, no native code.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// A level in XSB text (# wall, space floor, . goal, $ box, * box on a goal,
// @ player, + player on a goal): width, height, walls (1 a wall), goals,
// boxes and player, as square indexes (y * w + x). Short rows and the
// outermost frame count as walls.
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which a box alone can be pushed onto some goal (reverse BFS
// from the goals: y is live when a box at y pushed toward a live square x
// has the player able to stand behind it).
function liveSquares(level) {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y < 0 || y >= wall.length) continue;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// The squares the player reaches from `from` with `occupied` boxes in the
// way: the lowest reachable square (the region's name) and the seen mask.
function regionOf(w, wall, occupied, from) {
  const seen = new Uint8Array(wall.length);
  const stack = [from];
  seen[from] = 1;
  let low = from;
  for (let i = 0; i < stack.length; i++) {
    const p = stack[i];
    if (p < low) low = p;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length) continue;
      if (!wall[q] && !occupied[q] && !seen[q]) {
        seen[q] = 1;
        stack.push(q);
      }
    }
  }
  return { low, seen };
}

// The level's rooms: cut the dead (non-live) floor squares of the player's
// row (the hallway) out of the floor; each part left holding a box is a
// room, its squares, boxes and goals. Null when the level is no set of
// rooms (a box starts in the hallway, a goal falls outside every part, or
// some part holds more or fewer boxes than goals).
function findRooms(level) {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i];
      const px = p % w;
      for (const [dx, dy] of DIRS) {
        if (dx === -1 && px === 0) continue;
        if (dx === 1 && px === w - 1) continue;
        const q = p + dy * w + dx;
        if (q < 0 || q >= wall.length) continue;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Whether the box just pushed to `t` sits in a frozen 2x2 block of walls
// and boxes with a box off its goal.
function frozenBlock(w, wall, occupied, goal, t) {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) ||
        (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

// The shortest walk from `from` to `to` around walls and boxes, in lower
// LURD letters (the first direction first on ties, DIRS order). '' when
// from === to, null when `to` is out of reach.
function walkTo(w, wall, occupied, from, to) {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const px = p % w;
      if (dx === -1 && px === 0) continue;
      if (dx === 1 && px === w - 1) continue;
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

// The square the player stands on after `moves` (LURD, case aside) from `from`.
function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

// Binary heap ordered by (f asc, g desc, counter asc): deterministic.
function heapPush(heap, e, less) {
  let i = heap.length;
  heap.push(e);
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (!less(heap[i], heap[p])) break;
    const t = heap[i]; heap[i] = heap[p]; heap[p] = t;
    i = p;
  }
}
function heapPop(heap, less) {
  const top = heap[0];
  const last = heap.pop();
  if (heap.length > 0) {
    heap[0] = last;
    for (let i = 0; ; ) {
      let c = 2 * i + 1;
      if (c >= heap.length) break;
      if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
      if (!less(heap[c], heap[i])) break;
      const t = heap[i]; heap[i] = heap[c]; heap[c] = t;
      i = c;
    }
  }
  return top;
}

// Push-optimal A* over one room (a sub-level). The heuristic is the sum
// over boxes of the shortest-path distance to the nearest goal: admissible
// and consistent (one push moves one box one square, so the estimate drops
// by at most 1 per push), so the first goal state popped is push-optimal.
// requireExit: the goal state must also have the player able to reach a
// hallway cell, so the next room can be entered. Returns { pushes, moves,
// expansions } or null when the search gives up (budget spent in expansions,
// never the clock) or the room has no solution.
function solveRoom(sub, opts = {}) {
  const { w, wall, goals, boxes, player } = sub;
  const maxExpansions = opts.maxExpansions ?? 250000;
  const requireExit = !!opts.requireExit;
  const hallwayRow = opts.hallwayRow ?? -1;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(sub);
  if (boxes.some((b) => !live[b])) return null;

  // BFS distances from each goal over floor (walls only).
  const distToGoal = goals.map((g) => {
    const d = new Int32Array(size).fill(-1);
    const q = [g];
    d[g] = 0;
    for (let i = 0; i < q.length; i++) {
      const p = q[i];
      const px = p % w;
      for (const [dx, dy] of DIRS) {
        if (dx === -1 && px === 0) continue;
        if (dx === 1 && px === w - 1) continue;
        const n = p + dy * w + dx;
        if (n < 0 || n >= size) continue;
        if (!wall[n] && d[n] < 0) {
          d[n] = d[p] + 1;
          q.push(n);
        }
      }
    }
    return d;
  });
  const hOf = (bs) => {
    let s = 0;
    for (const b of bs) {
      let m = Infinity;
      for (const d of distToGoal) if (d[b] >= 0 && d[b] < m) m = d[b];
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  };

  const solved = (bs) => bs.every((b) => goal[b]);
  const keyOf = (bs, low) => bs.join(',') + '|' + low;

  const statesBoxes = [boxes.slice().sort((a, b) => a - b)];
  const statesPlayer = [player];
  const parent = [-1];
  const pushDir = [-1];
  const gScore = [0];
  const best = new Map();
  const occupied = new Uint8Array(size);
  const first = regionOf(w, wall, occupied, player);
  best.set(keyOf(statesBoxes[0], first.low), 0);
  const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heap = [];
  const h0 = hOf(statesBoxes[0]);
  if (h0 === Infinity) return null;
  let counter = 0;
  heapPush(heap, [h0, 0, counter++, 0], less);

  const reachesHallway = (seen) => {
    if (hallwayRow < 0) return true;
    for (let i = hallwayRow * w; i < (hallwayRow + 1) * w; i++) {
      if (!wall[i] && seen[i]) return true;
    }
    return false;
  };

  let found = -1;
  let expansions = 0;
  while (heap.length > 0) {
    const [, g, , s] = heapPop(heap, less);
    if (g !== gScore[s]) continue;
    if (expansions >= maxExpansions) return null;
    expansions++;
    const bs = statesBoxes[s];
    for (const b of bs) occupied[b] = 1;
    const { seen } = regionOf(w, wall, occupied, statesPlayer[s]);
    if (solved(bs) && (!requireExit || reachesHallway(seen))) {
      found = s;
      for (const x of bs) occupied[x] = 0;
      break;
    }
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi];
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di];
        const d = dy * w + dx;
        const t = b + d;
        const from = b - d;
        if (t < 0 || t >= size || from < 0 || from >= size) continue;
        if (!seen[from] || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, t);
        let key = null;
        if (!frozen) {
          const next = bs.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          key = keyOf(next, regionOf(w, wall, occupied, b).low);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || key === null) continue;
        const ng = g + 1;
        const prevG = best.get(key);
        if (prevG !== undefined && prevG <= ng) continue;
        best.set(key, ng);
        const ns = statesBoxes.length;
        const next = bs.slice();
        next[bi] = t;
        next.sort((a, b2) => a - b2);
        statesBoxes.push(next);
        statesPlayer.push(b);
        parent.push(s);
        pushDir.push(di);
        gScore.push(ng);
        const h = hOf(next);
        if (h === Infinity) continue;
        heapPush(heap, [ng + h, ng, counter++, ns], less);
      }
    }
    for (const x of bs) occupied[x] = 0;
  }
  if (found < 0) return null;
  // Reconstruct the push path, then the moves: walk to each push, push.
  const path = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let curPlayer = player;
  let curBoxes = statesBoxes[0].slice();
  for (const s of path) {
    const nextBoxes = statesBoxes[s];
    const from = curBoxes.find((b) => !nextBoxes.includes(b));
    const di = pushDir[s];
    const [dx, dy, , pushLetter] = DIRS[di];
    const d = dy * w + dx;
    for (const b of curBoxes) occupied[b] = 1;
    const walk = walkTo(w, wall, occupied, curPlayer, from - d);
    for (const b of curBoxes) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk + pushLetter;
    curPlayer = from;
    curBoxes = nextBoxes;
  }
  return { pushes: path.length, moves, expansions };
}

// Solve one whole level (XSB text) room by room: each room's push-optimal
// solution in turn, the other rooms' boxes standing as walls (on their
// goals once their room is done), the player walking on from where the
// last room left it. A box never leaves its room, so the rooms' optima add
// up to the level's. Returns { moves, pushes, ok }: ok is false when some
// room's search gave up (then moves holds the solved rooms' prefix).
function solveLevel(text) {
  const level = parseLevel(text);
  const rooms = findRooms(level);
  const hallwayRow = Math.floor(level.player / level.w);
  let player = level.player;
  let moves = '';
  let totalPushes = 0;
  if (rooms === null) {
    const sub = { w: level.w, h: level.h, wall: level.wall, goals: level.goals, boxes: level.boxes, player };
    const r = solveRoom(sub, { hallwayRow: -1 });
    if (r === null) return { moves: '', pushes: 0, ok: false };
    return { moves: r.moves, pushes: r.pushes, ok: true };
  }
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri];
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      const src = other.done ? other.goals : other.boxes;
      for (const b of src) wall[b] = 1;
    }
    const sub = {
      w: level.w, h: level.h, wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    // First try with the exit condition (the goal state must let the
    // player reach the hallway); fall back to the plain optimum when that
    // finds nothing, so a room is never lost to the exit condition alone.
    let r = solveRoom(sub, { hallwayRow, requireExit: ri < rooms.length - 1 });
    if (r === null) r = solveRoom(sub, { hallwayRow, requireExit: false });
    if (r === null) return { moves, pushes: totalPushes, ok: false };
    moves += r.moves;
    totalPushes += r.pushes;
    player = endOf(level.w, player, r.moves);
    room.done = true;
    // Walk out to the hallway before the next room, when a walk exists.
    if (ri < rooms.length - 1) {
      const occ = new Uint8Array(level.wall.length);
      for (const other of rooms) {
        const src = other.done ? other.goals : other.boxes;
        for (const b of src) occ[b] = 1;
      }
      const prev = new Int32Array(level.wall.length).fill(-1);
      const how = new Array(level.wall.length);
      const queue = [player];
      prev[player] = player;
      let target = -1;
      for (let qi = 0; qi < queue.length; qi++) {
        const p = queue[qi];
        if (Math.floor(p / level.w) === hallwayRow && !level.wall[p]) { target = p; break; }
        const px = p % level.w;
        for (const [dx, dy, letter] of DIRS) {
          if (dx === -1 && px === 0) continue;
          if (dx === 1 && px === level.w - 1) continue;
          const q = p + dy * level.w + dx;
          if (q < 0 || q >= level.wall.length) continue;
          if (level.wall[q] || occ[q] || prev[q] !== -1) continue;
          prev[q] = p;
          how[q] = letter;
          queue.push(q);
        }
      }
      if (target >= 0 && target !== player) {
        let walk = '';
        for (let p = target; p !== player; p = prev[p]) walk = how[p] + walk;
        moves += walk;
        player = target;
      }
    }
  }
  return { moves, pushes: totalPushes, ok: true };
}

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')).moves;
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
