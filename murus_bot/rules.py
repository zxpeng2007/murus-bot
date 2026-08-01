"""The rules of the wall game and the move notation Murus speaks.

Self-contained and dependency-free: this module is the only thing a bot needs
in order to know what its legal moves are and what the board looks like after
each of them. It is written for clarity, because an engine makes one move at a
time and a few hundred microseconds either way are irrelevant next to the
thinking that follows.

The spec is API.md at https://github.com/zxpeng2007/murus. In summary:

* A 9x9 board. Files ``a``-``i`` run left to right, ranks ``1``-``9`` bottom to
  top, from Player 1's point of view.
* Player 1 starts on ``e1`` and wins by reaching rank 9; Player 2 starts on
  ``e9`` and wins by reaching rank 1. There are no draws.
* Each turn: step the pawn one square orthogonally, or place one of your ten
  walls. A wall spans two cells; walls may not overlap or cross, and no wall
  may leave *either* player without a path to their goal.
* A pawn facing the opponent may jump over them. If the square directly behind
  the opponent is off the board or walled off, the jumper may instead step
  diagonally past them, to either side that is not walled off.

Internals worth knowing if you write your own engine:

* Cells are ``row * 9 + col`` with row 0 = rank 1 and col 0 = file a, so
  ``e1`` is cell 4 and ``e9`` is cell 76.
* Wall slots are an 8x8 grid ``(wr, wc)``; the token for a slot is the
  orientation letter plus ``FILES[wc]`` and ``wr + 1``, so anchors run
  ``a1``-``h8``. A horizontal wall at ``(wr, wc)`` lies on the edge between
  rows ``wr`` and ``wr + 1`` and spans columns ``wc`` and ``wc + 1``; a
  vertical wall at ``(wr, wc)`` lies between columns ``wc`` and ``wc + 1`` and
  spans rows ``wr`` and ``wr + 1``.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

__all__ = [
    "N", "NUM_CELLS", "NUM_WALL_SLOTS", "WALLS_PER_PLAYER", "FILES", "INF",
    "HORIZONTAL", "VERTICAL", "START_CELLS", "GOAL_ROWS",
    "IllegalMove", "State",
    "initial_state", "legal_tokens", "apply_token", "winner", "replay",
    "cell_of", "row_col", "square_token", "parse_square",
    "wall_token", "parse_wall", "is_wall_token", "wall_edges",
]

N = 9                                   # board is N x N cells
W = N - 1                               # wall slots are W x W
NUM_CELLS = N * N                       # 81
NUM_WALL_SLOTS = W * W                  # 64
WALLS_PER_PLAYER = 10
FILES = "abcdefghi"
RANKS = "123456789"

HORIZONTAL = 0
VERTICAL = 1

START_CELLS = (4, 76)                   # e1, e9
GOAL_ROWS = (8, 0)                      # rank 9 for Player 1, rank 1 for Player 2

#: Stand-in for "no path", larger than any real distance on this board.
INF = 10 ** 6

# The four steps, in a fixed order used as an index throughout. Names are the
# board as drawn: rank 9 is up, file i is right.
DOWN, UP, RIGHT, LEFT = 0, 1, 2, 3
DELTAS = ((-1, 0), (1, 0), (0, 1), (0, -1))

# The two sideways steps for each direction, used for diagonal jumps.
PERPENDICULAR = ((RIGHT, LEFT), (RIGHT, LEFT), (DOWN, UP), (DOWN, UP))


def cell_of(row: int, col: int) -> int:
    return row * N + col


def row_col(cell: int) -> tuple[int, int]:
    return divmod(cell, N)


# ------------------------------------------------------------------ notation

def square_token(cell: int) -> str:
    """Cell index -> square token, e.g. 4 -> ``'e1'``."""
    row, col = divmod(int(cell), N)
    return f"{FILES[col]}{row + 1}"


def parse_square(token: str) -> int | None:
    """Square token -> cell index, or None if the token is malformed."""
    if len(token) != 2:
        return None
    col = FILES.find(token[0])
    if col < 0 or token[1] not in RANKS:
        return None
    return cell_of(int(token[1]) - 1, col)


def wall_token(orientation: int, wr: int, wc: int) -> str:
    """Wall slot -> token, e.g. ``(HORIZONTAL, 2, 3)`` -> ``'hd3'``."""
    return ("h" if orientation == HORIZONTAL else "v") + FILES[wc] + str(wr + 1)


def parse_wall(token: str) -> tuple[int, int, int] | None:
    """Wall token -> ``(orientation, wr, wc)``, or None if malformed.

    Says nothing about legality: ``ha1`` parses on an empty board and on a
    board that already has a wall there.
    """
    if len(token) != 3 or token[0] not in "hv":
        return None
    wc = FILES.find(token[1])
    if not 0 <= wc < W or token[2] not in RANKS[:W]:
        return None
    return (HORIZONTAL if token[0] == "h" else VERTICAL, int(token[2]) - 1, wc)


def is_wall_token(token: str) -> bool:
    """True for wall tokens, False for pawn moves. Does not validate further.

    Length is what separates them, not the leading letter: file h collides with
    the horizontal-wall prefix, so ``h2`` is a pawn move to h2 while ``hh2`` is
    a wall.
    """
    return len(token) == 3 and token[0] in "hv"


def wall_edges(orientation: int, wr: int, wc: int) -> tuple[tuple[int, int],
                                                           tuple[int, int]]:
    """The two board edges a wall at this slot blocks, as sorted cell pairs."""
    if orientation == HORIZONTAL:
        pairs = ((cell_of(wr, wc), cell_of(wr + 1, wc)),
                 (cell_of(wr, wc + 1), cell_of(wr + 1, wc + 1)))
    else:
        pairs = ((cell_of(wr, wc), cell_of(wr, wc + 1)),
                 (cell_of(wr + 1, wc), cell_of(wr + 1, wc + 1)))
    return tuple(tuple(sorted(pair)) for pair in pairs)  # type: ignore[return-value]


# ------------------------------------------------------------- board tables

def _build_neighbours() -> list[int]:
    """``_NEIGHBOUR[cell * 4 + d]`` is the cell one step in direction d, or -1."""
    out = [-1] * (NUM_CELLS * 4)
    for cell in range(NUM_CELLS):
        row, col = divmod(cell, N)
        for d, (dr, dc) in enumerate(DELTAS):
            nr, nc = row + dr, col + dc
            if 0 <= nr < N and 0 <= nc < N:
                out[cell * 4 + d] = cell_of(nr, nc)
    return out


def _build_base_passable() -> list[bool]:
    """Which steps exist on an empty board; walls only ever clear entries."""
    return [nb >= 0 for nb in _NEIGHBOUR]


_NEIGHBOUR = _build_neighbours()
_BASE_PASSABLE = _build_base_passable()


def _slot(wr: int, wc: int) -> int:
    return wr * W + wc


def _passable_for(walls_h: frozenset[int], walls_v: frozenset[int]) -> list[bool]:
    """Step table for a wall configuration: ``[cell * 4 + direction] -> bool``.

    Computed once per position and reused by every path query, which is what
    keeps move generation cheap despite the connectivity rule.
    """
    passable = list(_BASE_PASSABLE)
    for slot in walls_h:
        wr, wc = divmod(slot, W)
        top, bottom = wr * N + wc, (wr + 1) * N + wc
        passable[top * 4 + UP] = False
        passable[(top + 1) * 4 + UP] = False
        passable[bottom * 4 + DOWN] = False
        passable[(bottom + 1) * 4 + DOWN] = False
    for slot in walls_v:
        wr, wc = divmod(slot, W)
        left, right = wr * N + wc, wr * N + wc + 1
        passable[left * 4 + RIGHT] = False
        passable[(left + N) * 4 + RIGHT] = False
        passable[right * 4 + LEFT] = False
        passable[(right + N) * 4 + LEFT] = False
    return passable


def _wall_step_indices(orientation: int, wr: int, wc: int) -> tuple[int, ...]:
    """The four ``passable`` entries a wall at this slot clears."""
    if orientation == HORIZONTAL:
        top, bottom = wr * N + wc, (wr + 1) * N + wc
        return (top * 4 + UP, (top + 1) * 4 + UP,
                bottom * 4 + DOWN, (bottom + 1) * 4 + DOWN)
    left, right = wr * N + wc, wr * N + wc + 1
    return (left * 4 + RIGHT, (left + N) * 4 + RIGHT,
            right * 4 + LEFT, (right + N) * 4 + LEFT)


# -------------------------------------------------------------- path search

def _distances(passable: list[bool], sources: Iterable[int]) -> list[int]:
    """BFS from one or many cells at once: steps from every cell to the nearest.

    Multi-source is what makes "how far to the goal" one search rather than
    nine: seed the whole goal rank and every cell learns its distance to the
    closest square of it.
    """
    dist = [INF] * NUM_CELLS
    queue = deque(sources)
    for cell in queue:
        dist[cell] = 0
    while queue:
        cell = queue.popleft()
        nd = dist[cell] + 1
        base = cell * 4
        for d in range(4):
            if not passable[base + d]:
                continue
            nxt = _NEIGHBOUR[base + d]
            if dist[nxt] > nd:
                dist[nxt] = nd
                queue.append(nxt)
    return dist


def _reaches_row(passable: list[bool], start: int, goal_row: int) -> bool:
    """True if a pawn on ``start`` can still get to ``goal_row``.

    Floods from the pawn and stops the moment the goal row is touched, which on
    a live board is usually after a handful of cells.
    """
    if start // N == goal_row:
        return True
    seen = bytearray(NUM_CELLS)
    seen[start] = 1
    queue = deque((start,))
    while queue:
        cell = queue.popleft()
        base = cell * 4
        for d in range(4):
            if not passable[base + d]:
                continue
            nxt = _NEIGHBOUR[base + d]
            if seen[nxt]:
                continue
            if nxt // N == goal_row:
                return True
            seen[nxt] = 1
            queue.append(nxt)
    return False


# ------------------------------------------------------------------- errors

class IllegalMove(ValueError):
    """Raised by :func:`apply_token` for anything the rules do not allow."""


# -------------------------------------------------------------------- state

class State:
    """One position. Treat instances as immutable: moves return a new State.

    Attributes:
        pawns: ``(cell, cell)`` for Player 1 and Player 2. See
            :func:`square_token` to render one, or :meth:`pawn_square`.
        walls_h, walls_v: frozensets of occupied wall slots, ``wr * 8 + wc``.
        walls_left: ``(int, int)`` walls still in hand.
        turn: 0 if Player 1 is to move, 1 if Player 2.
        ply: number of moves played so far.
    """

    __slots__ = ("pawns", "walls_h", "walls_v", "walls_left", "turn", "ply",
                 "_passable", "_dist", "_tokens")

    def __init__(self, pawns: tuple[int, int] = START_CELLS,
                 walls_h: Iterable[int] = (), walls_v: Iterable[int] = (),
                 walls_left: tuple[int, int] = (WALLS_PER_PLAYER,) * 2,
                 turn: int = 0, ply: int = 0):
        self.pawns = (int(pawns[0]), int(pawns[1]))
        self.walls_h = frozenset(walls_h)
        self.walls_v = frozenset(walls_v)
        self.walls_left = (int(walls_left[0]), int(walls_left[1]))
        self.turn = int(turn)
        self.ply = int(ply)
        self._passable: list[bool] | None = None
        self._dist: dict[int, list[int]] = {}
        self._tokens: frozenset[str] | None = None

    # -- reading the position ----------------------------------------------

    @property
    def passable(self) -> list[bool]:
        """``[cell * 4 + direction] -> bool``: is that single step available?"""
        if self._passable is None:
            self._passable = _passable_for(self.walls_h, self.walls_v)
        return self._passable

    def pawn_square(self, player: int) -> str:
        """Where a pawn stands, as a token: ``state.pawn_square(0) == 'e1'``."""
        return square_token(self.pawns[player])

    def wall_tokens(self) -> set[str]:
        """Every wall on the board, as tokens. Ownership is not recorded."""
        return ({wall_token(HORIZONTAL, *divmod(s, W)) for s in self.walls_h} |
                {wall_token(VERTICAL, *divmod(s, W)) for s in self.walls_v})

    def distance_map(self, player: int) -> list[int]:
        """Steps from every cell to ``player``'s goal rank; ``INF`` if cut off."""
        if player not in self._dist:
            goal = GOAL_ROWS[player] * N        # a rank is nine contiguous cells
            self._dist[player] = _distances(self.passable, range(goal, goal + N))
        return self._dist[player]

    def distance_from(self, cell: int) -> list[int]:
        """Steps from ``cell`` to every other cell; ``INF`` if unreachable."""
        return _distances(self.passable, (cell,))

    def distance_to_goal(self, player: int) -> int | None:
        """Length of ``player``'s shortest path, or None if there is none.

        None cannot happen in a legal game — walls that would cause it are
        rejected — but it is the honest answer for a hand-built position.
        """
        d = self.distance_map(player)[self.pawns[player]]
        return None if d >= INF else d

    def shortest_path(self, player: int) -> list[int] | None:
        """One shortest path as a list of cells, pawn first, goal last."""
        dist = self.distance_map(player)
        cell = self.pawns[player]
        if dist[cell] >= INF:
            return None
        path = [cell]
        passable = self.passable
        while dist[cell] > 0:
            base = cell * 4
            for d in range(4):
                if not passable[base + d]:
                    continue
                nxt = _NEIGHBOUR[base + d]
                if dist[nxt] == dist[cell] - 1:
                    path.append(nxt)
                    cell = nxt
                    break
            else:       # a finite distance always has a neighbour one closer
                break
        return path

    def __repr__(self) -> str:
        return (f"<State ply {self.ply}, p1 {self.pawn_square(0)}, "
                f"p2 {self.pawn_square(1)}, walls {self.walls_left[0]}/"
                f"{self.walls_left[1]}, player {self.turn + 1} to move>")

    def render(self) -> str:
        """The board as text, rank 9 at the top. For eyeballing a position."""
        lines = []
        for row in range(N - 1, -1, -1):
            cells, below = [], []
            for col in range(N):
                cell = cell_of(row, col)
                cells.append("1" if cell == self.pawns[0] else
                             "2" if cell == self.pawns[1] else ".")
                if col < W:
                    cells.append("|" if not self.passable[cell * 4 + RIGHT] else " ")
                below.append("-" if row and not self.passable[cell * 4 + DOWN] else " ")
                if col < W:
                    below.append(" ")
            lines.append(f"{row + 1} " + "".join(cells))
            if row:
                lines.append("  " + "".join(below))
        lines.append("  " + " ".join(FILES))
        lines.append(f"  walls in hand: {self.walls_left[0]} / {self.walls_left[1]}"
                     f"   player {self.turn + 1} to move")
        return "\n".join(lines)


def initial_state() -> State:
    """The starting position: pawns on e1 and e9, ten walls each, Player 1 to move."""
    return State()


# --------------------------------------------------------- move generation

def _pawn_destinations(state: State) -> set[int]:
    """Cells the side to move may step, jump, or side-step to."""
    passable = state.passable
    cur = state.pawns[state.turn]
    opp = state.pawns[1 - state.turn]
    out: set[int] = set()
    for d in range(4):
        if not passable[cur * 4 + d]:
            continue
        nxt = _NEIGHBOUR[cur * 4 + d]
        if nxt != opp:
            out.add(nxt)
            continue
        # The opponent is in the way. Jump straight over them if the square
        # behind is on the board and not walled off; otherwise go around, to
        # whichever sides are open.
        if passable[nxt * 4 + d]:
            out.add(_NEIGHBOUR[nxt * 4 + d])
        else:
            for pd in PERPENDICULAR[d]:
                if passable[nxt * 4 + pd]:
                    out.add(_NEIGHBOUR[nxt * 4 + pd])
    return out


def _wall_conflicts(state: State, orientation: int, wr: int, wc: int) -> bool:
    """True if the wall overlaps or crosses one already on the board."""
    if orientation == HORIZONTAL:
        return (_slot(wr, wc) in state.walls_h
                or (wc > 0 and _slot(wr, wc - 1) in state.walls_h)
                or (wc < W - 1 and _slot(wr, wc + 1) in state.walls_h)
                or _slot(wr, wc) in state.walls_v)
    return (_slot(wr, wc) in state.walls_v
            or (wr > 0 and _slot(wr - 1, wc) in state.walls_v)
            or (wr < W - 1 and _slot(wr + 1, wc) in state.walls_v)
            or _slot(wr, wc) in state.walls_h)


def _legal_wall_slots(state: State) -> list[tuple[int, int, int]]:
    """Every legal ``(orientation, wr, wc)`` for the side to move.

    Checking the connectivity rule for all 128 candidates would mean 256 floods
    per move. Instead, take one concrete shortest path per player as a witness:
    a wall that blocks no edge of a player's witnessed path cannot possibly cut
    that player off, since the witness survives it. Only the handful of walls
    that do cross a path need the flood, and only for the player crossed.
    """
    if state.walls_left[state.turn] <= 0:
        return []

    witness = []
    for player in (0, 1):
        path = state.shortest_path(player) or []
        witness.append({tuple(sorted(pair)) for pair in zip(path, path[1:])})

    passable = state.passable
    out: list[tuple[int, int, int]] = []
    for orientation in (HORIZONTAL, VERTICAL):
        for wr in range(W):
            for wc in range(W):
                if _wall_conflicts(state, orientation, wr, wc):
                    continue
                edges = wall_edges(orientation, wr, wc)
                cuts = [p for p in (0, 1)
                        if edges[0] in witness[p] or edges[1] in witness[p]]
                if cuts:
                    # Overlapping and crossing walls are already rejected, so
                    # no other wall clears these same steps and putting them
                    # back afterwards restores the position exactly.
                    steps = _wall_step_indices(orientation, wr, wc)
                    for i in steps:
                        passable[i] = False
                    ok = all(_reaches_row(passable, state.pawns[p], GOAL_ROWS[p])
                             for p in cuts)
                    for i in steps:
                        passable[i] = True
                    if not ok:
                        continue
                out.append((orientation, wr, wc))
    return out


def legal_tokens(state: State) -> set[str]:
    """Every move token legal in this position.

    Empty once somebody has won: a finished game has no moves, and the server
    rejects any that are offered.
    """
    if state._tokens is None:
        if winner(state) is not None:
            state._tokens = frozenset()
        else:
            state._tokens = frozenset(
                {square_token(cell) for cell in _pawn_destinations(state)} |
                {wall_token(o, wr, wc) for o, wr, wc in _legal_wall_slots(state)})
    return set(state._tokens)


def apply_token(state: State, token: str) -> State:
    """The position after the side to move plays ``token``.

    Raises :class:`IllegalMove` — with a reason — for anything else.
    """
    if winner(state) is not None:
        raise IllegalMove(f"{token}: the game is already over")
    mover = state.turn

    if is_wall_token(token):
        parsed = parse_wall(token)
        if parsed is None:
            raise IllegalMove(f"{token!r}: not a wall token")
        orientation, wr, wc = parsed
        if state.walls_left[mover] <= 0:
            raise IllegalMove(f"{token}: player {mover + 1} has no walls left")
        if _wall_conflicts(state, orientation, wr, wc):
            raise IllegalMove(f"{token}: overlaps or crosses a wall already placed")
        slot = _slot(wr, wc)
        walls_left = list(state.walls_left)
        walls_left[mover] -= 1
        after = State(
            pawns=state.pawns,
            walls_h=state.walls_h | {slot} if orientation == HORIZONTAL else state.walls_h,
            walls_v=state.walls_v if orientation == HORIZONTAL else state.walls_v | {slot},
            walls_left=(walls_left[0], walls_left[1]),
            turn=1 - mover, ply=state.ply + 1)
        for player in (0, 1):
            if after.distance_to_goal(player) is None:
                raise IllegalMove(
                    f"{token}: would leave player {player + 1} no path to their goal")
        return after

    cell = parse_square(token)
    if cell is None:
        raise IllegalMove(f"{token!r}: not a move token")
    if cell not in _pawn_destinations(state):
        raise IllegalMove(f"{token}: not a legal pawn move here")
    pawns = list(state.pawns)
    pawns[mover] = cell
    return State(pawns=(pawns[0], pawns[1]),
                 walls_h=state.walls_h, walls_v=state.walls_v,
                 walls_left=state.walls_left, turn=1 - mover, ply=state.ply + 1)


def winner(state: State) -> int | None:
    """0 if Player 1 has won, 1 if Player 2 has, None while the game runs."""
    if state.pawns[0] // N == GOAL_ROWS[0]:
        return 0
    if state.pawns[1] // N == GOAL_ROWS[1]:
        return 1
    return None


def replay(tokens: Iterable[str], state: State | None = None) -> State:
    """Apply a token sequence — a game record — from the start, or from ``state``."""
    out = initial_state() if state is None else state
    for token in tokens:
        out = apply_token(out, token)
    return out
