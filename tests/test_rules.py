"""Tests for the rules module.

Everything here runs on a plain checkout with nothing but pytest installed. The
last test is the exception in one direction only: it compares this module move
for move against the private engine that runs the house bot, and skips itself
when that engine is not importable. It is how the maintainers keep the two
implementations honest; it is not something a third party needs to pass.
"""

from __future__ import annotations

import os
import random

import pytest

from murus_bot import rules


def tokens_of(*record: str) -> list[str]:
    """A game record, written as one comma-separated string per fragment."""
    return [t for fragment in record for t in fragment.split(",") if t]


# ------------------------------------------------------------------ notation

def test_initial_position():
    state = rules.initial_state()
    assert state.pawn_square(0) == "e1"
    assert state.pawn_square(1) == "e9"
    assert state.walls_left == (10, 10)
    assert state.turn == 0
    assert state.ply == 0
    assert state.wall_tokens() == set()
    assert rules.winner(state) is None
    # Three steps off e1 (the fourth is off the board) and all 128 walls.
    legal = rules.legal_tokens(state)
    assert {t for t in legal if not rules.is_wall_token(t)} == {"e2", "d1", "f1"}
    assert len(legal) == 3 + 128


def test_square_tokens_round_trip():
    seen = set()
    for cell in range(rules.NUM_CELLS):
        token = rules.square_token(cell)
        assert rules.parse_square(token) == cell
        seen.add(token)
    assert len(seen) == 81
    assert rules.square_token(0) == "a1"
    assert rules.square_token(80) == "i9"
    assert rules.parse_square("e1") == 4
    assert rules.parse_square("e9") == 76


def test_wall_tokens_round_trip():
    seen = set()
    for orientation in (rules.HORIZONTAL, rules.VERTICAL):
        for wr in range(8):
            for wc in range(8):
                token = rules.wall_token(orientation, wr, wc)
                assert rules.parse_wall(token) == (orientation, wr, wc)
                assert rules.is_wall_token(token)
                seen.add(token)
    assert len(seen) == 128
    # Anchors stop at h8: there is no ninth file or rank of wall slots.
    assert rules.wall_token(rules.HORIZONTAL, 0, 0) == "ha1"
    assert rules.wall_token(rules.VERTICAL, 7, 7) == "vh8"
    assert rules.parse_wall("hi1") is None
    assert rules.parse_wall("ha9") is None


def test_a_wall_letter_is_not_a_file_letter():
    # "h2" is a pawn move to h2; "hh2" is a wall. Length tells them apart.
    assert not rules.is_wall_token("h2")
    assert rules.parse_square("h2") == rules.cell_of(1, 7)
    assert rules.is_wall_token("hh2")


# --------------------------------------------------------------- pawn moves

def test_pawn_steps_are_orthogonal_and_stop_at_the_edge():
    state = rules.replay(tokens_of("d1,e8,c1,e7,b1,e6,a1,e5"))
    assert state.pawn_square(0) == "a1"
    steps = {t for t in rules.legal_tokens(state) if not rules.is_wall_token(t)}
    assert steps == {"a2", "b1"}       # the corner has two neighbours


def test_walls_block_steps():
    # hd1 lies between ranks 1 and 2, spanning files d and e.
    state = rules.replay(tokens_of("hd1"))
    assert state.turn == 1
    steps = {t for t in rules.legal_tokens(state) if not rules.is_wall_token(t)}
    assert steps == {"e8", "d9", "f9"}          # Player 2 is untouched by it
    forward = rules.replay(tokens_of("hd1,e8"))
    steps = {t for t in rules.legal_tokens(forward) if not rules.is_wall_token(t)}
    assert "e2" not in steps           # Player 1 can no longer step off e1
    assert steps == {"d1", "f1"}


def test_straight_jump_over_the_opponent():
    state = rules.replay(tokens_of("e2,e8,e3,e7,e4,e6,e5"))
    assert (state.pawn_square(0), state.pawn_square(1)) == ("e5", "e6")
    assert state.turn == 1             # Player 2 faces Player 1
    steps = {t for t in rules.legal_tokens(state) if not rules.is_wall_token(t)}
    assert "e4" in steps               # over the top
    assert "e5" not in steps           # never onto the opponent
    assert steps == {"e4", "e7", "d6", "f6"}


def test_diagonal_when_the_square_behind_is_walled_off():
    state = rules.replay(tokens_of("e2,e8,e3,e7,e4,e6,e5,he6"))
    assert (state.pawn_square(0), state.pawn_square(1)) == ("e5", "e6")
    assert state.turn == 0
    steps = {t for t in rules.legal_tokens(state) if not rules.is_wall_token(t)}
    assert "e7" not in steps           # the jump is walled off
    assert {"d6", "f6"} <= steps       # so go around, either side
    assert steps == {"d5", "f5", "e4", "d6", "f6"}


def test_diagonal_when_the_jump_would_leave_the_board():
    state = rules.replay(tokens_of(
        "e2,d9,e3,e9,e4,d9,e5,e9,e6,d9,e7,e9,ha1,d9,e8,e9"))
    assert (state.pawn_square(0), state.pawn_square(1)) == ("e8", "e9")
    steps = {t for t in rules.legal_tokens(state) if not rules.is_wall_token(t)}
    assert "e9" not in steps
    assert {"d9", "f9"} <= steps       # there is no square behind e9
    assert rules.winner(rules.apply_token(state, "f9")) == 0


# -------------------------------------------------------------------- walls

def test_walls_may_not_overlap_or_cross():
    state = rules.replay(tokens_of("hd3"))
    assert state.wall_tokens() == {"hd3"}
    legal = rules.legal_tokens(state)
    for blocked in ("hd3", "hc3", "he3", "vd3"):
        assert blocked not in legal, blocked
        with pytest.raises(rules.IllegalMove):
            rules.apply_token(state, blocked)
    for allowed in ("hb3", "hf3", "hd2", "hd4", "vc3", "ve3"):
        assert allowed in legal, allowed


def test_a_wall_may_not_seal_the_opponent_off():
    # Player 1 in the corner behind ha1: only the a/b pocket is left, and vb1
    # would close it.
    state = rules.replay(tokens_of("d1,e8,c1,e7,b1,e6,a1,e5,ha1"))
    assert state.turn == 1
    assert state.distance_to_goal(0) is not None
    assert "vb1" not in rules.legal_tokens(state)
    with pytest.raises(rules.IllegalMove, match="no path"):
        rules.apply_token(state, "vb1")


def test_a_wall_may_not_seal_yourself_off_either():
    state = rules.replay(tokens_of("d1,e8,c1,e7,b1,e6,a1,e5,ha1,e4"))
    assert state.turn == 0
    assert "vb1" not in rules.legal_tokens(state)
    with pytest.raises(rules.IllegalMove, match="no path"):
        rules.apply_token(state, "vb1")


def test_no_walls_in_hand_means_no_wall_moves():
    state = rules.State(walls_left=(0, 10))
    assert not any(rules.is_wall_token(t) for t in rules.legal_tokens(state))
    with pytest.raises(rules.IllegalMove, match="no walls left"):
        rules.apply_token(state, "hd3")
    # The other player still has theirs.
    after = rules.apply_token(state, "e2")
    assert sum(rules.is_wall_token(t) for t in rules.legal_tokens(after)) == 128


def test_malformed_and_illegal_tokens_are_refused():
    state = rules.initial_state()
    for token in ("", "e", "e0", "z9", "j1", "e2e2", "hi1", "ha9", "xd3"):
        with pytest.raises(rules.IllegalMove):
            rules.apply_token(state, token)
    for token in ("e3", "d2", "e9", "a1"):      # legal squares, not legal moves
        with pytest.raises(rules.IllegalMove, match="not a legal pawn move"):
            rules.apply_token(state, token)


# ------------------------------------------------------------------ scoring

def test_distances_and_paths():
    state = rules.initial_state()
    assert state.distance_to_goal(0) == 8
    assert state.distance_to_goal(1) == 8
    path = state.shortest_path(0)
    assert path[0] == state.pawns[0]
    assert len(path) == 9
    assert path[-1] // rules.N == rules.GOAL_ROWS[0]
    # A wall in the way costs steps, but never all of them: e1 has to set off
    # sideways before it can start climbing.
    walled = rules.replay(tokens_of("hd1"))
    assert walled.distance_to_goal(0) == 9


def test_a_finished_game_has_no_moves():
    state = rules.replay(tokens_of(
        "e2,d9,e3,e9,e4,d9,e5,e9,e6,d9,e7,e9,ha1,d9,e8,e9,f9"))
    assert rules.winner(state) == 0
    assert rules.legal_tokens(state) == set()
    with pytest.raises(rules.IllegalMove, match="already over"):
        rules.apply_token(state, "e9")


# ------------------------------------------------------------- a real game

# Played on murus.net. Player 2 wins by a nose, having spent every wall.
QJ4EC3 = ("e2,e8,e3,e7,e4,e6,hc3,vd4,e5,e4,e6,e3,he2,he3,e7,hd7,vd2,hh2,va3,"
          "hg3,f7,hf7,hh4,hh7,vf1,vg4,hg1,f3,e7,vc6,e6,hb5,d6,g3,d5,g2,c5,h2,"
          "b5,i2,a5,i1")


def test_a_real_game_replays_to_its_recorded_result():
    moves = tokens_of(QJ4EC3)
    assert len(moves) == 42

    state = rules.initial_state()
    for ply, token in enumerate(moves, start=1):
        assert rules.winner(state) is None, f"game ended early, at ply {ply}"
        assert token in rules.legal_tokens(state), f"ply {ply}: {token}"
        assert state.turn == (ply - 1) % 2
        state = rules.apply_token(state, token)

    assert rules.winner(state) == 1                 # Player 2
    assert state.pawn_square(0) == "a5"
    assert state.pawn_square(1) == "i1"
    assert state.walls_left == (3, 0)
    assert state.ply == 42
    assert len(state.wall_tokens()) == 17


# ---------------------------------------------------- differential testing

DIFF_GAMES = int(os.environ.get("MURUS_BOT_DIFF_GAMES", "2000"))
DIFF_PLY_CAP = 400
# Random play is mostly wall placement -- 128 wall actions against a handful of
# pawn moves -- which spends the walls in the first few plies and then wanders
# for hundreds more. Favouring pawn moves finishes games and, more to the
# point, walks the pawns into each other, which is where jumps, diagonals and
# the board edges live.
DIFF_PAWN_BIAS = 0.6


def _engine_tokens(fr, np, engine_state, scratch) -> dict[str, int]:
    """The engine's legal moves as ``token -> action``, in its own terms."""
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    fr.legal_mask(engine_state, mask, scratch)
    pawn = fr.IDX_P0 if int(engine_state[fr.IDX_TURN]) == 0 else fr.IDX_P1
    out: dict[str, int] = {}
    for action in np.nonzero(mask)[0]:
        action = int(action)
        if action < fr.MOVE_BASE:
            orientation, slot = divmod(action, fr.NUM_WALL_SLOTS)
            out[rules.wall_token(orientation, *divmod(slot, 8))] = action
        else:
            after = engine_state.copy()
            fr.apply_action(after, action, scratch)
            out[rules.square_token(int(after[pawn]))] = action
    return out


def _ours(state) -> tuple:
    return (state.pawns, sorted(state.walls_h), sorted(state.walls_v),
            state.walls_left, state.turn)


def _theirs(fr, np, engine_state) -> tuple:
    walls_h = np.nonzero(engine_state[fr.WH_OFF:fr.WH_OFF + 64])[0]
    walls_v = np.nonzero(engine_state[fr.WV_OFF:fr.WV_OFF + 64])[0]
    return ((int(engine_state[fr.IDX_P0]), int(engine_state[fr.IDX_P1])),
            [int(i) for i in walls_h], [int(i) for i in walls_v],
            (int(engine_state[fr.IDX_WL0]), int(engine_state[fr.IDX_WL1])),
            int(engine_state[fr.IDX_TURN]))


def test_agrees_with_the_reference_engine():
    """Play random games against the engine that runs the house bot.

    Every position is compared in full, and every legal move set, before a move
    is drawn from ours and applied to both. Any disagreement about the rules --
    a jump, a wall that seals someone in, an off-by-one in notation -- shows up
    as a mismatch within a few plies.
    """
    fr = pytest.importorskip(
        "quoridor.fastrules",
        reason="the private engine is not installed, so there is nothing to "
               "compare against; every other test covers the rules directly")
    np = pytest.importorskip("numpy")
    fr.warmup()

    rng = random.Random(20260801)
    scratch = fr.make_scratch()
    plies = finished = 0

    for game in range(DIFF_GAMES):
        ours = rules.initial_state()
        theirs = fr.initial_state()
        for _ in range(DIFF_PLY_CAP):
            if rules.winner(ours) is not None:
                finished += 1
                break
            where = f"game {game}, ply {ours.ply}"
            assert _ours(ours) == _theirs(fr, np, theirs), where

            mine = rules.legal_tokens(ours)
            engine = _engine_tokens(fr, np, theirs, scratch)
            assert mine == set(engine), f"{where}: {sorted(mine ^ set(engine))}"

            steps = sorted(t for t in mine if not rules.is_wall_token(t))
            walls = sorted(t for t in mine if rules.is_wall_token(t))
            pool = steps if (not walls or rng.random() < DIFF_PAWN_BIAS) else walls
            token = rng.choice(pool or walls)

            ours = rules.apply_token(ours, token)
            fr.apply_action(theirs, engine[token], scratch)
            plies += 1

        engine_winner = int(fr.winner(theirs))
        assert rules.winner(ours) == (None if engine_winner < 0 else engine_winner)

    assert plies > 20 * DIFF_GAMES, "the games were suspiciously short"
    assert finished > DIFF_GAMES // 2, "hardly any game reached a goal"
