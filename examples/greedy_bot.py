"""A shortest-path bot: it runs, and it walls you when walling pays.

The whole engine is two ideas.

*The race.* Both players are running a shortest path to their own rank, so the
position is worth ``opponent's distance - my distance`` to me. Every candidate
move is scored by what that difference becomes.

*When a wall is worth it.* A pawn move buys exactly one step of progress, so a
wall has to beat that: it must cost the opponent at least two steps more than
it costs me, or it is a wasted turn and a wasted wall. That falls out of the
scoring rather than being a special case — a wall that gains one is simply
worth less than walking. Only walls that cut an edge of *some* shortest path
of the opponent's can lengthen it at all, so those are the only ones examined.

Not a strong engine — it never looks past its own move, so it can be lured into
spending walls that a deeper search would keep — but it plays a sensible game,
finishes what it starts, and gives a new engine something real to beat.

    export PALISADE_TOKEN=pal_...
    python examples/greedy_bot.py --seek 300+3

Requires only httpx.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

from palisade_bot import BotRunner, PalisadeClient, PalisadeError, parse_seek, rules

log = logging.getLogger("greedy_bot")


class GreedyBot:
    """Shortest-path racing with opportunistic walls.

    Args:
        seed: seed for tie-breaking between equally good moves.
        wall_slack: how far ahead in the race the bot will still consider
            spending a wall. Walls are a finite resource and a bot that is
            winning the race does not need them.
    """

    def __init__(self, seed: int | None = None, wall_slack: int = 2):
        self.rng = random.Random(seed)
        self.wall_slack = wall_slack

    # -- the callback the runner wants --------------------------------------

    def choose(self, state: rules.State, seat: int, clock) -> str | None:
        me, opponent = seat, 1 - seat
        legal = rules.legal_tokens(state)
        if not legal:
            return None

        step, my_distance = self._best_step(state, me, legal)
        if step is not None and my_distance == 0:
            return step                              # the goal, this move

        opponent_distance = state.distance_to_goal(opponent)
        if step is None:                             # boxed in; a wall it is
            wall, _ = self._best_wall(state, me, opponent, legal)
            return wall or sorted(legal)[0]

        # What the race looks like after each option. Walking always subtracts
        # one from my distance and leaves the opponent's alone; a wall does the
        # reverse, which is why it needs to gain two to be worth the tempo.
        walking = opponent_distance - my_distance
        if (state.walls_left[me] > 0
                and opponent_distance <= my_distance + self.wall_slack):
            wall, walling = self._best_wall(state, me, opponent, legal)
            if wall is not None and walling > walking:
                return wall
        return step

    # -- pawn moves ---------------------------------------------------------

    def _best_step(self, state: rules.State, me: int,
                   legal: set[str]) -> tuple[str | None, int]:
        """The pawn move that gets closest to my goal, and that distance.

        Walls do not move, so every destination can be read straight off the
        one distance map instead of building a position per candidate.
        """
        distance = state.distance_map(me)
        best: list[str] = []
        best_distance = rules.INF
        for token in legal:
            if rules.is_wall_token(token):
                continue
            cell = rules.parse_square(token)
            if distance[cell] < best_distance:
                best_distance, best = distance[cell], [token]
            elif distance[cell] == best_distance:
                best.append(token)
        if not best:
            return None, rules.INF
        return self.rng.choice(sorted(best)), best_distance

    # -- walls --------------------------------------------------------------

    def _best_wall(self, state: rules.State, me: int, opponent: int,
                   legal: set[str]) -> tuple[str | None, float]:
        """The most valuable wall and the race score it leaves behind."""
        best, best_score = None, -rules.INF
        best_damage = -rules.INF
        for token in self._wall_candidates(state, opponent, legal):
            after = rules.apply_token(state, token)
            # Legal walls always leave both players a path, so neither distance
            # can be None here.
            mine = after.distance_to_goal(me)
            theirs = after.distance_to_goal(opponent)
            score = theirs - mine
            damage = theirs - state.distance_to_goal(opponent)
            if (score, damage) > (best_score, best_damage):
                best, best_score, best_damage = token, score, damage
        return best, best_score

    def _wall_candidates(self, state: rules.State, opponent: int,
                         legal: set[str]) -> list[str]:
        """Legal walls that could lengthen the opponent's shortest path.

        A wall changes that distance only if it blocks an edge lying on one of
        the opponent's shortest paths. Comparing the distance *from* the pawn
        with the distance *to* the goal identifies those edges exactly: the
        step u->v is on a shortest path when going through it costs the whole
        journey and no more. Typically a dozen walls survive this out of 128,
        which is what makes evaluating every one of them affordable.
        """
        pawn = state.pawns[opponent]
        to_goal = state.distance_map(opponent)
        from_pawn = state.distance_from(pawn)
        total = to_goal[pawn]
        if total >= rules.INF:
            return []

        candidates = []
        for token in legal:
            if not rules.is_wall_token(token):
                continue
            for u, v in rules.wall_edges(*rules.parse_wall(token)):
                if (from_pawn[u] + 1 + to_goal[v] == total
                        or from_pawn[v] + 1 + to_goal[u] == total):
                    candidates.append(token)
                    break
        return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", default="https://murus.net",
                        help="server base URL (default %(default)s)")
    parser.add_argument("--token", default=os.environ.get("PALISADE_TOKEN"),
                        help="API token with the play scope "
                             "(default: $PALISADE_TOKEN)")
    parser.add_argument("--seek", action="append", default=[], metavar="300+3",
                        help="keep a seek with this time control open while "
                             "idle; suffix :casual for unrated; repeat to "
                             "rotate through several")
    parser.add_argument("--accept", default="all",
                        choices=["all", "rated", "casual", "none"],
                        help="which challenges to accept (default %(default)s)")
    parser.add_argument("--games", type=int, default=None,
                        help="stop after this many games (default: never)")
    parser.add_argument("--wall-slack", type=int, default=2,
                        help="consider walls while the opponent is within this "
                             "many steps of my distance (default %(default)s)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for tie-breaking")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    # httpx logs a line per request, which at a move a second is just noise.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not args.token:
        parser.error("no API token: pass --token or set PALISADE_TOKEN")

    bot = GreedyBot(args.seed, wall_slack=args.wall_slack)
    with PalisadeClient(args.token, server=args.server) as client:
        runner = BotRunner(client, bot.choose, accept=args.accept,
                           seeks=[parse_seek(s) for s in args.seek],
                           max_games=args.games)
        try:
            runner.run()
        except KeyboardInterrupt:
            log.info("stopping")
        except PalisadeError as exc:
            log.error("%s", exc)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
