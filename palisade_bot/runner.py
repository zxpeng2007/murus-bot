"""The play loop: everything between the API and your engine.

:class:`BotRunner` owns the streams, the challenge policy, the seeks and the
position bookkeeping. Your engine is one callback::

    def choose(state, seat, clock):
        return "e2"

    BotRunner(client, choose, seeks=[parse_seek("300+3")]).run()

``state`` is a :class:`palisade_bot.rules.State` rebuilt by replaying the
server's move list, ``seat`` is 0 for Player 1 and 1 for Player 2, and
``clock`` is a :class:`Clock` with both remaining times and a budget helper.
Return a move token, or None if you have nothing to play.

The runner plays one game at a time. Challenges arriving while it is busy are
declined, and a game that starts anyway — a parked seek matching in the same
instant — is queued and played next.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

import httpx

from palisade_bot import rules
from palisade_bot.client import PalisadeClient, PalisadeError

log = logging.getLogger(__name__)

#: What a ``choose`` callback looks like.
Choose = Callable[[rules.State, int, "Clock"], "str | None"]

_SEEK_RE = re.compile(r"^(\d+)\s*[+x]\s*(\d+)(?::(rated|casual))?$")


@dataclass(frozen=True)
class Seek:
    """A time control to sit in the lobby with."""

    initial: int
    increment: int
    rated: bool = True

    def __str__(self) -> str:
        return (f"{self.initial}+{self.increment} "
                f"{'rated' if self.rated else 'casual'}")


def parse_seek(spec: str) -> Seek:
    """``'300+3'`` or ``'300+3:casual'`` -> :class:`Seek`. Rated by default."""
    match = _SEEK_RE.match(spec.strip())
    if match is None:
        raise ValueError(f"seek must look like 300+3 or 300+3:casual, not {spec!r}")
    return Seek(int(match.group(1)), int(match.group(2)),
                rated=match.group(3) != "casual")


@dataclass(frozen=True)
class Clock:
    """The time control and what is left of it, in seconds.

    ``remaining`` is None for a game that carries no clock information yet.
    """

    initial: float
    increment: float
    remaining: float | None = None
    opponent_remaining: float | None = None

    def budget(self, target: float = 3.0, divisor: float = 20.0,
               floor: float = 0.05) -> float:
        """Seconds it is safe to spend on this move.

        Two rules. Never spend more than ``1/divisor`` of the remaining time on
        one move, so a long game cannot run the clock down to nothing; and add
        most of the increment back, since a move that costs no more than the
        increment leaves the clock where it was. Losing on time is a real way
        to lose these games, and the only one that is entirely self-inflicted.
        """
        allowance = target
        if self.remaining is not None:
            allowance = min(allowance,
                            self.remaining / divisor + 0.8 * self.increment)
        return max(floor, allowance)


class BotRunner:
    """Connects a ``choose`` callback to a Palisade server.

    Args:
        client: an authenticated :class:`~palisade_bot.client.PalisadeClient`.
        choose: your engine, called when it is your turn.
        accept: challenge policy — ``all``, ``rated``, ``casual`` or ``none``.
        seeks: time controls to rotate through while idle. Empty means the bot
            waits to be challenged instead of looking for games.
        max_games: stop after this many finished games (None: run forever).
        resign_on_error: resign if ``choose`` raises or returns an illegal
            move, rather than letting the clock run out on the opponent. See
            https://murus.net/#/fairplay — "if your engine crashes mid-game,
            resign rather than letting the clock run out on someone".
    """

    ACCEPT_POLICIES = ("all", "rated", "casual", "none")

    def __init__(self, client: PalisadeClient, choose: Choose, *,
                 accept: str = "all", seeks: Sequence[Seek] = (),
                 max_games: int | None = None, resign_on_error: bool = True):
        if accept not in self.ACCEPT_POLICIES:
            raise ValueError(f"accept must be one of {self.ACCEPT_POLICIES}")
        self.client = client
        self.choose = choose
        self.accept = accept
        self.seeks = list(seeks)
        self.max_games = max_games
        self.resign_on_error = resign_on_error

        self.username = ""
        self.games_played = 0
        self._seek_index = 0
        self._seek_open = False
        self._game_thread: threading.Thread | None = None
        self._current_game: str | None = None
        self._pending: deque[dict] = deque()

    # -- account ------------------------------------------------------------

    def verify_account(self) -> dict:
        """Check the token and warn if the account is not declared as a bot."""
        account = self.client.account()
        self.username = account["username"]
        log.info("account %s: rating %s, %s rated games",
                 account["username"], account["rating"], account["games"])
        if not account.get("bot"):
            log.warning(
                "%s is not flagged as a bot, so opponents cannot see they are "
                "playing an engine. POST /api/bot/upgrade while the account "
                "has no rated games (see https://murus.net/#/fairplay).",
                account["username"])
        return account

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        """Stream events and play until ``max_games`` is reached, or forever."""
        self.verify_account()
        try:
            for event in self.client.stream_events(on_connect=self._on_connect):
                self._tick()
                if self._done():
                    return
                self._handle_event(event)
        finally:
            self._cancel_seek()

    def _on_connect(self) -> None:
        log.info("event stream connected")
        # A reconnect loses track of any parked seek. Posting a fresh one
        # simply replaces it server-side, so assume it is gone.
        self._seek_open = False

    def _done(self) -> bool:
        return (self.max_games is not None
                and self.games_played >= self.max_games
                and self._game_thread is None
                and not self._pending)

    def _tick(self) -> None:
        """Idle work, run on every event including keepalives."""
        if self._game_thread is not None and not self._game_thread.is_alive():
            self._game_thread = None
            self._current_game = None
        if self._game_thread is None and self._pending:
            self._start_game(self._pending.popleft())
        if (self._game_thread is None and self.seeks and not self._seek_open
                and not self._done()):
            self._post_seek()

    def _handle_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "challenge":
            self._handle_challenge(event["challenge"])
        elif kind == "gameStart":
            self._handle_game_start(event["game"])
        elif kind == "gameFinish":
            game = event["game"]
            log.info("game %s finished: winner %s (%s)",
                     game["id"], game.get("winner"), game.get("reason"))

    def _handle_game_start(self, game: dict) -> None:
        known = (game["id"] == self._current_game
                 or any(g["id"] == game["id"] for g in self._pending))
        if known:
            return
        if self._game_thread is not None:
            log.info("game %s queued: already playing %s",
                     game["id"], self._current_game)
            self._pending.append(game)
        else:
            self._start_game(game)

    def _handle_challenge(self, challenge: dict) -> None:
        busy = self._game_thread is not None or bool(self._pending)
        wanted = {"all": True, "rated": challenge["rated"],
                  "casual": not challenge["rated"], "none": False}[self.accept]
        action = "accept" if wanted and not busy else "decline"
        clock = challenge["clock"]
        log.info("challenge %s from %s (%s %s+%s): %s%s",
                 challenge["id"], challenge["challenger"],
                 "rated" if challenge["rated"] else "casual",
                 clock["initial"], clock["increment"], action,
                 " (busy)" if wanted and busy else "")
        try:
            if action == "accept":
                self.client.accept_challenge(challenge["id"])
            else:
                self.client.decline_challenge(challenge["id"])
        except (PalisadeError, httpx.HTTPError) as exc:
            # Usually the challenger withdrew in the meantime.
            log.info("could not %s challenge %s: %s", action, challenge["id"], exc)

    # -- seeks --------------------------------------------------------------

    def _post_seek(self) -> None:
        seek = self.seeks[self._seek_index % len(self.seeks)]
        self._seek_index += 1
        try:
            reply = self.client.seek(initial=seek.initial,
                                     increment=seek.increment, rated=seek.rated)
        except (PalisadeError, httpx.HTTPError) as exc:
            log.warning("seek %s failed: %s", seek, exc)
            return
        if reply.get("matched"):
            log.info("seek %s matched immediately", seek)
        else:
            self._seek_open = True
            log.info("seek %s open", seek)

    def _cancel_seek(self) -> None:
        if not self._seek_open:
            return
        self._seek_open = False
        try:
            self.client.cancel_seek()
        except (PalisadeError, httpx.HTTPError):
            pass

    # -- playing ------------------------------------------------------------

    def _start_game(self, game: dict) -> None:
        # A parked seek could otherwise match mid-game and stack a second one.
        self._cancel_seek()
        opponent = game.get("opponent", {})
        log.info("game %s: playing %s against %s (%s)", game["id"],
                 game["color"], opponent.get("username", "?"),
                 opponent.get("rating", "?"))
        self._current_game = game["id"]
        self._game_thread = threading.Thread(
            target=self._play_game, name=f"game-{game['id']}",
            args=(game["id"], 0 if game["color"] == "first" else 1), daemon=True)
        self._game_thread.start()

    def _play_game(self, game_id: str, seat: int) -> None:
        """Follow one game to its end, in its own thread."""
        control = Clock(initial=0.0, increment=0.0)
        answered = -1
        try:
            for message in self.client.stream_game(game_id):
                kind = message.get("type")
                if kind == "gameFull":
                    clock = message.get("clock") or {}
                    control = Clock(initial=float(clock.get("initial", 0)),
                                    increment=float(clock.get("increment", 0)))
                    state = message["state"]
                elif kind == "gameState":
                    state = message
                else:
                    continue
                if state.get("status") != "active":
                    self._report_result(game_id, seat, state)
                    return
                answered = self._consider_move(game_id, seat, state, control,
                                               answered)
        except PalisadeError as exc:
            log.error("game %s: stream refused (%s)", game_id, exc)
        finally:
            self.games_played += 1

    def _report_result(self, game_id: str, seat: int, state: dict) -> None:
        winner = state.get("winner")
        outcome = ("won" if winner == ("first", "second")[seat]
                   else "lost" if winner else "no result")
        log.info("game %s: %s, winner %s (%s) -- we %s", game_id,
                 state.get("status"), winner, state.get("reason"), outcome)

    def _consider_move(self, game_id: str, seat: int, state: dict,
                       control: Clock, answered: int) -> int:
        """Play if it is our turn. Returns the ply we last answered."""
        moves = [t for t in state.get("moves", "").split(",") if t]
        if (len(moves) % 2 == 0) != (seat == 0):
            return answered              # opponent to move
        if len(moves) == answered:
            return answered              # a reconnect resent a state we answered

        try:
            position = rules.replay(moves)
        except rules.IllegalMove as exc:
            # The server and this rules module disagree, which is a bug in one
            # of them. Playing on from a position we cannot trust would be
            # worse than losing this game.
            log.error("game %s: cannot rebuild the position: %s", game_id, exc)
            return answered

        clock = Clock(initial=control.initial, increment=control.increment,
                      remaining=state.get("p1time" if seat == 0 else "p2time"),
                      opponent_remaining=state.get("p2time" if seat == 0 else "p1time"))
        started = time.perf_counter()
        try:
            token = self.choose(position, seat, clock)
        except Exception:
            log.exception("game %s: choose() failed", game_id)
            self._bail_out(game_id)
            return answered
        if token is None:
            return answered

        if token not in rules.legal_tokens(position):
            log.error("game %s: choose() returned %r, which is not legal at ply %d",
                      game_id, token, len(moves) + 1)
            self._bail_out(game_id)
            return answered

        log.info("game %s ply %d: %s (%.1fs%s)", game_id, len(moves) + 1, token,
                 time.perf_counter() - started,
                 f", clock {clock.remaining:.0f}s" if clock.remaining else "")
        return len(moves) if self._post_move(game_id, token) else answered

    def _post_move(self, game_id: str, token: str, attempts: int = 3) -> bool:
        for attempt in range(attempts):
            try:
                self.client.play(game_id, token)
                return True
            except PalisadeError as exc:
                if exc.status == 429:    # rate limited: the only case worth retrying
                    time.sleep(1.0 + attempt)
                    continue
                # Usually a race: the game ended, or the opponent's move landed
                # first. The stream will say so; nothing to do here.
                log.warning("game %s: move %s rejected: %s", game_id, token, exc)
                return False
            except httpx.HTTPError as exc:
                log.warning("game %s: move %s failed: %r", game_id, token, exc)
                time.sleep(1.0 + attempt)
        log.error("game %s: giving up on move %s", game_id, token)
        return False

    def _bail_out(self, game_id: str) -> None:
        """Resign a game this bot can no longer play properly."""
        if not self.resign_on_error:
            return
        log.error("game %s: resigning", game_id)
        try:
            self.client.resign(game_id)
        except (PalisadeError, httpx.HTTPError) as exc:
            log.error("game %s: resignation failed: %s", game_id, exc)
