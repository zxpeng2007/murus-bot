"""HTTP client for the Palisade API.

One thin wrapper per endpoint plus the two ndjson streams, exactly as
documented in API.md (https://github.com/zxpeng2007/palisade/blob/main/API.md).
Nothing here knows the rules of the game; see :mod:`palisade_bot.rules` for
those and :mod:`palisade_bot.runner` for the loop that ties them together.

    from palisade_bot.client import PalisadeClient

    with PalisadeClient(token="pal_...") as client:
        print(client.account())
"""

from __future__ import annotations

import json
import logging
import time
from typing import Iterator

import httpx

DEFAULT_SERVER = "https://murus.net"

#: Keepalives arrive every few seconds on both streams, so a long silence means
#: the connection is dead rather than quiet. Reconnecting is cheap; hanging on a
#: half-open socket while the clock runs is not.
STREAM_IDLE_TIMEOUT = 60.0

log = logging.getLogger(__name__)


class PalisadeError(RuntimeError):
    """An error response from the server. ``status`` is the HTTP status code."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


def _reason(response: httpx.Response) -> str:
    """The server's ``{"error": ...}`` message, or the raw body."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or response.reason_phrase
    if isinstance(body, dict) and "error" in body:
        return str(body["error"])
    return response.text.strip()


class PalisadeClient:
    """Authenticated access to one Palisade server.

    Args:
        token: an API token (``pal_...``) with at least the ``play`` scope.
        server: base URL. Defaults to the public arena at murus.net.
        timeout: seconds for ordinary requests. Streams use their own.
    """

    def __init__(self, token: str, server: str = DEFAULT_SERVER,
                 timeout: float = 15.0):
        if not token:
            raise ValueError("an API token is required")
        self.server = server.rstrip("/")
        self._http = httpx.Client(
            base_url=self.server,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            follow_redirects=True)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "PalisadeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- plumbing -----------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> dict:
        response = self._http.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise PalisadeError(response.status_code, _reason(response))
        if not response.content:
            return {}
        return response.json()

    def get(self, path: str, **kwargs) -> dict:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> dict:
        return self._request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs) -> dict:
        return self._request("DELETE", path, **kwargs)

    # -- account ------------------------------------------------------------

    def account(self) -> dict:
        """``{id, username, bot, rating, rd, games}`` for the token's account."""
        return self.get("/api/account")

    def upgrade_to_bot(self) -> dict:
        """Flag the account as an engine. Only possible before any rated game.

        Required by the fair-play policy: an engine account carries a BOT tag
        so opponents can see what they are sitting down against.
        """
        return self.post("/api/bot/upgrade")

    def user(self, username: str) -> dict:
        return self.get(f"/api/user/{username}")

    # -- challenges and seeks ----------------------------------------------

    def challenge(self, username: str, *, initial: int, increment: int,
                  rated: bool = True, color: str = "random") -> dict:
        return self.post(f"/api/challenge/{username}", json={
            "rated": rated, "color": color,
            "clock": {"initial": initial, "increment": increment}})

    def accept_challenge(self, challenge_id: str) -> dict:
        return self.post(f"/api/challenge/{challenge_id}/accept")

    def decline_challenge(self, challenge_id: str) -> dict:
        return self.post(f"/api/challenge/{challenge_id}/decline")

    def cancel_challenge(self, challenge_id: str) -> dict:
        return self.post(f"/api/challenge/{challenge_id}/cancel")

    def seek(self, *, initial: int, increment: int, rated: bool = True) -> dict:
        """Join the lobby pool. ``{"matched": true}`` if it paired immediately."""
        return self.post("/api/seek", json={
            "rated": rated,
            "clock": {"initial": initial, "increment": increment}})

    def cancel_seek(self) -> dict:
        return self.delete("/api/seek")

    # -- playing ------------------------------------------------------------

    def game(self, game_id: str) -> dict:
        return self.get(f"/api/game/{game_id}")

    def play(self, game_id: str, token: str) -> dict:
        """Submit a move. Raises :class:`PalisadeError` if the server refuses."""
        return self.post(f"/api/game/{game_id}/move/{token}")

    def resign(self, game_id: str) -> dict:
        return self.post(f"/api/game/{game_id}/resign")

    def abort(self, game_id: str) -> dict:
        """Abandon a game before ply 2; no rating change."""
        return self.post(f"/api/game/{game_id}/abort")

    # -- streams ------------------------------------------------------------

    def stream(self, path: str, on_connect=None) -> Iterator[dict]:
        """One connection to an ndjson endpoint, yielding parsed objects.

        ``on_connect`` fires once the server has accepted the request, before
        the first line. Ends when the server closes the stream; raises
        :class:`PalisadeError` on a refusal and ``httpx.HTTPError`` if the
        connection breaks.

        The server's blank keepalive lines surface as ``{"type": "keepalive"}``
        rather than being dropped, so a caller with nothing else to wake it can
        use them as an idle heartbeat. Ignore them if you have no use for them.
        """
        timeout = httpx.Timeout(STREAM_IDLE_TIMEOUT, connect=15.0)
        with self._http.stream("GET", path, timeout=timeout) as response:
            if response.status_code >= 400:
                response.read()
                raise PalisadeError(response.status_code, _reason(response))
            if on_connect is not None:
                on_connect()
            for line in response.iter_lines():
                yield json.loads(line) if line.strip() else {"type": "keepalive"}

    def stream_forever(self, path: str, *, on_connect=None,
                       first_delay: float = 1.0, max_delay: float = 30.0
                       ) -> Iterator[dict]:
        """As :meth:`stream`, reconnecting with exponential backoff.

        A stream can end without warning — the server may sever a slow consumer
        — so an unexpected close is not an error, it is a reconnect. Both
        Palisade streams resend everything needed to resync on the first line
        after reconnecting, so no state is lost.

        4xx responses are raised instead: a rejected token or a game id that
        does not exist will not start working on the next attempt.
        """
        delay = first_delay
        while True:
            try:
                for message in self.stream(path, on_connect=on_connect):
                    delay = first_delay
                    yield message
                log.info("stream %s closed by the server; reconnecting", path)
            except PalisadeError as exc:
                if exc.status < 500:
                    raise
                log.warning("stream %s failed: %s", path, exc)
            except httpx.HTTPError as exc:
                log.warning("stream %s dropped: %r", path, exc)
            time.sleep(delay)
            delay = min(delay * 2, max_delay)

    def stream_events(self, **kwargs) -> Iterator[dict]:
        """Account events: challenges, game starts, game finishes."""
        return self.stream_forever("/api/stream/event", **kwargs)

    def stream_game(self, game_id: str, **kwargs) -> Iterator[dict]:
        """One game: a ``gameFull`` then a ``gameState`` per move."""
        return self.stream_forever(f"/api/game/stream/{game_id}", **kwargs)
