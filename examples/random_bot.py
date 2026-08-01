"""The smallest complete Palisade bot: it plays a legal move, chosen at random.

Useless as an opponent, but it exercises every part of the loop, so it is the
right thing to run first when you are checking a token, a server address, or a
firewall.

    export PALISADE_TOKEN=pal_...
    python examples/random_bot.py --seek 300+3 --games 1

Requires only httpx. See greedy_bot.py for something that tries to win.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

from palisade_bot import BotRunner, PalisadeClient, PalisadeError, parse_seek, rules


class RandomBot:
    """Uniformly random over the legal moves of the position."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose(self, state: rules.State, seat: int, clock) -> str | None:
        # sorted() so a given seed replays identically: legal_tokens is a set,
        # and set iteration order is not a promise.
        tokens = sorted(rules.legal_tokens(state))
        return self.rng.choice(tokens) if tokens else None


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
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for the move choice")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    # httpx logs a line per request, which at a move a second is just noise.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not args.token:
        parser.error("no API token: pass --token or set PALISADE_TOKEN")

    bot = RandomBot(args.seed)
    with PalisadeClient(args.token, server=args.server) as client:
        runner = BotRunner(client, bot.choose, accept=args.accept,
                           seeks=[parse_seek(s) for s in args.seek],
                           max_games=args.games)
        try:
            runner.run()
        except KeyboardInterrupt:
            logging.info("stopping")
        except PalisadeError as exc:
            logging.error("%s", exc)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
