"""Everything you need to put an engine on a Murus board.

Three pieces, usable separately:

* :mod:`murus_bot.rules` — the game itself, in plain Python. Legal moves,
  positions, notation, shortest paths. No dependencies, no server needed.
* :mod:`murus_bot.client` — one method per API endpoint, plus the two
  ndjson streams.
* :mod:`murus_bot.runner` — the play loop. Give it a ``choose`` callback
  and it handles challenges, seeks, reconnects and the clock.

The shortest complete bot::

    import random
    from murus_bot import MurusClient, BotRunner, parse_seek, rules

    with MurusClient(token="mur_...") as client:
        BotRunner(client,
                  lambda state, seat, clock: random.choice(sorted(rules.legal_tokens(state))),
                  seeks=[parse_seek("300+3")]).run()
"""

from murus_bot import rules
from murus_bot.client import DEFAULT_SERVER, MurusClient, MurusError
from murus_bot.runner import BotRunner, Clock, Seek, parse_seek

__all__ = [
    "rules", "MurusClient", "MurusError", "DEFAULT_SERVER",
    "BotRunner", "Clock", "Seek", "parse_seek", "__version__",
]

__version__ = "1.0.0"
