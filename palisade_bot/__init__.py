"""Everything you need to put an engine on a Palisade board.

Three pieces, usable separately:

* :mod:`palisade_bot.rules` — the game itself, in plain Python. Legal moves,
  positions, notation, shortest paths. No dependencies, no server needed.
* :mod:`palisade_bot.client` — one method per API endpoint, plus the two
  ndjson streams.
* :mod:`palisade_bot.runner` — the play loop. Give it a ``choose`` callback
  and it handles challenges, seeks, reconnects and the clock.

The shortest complete bot::

    import random
    from palisade_bot import PalisadeClient, BotRunner, parse_seek, rules

    with PalisadeClient(token="pal_...") as client:
        BotRunner(client,
                  lambda state, seat, clock: random.choice(sorted(rules.legal_tokens(state))),
                  seeks=[parse_seek("300+3")]).run()
"""

from palisade_bot import rules
from palisade_bot.client import DEFAULT_SERVER, PalisadeClient, PalisadeError
from palisade_bot.runner import BotRunner, Clock, Seek, parse_seek

__all__ = [
    "rules", "PalisadeClient", "PalisadeError", "DEFAULT_SERVER",
    "BotRunner", "Clock", "Seek", "parse_seek", "__version__",
]

__version__ = "1.0.0"
