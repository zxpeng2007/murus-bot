# palisade-bot

Write an engine for the wall game and put it on the ladder at
[murus.net](https://murus.net).

Palisade is a lichess-style arena for the wall game — a 9×9 board, two pawns
racing to the opposite rank, twenty walls in the way. People play in the
browser, engines play through the HTTP API, and both share one rating list.
This repository is the client side: everything you need to connect an engine,
and nothing you need to run the server.

The engine side of the site — the ladder, who is playing, and a quickstart —
is at [murus.net/#/engines](https://murus.net/#/engines).

Three pieces, useful separately.

| module | what it does |
|---|---|
| `palisade_bot.rules` | the game itself, in plain Python: legal moves, positions, notation, shortest paths. No dependencies. |
| `palisade_bot.client` | one method per API endpoint, plus the two ndjson streams. |
| `palisade_bot.runner` | the play loop: challenges, seeks, reconnects, the clock. You supply one function. |

The only dependency is [httpx](https://www.python-httpx.org/). Python 3.11+.

## Install

```sh
git clone https://github.com/zxpeng2007/palisade-bot
cd palisade-bot
pip install -e .
```

## Get a token

Three calls. Register, declare the account as an engine, mint a token — the
first two need the session cookie, so keep a jar between them.

```sh
# 1. Register. Username 2-20 characters of [A-Za-z0-9_-].
curl -c jar.txt -H 'Content-Type: application/json' \
     -d '{"username":"my-engine","password":"a-long-password"}' \
     https://murus.net/api/register

# 2. Declare it an engine. Only possible while it has no rated games, and
#    required before it plays: opponents are entitled to see the BOT tag.
curl -b jar.txt -X POST https://murus.net/api/bot/upgrade

# 3. Mint a token. The plaintext is shown exactly once.
curl -b jar.txt -H 'Content-Type: application/json' \
     -d '{"name":"my-engine","scopes":["play","bot"]}' \
     https://murus.net/api/token
# -> {"token":"pal_..."}

rm jar.txt
```

A token can never mint another token, so keep the password if you will want
more of them later.

## Play a game

```sh
export PALISADE_TOKEN=pal_...
python examples/random_bot.py --seek 300+3 --games 1
```

That opens a seek for a 5+3 game and plays whoever turns up, then exits.
Challenge it from the site, or start a second bot with the same time control
in another terminal and let the two of them find each other:

```sh
PALISADE_TOKEN=pal_other python examples/greedy_bot.py --seek 300+3 --games 1
```

Both examples take `--server` (default `https://murus.net`), `--accept`
(`all` | `rated` | `casual` | `none`), `--games`, and `--seek` repeated for
several time controls. Suffix a seek with `:casual` for an unrated game.

## Write your own bot

One function. It is handed the position, which seat you are, and the clock;
it returns a move token.

```python
import os
from palisade_bot import BotRunner, PalisadeClient, parse_seek, rules


def choose(state, seat, clock):
    """Return a move token, or None if there is nothing to play."""
    tokens = sorted(rules.legal_tokens(state))
    for token in tokens:
        if rules.winner(rules.apply_token(state, token)) == seat:
            return token                       # take a win when offered
    # Your engine goes here. This one just walks towards its own rank.
    distance = state.distance_map(seat)
    steps = [t for t in tokens if not rules.is_wall_token(t)]
    return min(steps, key=lambda t: distance[rules.parse_square(t)])


with PalisadeClient(os.environ["PALISADE_TOKEN"]) as client:
    BotRunner(client, choose, seeks=[parse_seek("300+3")]).run()
```

`seat` is 0 for Player 1 and 1 for Player 2 — the same numbering the rules
module uses everywhere, so `state.distance_to_goal(seat)` is your distance and
`1 - seat` is your opponent.

The runner plays one game at a time, declines challenges while it is busy,
reconnects to both streams with backoff, and rebuilds the position by replaying
the server's move list rather than trusting its own bookkeeping. If `choose`
raises or returns an illegal move it resigns rather than letting your opponent
watch the clock run out; pass `resign_on_error=False` if you would rather it
did not.

### The position

```python
rules.initial_state()                 # the starting position
rules.legal_tokens(state)             # set of legal tokens; empty once won
rules.apply_token(state, token)       # -> the position after it (never mutates)
rules.winner(state)                   # 0, 1, or None
rules.replay("e2,e8,e3".split(","))   # a game record -> a position

state.pawns                           # (cell, cell), cell = row * 9 + col
state.pawn_square(0)                  # 'e1'
state.walls_left                      # (10, 10)
state.wall_tokens()                   # {'hd3', 'vf6', ...} on the board
state.turn                            # 0 or 1
state.ply                             # moves played so far

state.distance_to_goal(player)        # steps along the shortest path, or None
state.distance_map(player)            # that distance from every cell
state.distance_from(cell)             # steps from a cell to every other
state.shortest_path(player)           # one shortest path, as cells
state.render()                        # the board as text, for eyeballing
```

Illegal moves raise `rules.IllegalMove` with a reason. States are immutable, so
you can keep them, hash your own keys off them, and search without copying.

### The clock

`clock.remaining` and `clock.opponent_remaining` are seconds, and
`clock.initial` / `clock.increment` are the time control.
`clock.budget(target=3.0)` turns that into how long this move is worth: at most
`target`, at most a twentieth of what is left, plus most of the increment back.
Convert it to depth, simulations, or a deadline however your engine prefers.

## Notation

| token | meaning |
|---|---|
| `e2` | pawn move, named by its destination square — jumps included |
| `hd3` | horizontal wall on the edge between ranks 3 and 4, spanning files d and e |
| `vd3` | vertical wall on the edge between files d and e, spanning ranks 3 and 4 |

Files `a`–`i` run left to right and ranks `1`–`9` bottom to top, from Player
1's point of view. Player 1 starts on `e1` and wins on reaching rank 9; Player
2 starts on `e9` and wins on reaching rank 1. Wall anchors run `a1`–`h8` —
there is no ninth file or rank of wall slots. A game record is the tokens in
ply order, comma separated:

```
e2,e8,e3,e7,e4,e6,hc3,vd4,e5,e4,...
```

Note that `h2` is a pawn move and `hh2` is a wall: length tells them apart, not
the leading letter.

## The rules

Each turn, step your pawn one square orthogonally or place one of your ten
walls. A wall spans two cells; walls may not overlap or cross, and no wall may
leave *either* player without a path to their goal. A pawn facing the opponent
jumps over them — and if the square directly behind is off the board or walled
off, steps diagonally past them instead, to either side that is open. First
pawn to the far rank wins. There are no draws.

The API and the full specification live in
[API.md](https://github.com/zxpeng2007/palisade/blob/main/API.md). Where this
library and that document disagree, the document is right.

## Fair play

Engines are welcome here — that is the point of the site — so the rule is not
"no engines", it is **be what your account says you are**. Declare the account
with `POST /api/bot/upgrade` before it plays, keep one account per engine, and
do not play a human account with an engine. Any strength, any hardware, any
opening book, no limits. The whole policy is at
[murus.net/#/fairplay](https://murus.net/#/fairplay); it is short and worth the
two minutes.

## Examples

| file | needs | what it is |
|---|---|---|
| `examples/random_bot.py` | httpx | a legal move, chosen at random. The right thing to run first when checking a token or a firewall. |
| `examples/greedy_bot.py` | httpx | shortest-path racing with opportunistic walls. Not strong, but a real opponent. |
| `examples/alphazero_bot.py` | a private engine package, PyTorch, and a checkpoint of your own | how to hang a search engine off the runner. No weights are shipped here. |

## Tests

```sh
python -m pytest -q
```

`tests/test_rules.py` covers the rules directly and replays a real game to its
recorded result. It also contains a differential test that plays two thousand
random games against the private engine behind the house bot, comparing every
position and every legal move set; that one skips itself when the engine is not
installed, which for everyone but the maintainers is always. Set
`PALISADE_BOT_DIFF_GAMES` to change how many games it plays.

## Licence

MIT. See [LICENSE](LICENSE).
