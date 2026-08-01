"""Drive an AlphaZero engine, for people who have one.

This example needs two things that are NOT part of murus-bot and are not
distributed with it:

  * the ``quoridor`` engine package — the private AlphaZero implementation
    behind the house bot on murus.net, which supplies the batched MCTS and the
    network; and
  * PyTorch, plus a trained checkpoint of your own. No weights are shipped
    here. Nothing in this repository will produce one for you.

If you do not have those, this file is still worth reading: it is the pattern
for hanging *any* search engine off the runner. Swap the three engine calls —
load, encode, search — for your own and the rest is unchanged. random_bot.py
and greedy_bot.py run with nothing but httpx.

    export MURUS_TOKEN=mur_...
    python examples/alphazero_bot.py --checkpoint best.pt --seek 300+3

The search time per move comes from the clock: the runner hands over a budget
in seconds, and the bot converts it to a simulation count using a throughput it
measures on this machine at startup. That spans two orders of magnitude between
a rented GPU and a shared vCPU, so measuring beats any constant.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

from murus_bot import BotRunner, MurusClient, MurusError, parse_seek, rules

log = logging.getLogger("alphazero_bot")

ENGINE_HINT = (
    "the AlphaZero engine package is private and is not on PyPI. If you have "
    "access, install it with `pip install -e /path/to/quoridor-alphazero`. If "
    "you do not, run examples/greedy_bot.py instead -- it needs nothing but "
    "httpx -- or point this file at your own engine.")
TORCH_HINT = (
    "install the build that matches your hardware from https://pytorch.org, "
    "e.g. `pip install torch`.")


def _need(module: str, hint: str):
    """Import a module the user may not have, and explain it if they do not."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise SystemExit(f"alphazero_bot needs `{module}`, which is not "
                         f"installed.\n\n{hint}\n\n(import error: {exc})")


def _require_engine() -> None:
    """Fail on the missing engine before anything it depends on.

    Checked with find_spec rather than an import, because importing the engine
    would pull in numpy, numba and torch — and a complaint about one of those
    is not the message someone without access to the engine needs to read.
    """
    if importlib.util.find_spec("quoridor") is None:
        raise SystemExit(f"alphazero_bot needs the `quoridor` engine package, "
                         f"which is not installed.\n\n{ENGINE_HINT}")


class AlphaZeroBot:
    """A neural-network engine behind the ``choose`` interface.

    Args:
        checkpoint: path to a torch checkpoint from the engine repo.
        device: ``auto``, ``cuda`` or ``cpu``.
        think: target seconds per move; a short clock shortens it further.
        max_sims: ceiling on simulations per move. The search tree is
            preallocated to this size, so it costs memory whether used or not.
    """

    # Leaves gathered per network call. A batch of N leaves is selected before
    # any of them is evaluated, so a big batch searches more nodes on staler
    # statistics; 32 is the setting that actually wins games.
    LEAF = 32
    MIN_RATE = 50.0            # simulations/second floor, if calibration fails

    def __init__(self, checkpoint: str, device: str = "auto",
                 think: float = 3.0, max_sims: int = 200_000):
        _require_engine()
        path = Path(checkpoint)
        if not path.is_file():
            raise SystemExit(f"no checkpoint at {path}. Train one with the "
                             f"engine repo, or pass --checkpoint.")

        self.np = _need("numpy", "install it with `pip install numpy`.")
        torch = _need("torch", TORCH_HINT)      # quoridor.net imports it
        self.fr = _need("quoridor.fastrules", ENGINE_HINT)
        mcts = _need("quoridor.mcts", ENGINE_HINT)
        net = _need("quoridor.net", ENGINE_HINT)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.think = think
        self.max_sims = max_sims
        self.fr.warmup()

        try:
            model, self.meta = net.load_checkpoint(str(path), device)
        except Exception as exc:                     # noqa: BLE001 - user input
            raise SystemExit(f"could not load {path}: {exc}")
        evaluator = net.NetEvaluator(model, device=device,
                                     graph_batches=(self.LEAF,))
        self.mcts = mcts.BatchedMCTS(evaluator, n_games=1,
                                     max_nodes=max_sims + self.LEAF + 256,
                                     leaf_batch=self.LEAF)
        self.scratch = self.fr.make_scratch()
        self.rate = self._calibrate()

    def _calibrate(self) -> float:
        """Measure simulations per second, while no clock is running.

        The first search also compiles the jitted kernels and captures the CUDA
        graph, so it is thrown away and a second one is timed.
        """
        start = self.fr.initial_state().reshape(1, -1)
        self.mcts.search(start.copy(), 512)
        probe = 2048
        began = time.perf_counter()
        self.mcts.search(start.copy(), probe)
        elapsed = time.perf_counter() - began
        return max(self.MIN_RATE, probe / elapsed if elapsed > 0 else 0.0)

    # -- the callback the runner wants --------------------------------------

    def choose(self, state: rules.State, seat: int, clock) -> str | None:
        encoded = self._encode(state)
        seconds = clock.budget(target=self.think)
        sims = int(max(256, min(self.max_sims, seconds * self.rate)))

        began = time.perf_counter()
        visits = self.mcts.search(encoded.reshape(1, -1).copy(), sims)[0]
        elapsed = time.perf_counter() - began
        if elapsed > 0.05:
            # Track observed throughput so the next move lands on its budget.
            self.rate = 0.7 * self.rate + 0.3 * (sims / elapsed)

        if visits.sum() <= 0:
            # A search from a decided position visits nothing, and argmax of
            # all zeros is action 0 -- a wall. Say "no move" instead.
            return None
        action = int(visits.argmax())
        value = float(self.mcts.root_value()[0])
        log.info("%d sims in %.1fs, eval %+.2f", sims, elapsed, value)
        return self._token(state, encoded, action)

    # -- engine <-> rules ---------------------------------------------------

    def _encode(self, state: rules.State):
        """A :class:`murus_bot.rules.State` as the engine's state array."""
        fr = self.fr
        encoded = self.np.zeros(fr.STATE_SIZE, dtype=self.np.uint8)
        for slot in state.walls_h:
            encoded[fr.WH_OFF + slot] = 1
        for slot in state.walls_v:
            encoded[fr.WV_OFF + slot] = 1
        encoded[fr.IDX_P0], encoded[fr.IDX_P1] = state.pawns
        encoded[fr.IDX_WL0], encoded[fr.IDX_WL1] = state.walls_left
        encoded[fr.IDX_TURN] = state.turn
        return encoded

    def _token(self, state: rules.State, encoded, action: int) -> str:
        """An engine action as an API move token."""
        fr = self.fr
        if action < fr.MOVE_BASE:
            orientation, slot = divmod(action, fr.NUM_WALL_SLOTS)
            return rules.wall_token(orientation, *divmod(slot, 8))
        # Pawn tokens name the destination square, and a jump's destination
        # depends on the position -- so play the action and see where it lands.
        after = encoded.copy()
        fr.apply_action(after, action, self.scratch)
        landed = after[fr.IDX_P0 if state.turn == 0 else fr.IDX_P1]
        return rules.square_token(int(landed))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", default="https://murus.net",
                        help="server base URL (default %(default)s)")
    parser.add_argument("--token", default=os.environ.get("MURUS_TOKEN"),
                        help="API token with the play scope "
                             "(default: $MURUS_TOKEN)")
    parser.add_argument("--checkpoint", required=True,
                        help="torch checkpoint for the network (not supplied "
                             "with this repository)")
    parser.add_argument("--think", type=float, default=3.0,
                        help="target seconds per move (default %(default)s)")
    parser.add_argument("--max-sims", type=int, default=200_000,
                        help="ceiling on simulations per move; sizes the "
                             "preallocated search tree (default %(default)s)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seek", action="append", default=[], metavar="300+3",
                        help="keep a seek with this time control open while "
                             "idle; suffix :casual for unrated; repeat to "
                             "rotate through several")
    parser.add_argument("--accept", default="all",
                        choices=["all", "rated", "casual", "none"],
                        help="which challenges to accept (default %(default)s)")
    parser.add_argument("--games", type=int, default=None,
                        help="stop after this many games (default: never)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    # httpx logs a line per request, which at a move a second is just noise.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not args.token:
        parser.error("no API token: pass --token or set MURUS_TOKEN")

    log.info("loading engine...")
    bot = AlphaZeroBot(args.checkpoint, device=args.device, think=args.think,
                       max_sims=args.max_sims)
    log.info("engine ready on %s: %s, %.0f sims/s", bot.device,
             bot.meta.get("checkpoint", args.checkpoint), bot.rate)

    with MurusClient(args.token, server=args.server) as client:
        runner = BotRunner(client, bot.choose, accept=args.accept,
                           seeks=[parse_seek(s) for s in args.seek],
                           max_games=args.games)
        try:
            runner.run()
        except KeyboardInterrupt:
            log.info("stopping")
        except MurusError as exc:
            log.error("%s", exc)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
