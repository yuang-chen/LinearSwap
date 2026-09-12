"""The three-stage workflow — verify → posttrain → evaluate — plus ``run`` chaining them.

Every stage module exposes ``add_args(parser)`` and ``main(args)`` and is wired
into the ``linswap.py`` command-line entry point at the repository root.
"""

from . import evaluate, posttrain, run, verify  # noqa: F401

STAGES = {
    "verify": verify,
    "posttrain": posttrain,
    "evaluate": evaluate,
    "run": run,
}
