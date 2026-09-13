"""The workflow — verify → (distill) → posttrain → evaluate — plus ``run`` chaining it.

Every stage module exposes ``add_args(parser)`` and ``main(args)`` and is wired
into the ``linswap.py`` command-line entry point at the repository root.
"""

from . import distill, evaluate, export, posttrain, run, verify  # noqa: F401

STAGES = {
    "verify": verify,
    "distill": distill,
    "posttrain": posttrain,
    "evaluate": evaluate,
    "run": run,
    "export": export,
}
