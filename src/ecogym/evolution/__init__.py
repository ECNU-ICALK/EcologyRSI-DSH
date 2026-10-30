"""Harness evolution: the outer loop over agent episodes.

``Evolution`` runs rounds of review -> candidate harnesses -> paired
multi-task comparison -> adoption on a suite's training tasks
(``loop``); an evolver session is a DSH agent that reviews the evidence and
writes candidates (``evolver``, ``evidence``); ``verify`` runs the one-time
checks of the result on holdout tasks and sealed periods; ``fold`` turns
the ledger events into a readable record.
"""

from .fold import EVENT_KINDS, fold, summary
from .loop import Evolution, EvolveConfig, RunStopped
from .verify import verify_holdout, verify_sealed

__all__ = [
           "EVENT_KINDS",
           "Evolution",
           "EvolveConfig",
           "RunStopped",
           "fold",
           "summary",
           "verify_holdout",
           "verify_sealed",
]
