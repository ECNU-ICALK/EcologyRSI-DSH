"""Sealed evaluation: each method sees validation and final_test once.

``seal`` records the exposure in the ledger *before* opening the stage, so a
crash after opening still counts as an exposure. The key is the task's data
(frame digest and the stage interval) and the genome digest; it ignores gate
settings, so re-scoring the same method on the same sealed rows under a
tweaked task card is refused too. final_test requires a prior validation
exposure of the same method.

Exposures are per ledger file; the CLI uses one default ledger
(``$ECOGYM_HOME/ledger.sqlite3``) so they are shared across runs.
"""

from __future__ import annotations

from typing import Any

from ..evaluation.gates import objective_interval
from ..evaluation.harness import Bench
from ..kernel.canonical import digest
from ..kernel.ledger import Ledger
from ..protocol.partitions import SEALED_STAGES
from ..protocol.views import SealedAccess
from ..solvers.base import Genome


def exposure_key(bench: Bench, genome: Genome, stage: str) -> str:
    return digest({
        "schema": "ecogym.exposure/1",
        "frame": bench.frame.digest,
        "stage": stage,
        "interval": bench.partitions.intervals[stage],
        "genome": genome.digest,
    })


def seal(bench: Bench, ledger: Ledger, genome: Genome, stage: str, *, run_id: str, label: str = "") -> dict[str, Any]:
    if stage not in SEALED_STAGES:
        raise ValueError(f"{stage} is not a sealed stage")
    if stage == "final_test":
        prior = exposure_key(bench, genome, "validation")
        if not any(item["key"] == prior for item in ledger.exposures(method=genome.digest)):
            raise PermissionError("evaluate this method on validation before final_test")
    key = exposure_key(bench, genome, stage)
    if not ledger.claim_exposure(key, task=bench.task.id, stage=stage, method=genome.digest, run_id=run_id):
        raise PermissionError(f"method {genome.digest[:12]} has already been evaluated on {stage}")
    evaluation = bench.evaluate(genome, stage, sealed=SealedAccess(stage, key))
    low, high = objective_interval(evaluation, None, bench.task.gates, {"sealed": key})
    others = [item for item in ledger.exposures(task=bench.task.id) if item["stage"] == stage]
    report = {
        "stage": stage,
        "label": label,
        "genome": genome.to_dict(),
        "digest": genome.digest,
        "exposure_key": key,
        "origins": evaluation.stage.size,
        "score": evaluation.score(),
        "interval": [low, high],
        "cells": evaluation.cells(),
        "methods_exposed_on_stage": len(others),
    }
    ledger.append(run_id, "sealed_evaluated", report)
    return report


__all__ = ["exposure_key", "seal"]
