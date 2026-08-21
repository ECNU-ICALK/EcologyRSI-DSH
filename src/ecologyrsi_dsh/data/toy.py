"""Deterministic crop-soil-water toy domain for the first runnable loop.

This is an engineering fixture, not a scientific crop model.  It exists to
prove that a TaskManifest, structured DSH proposals, time-forward evaluation,
promotion, and replay can work end to end without external data services.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from ..core.models import Candidate, Evaluation, Proposal, digest


_TOY_FLOAT_DECIMALS = 12


@dataclass(frozen=True, slots=True)
class Observation:
    day: int
    rainfall: float
    evapotranspiration: float
    soil_water: float


class ToyCropSoilWater:
    """Small deterministic time series with train/validation/test splits."""

    def __init__(self, *, seed: int = 0, days: int = 60) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if isinstance(days, bool) or not isinstance(days, int) or days < 15:
            raise ValueError("days must be an integer >= 15")
        self.seed = seed
        self.days = days
        self.observations = self._make_observations()
        self.dataset_digest = digest(
            {
                "schema": "toy-crop-soil-water/v1",
                "seed": self.seed,
                "days": self.days,
                "units": {
                    "time": "day",
                    "rainfall": "normalized water depth/day",
                    "evapotranspiration": "normalized water depth/day",
                    "soil_water": "normalized fraction [0,1]",
                },
                "observations": [
                    {
                        "day": item.day,
                        "rainfall": item.rainfall,
                        "evapotranspiration": item.evapotranspiration,
                        "soil_water": item.soil_water,
                    }
                    for item in self.observations
                ],
            }
        )

    def _make_observations(self) -> tuple[Observation, ...]:
        # The seed shifts the phase, but no random state or external file is
        # involved, making a replay byte-for-byte reproducible.
        phase = (self.seed % 17) / 17.0
        result: list[Observation] = []
        water = 0.52
        for day in range(self.days):
            # libm may differ by one ULP across CPU architectures. Quantize
            # every state transition so the fixed engineering fixture and its
            # digest are identical on ARM and x86 hosts.
            rainfall = round(
                max(
                    0.0,
                    0.07 * math.sin((day + self.seed) / 2.7)
                    + (0.08 if day % 9 == 0 else 0.0),
                ),
                _TOY_FLOAT_DECIMALS,
            )
            evap = round(
                0.045 + 0.012 * (1.0 + math.sin(day / 5.0 + phase)),
                _TOY_FLOAT_DECIMALS,
            )
            water = round(
                min(0.95, max(0.05, water + rainfall - evap)),
                _TOY_FLOAT_DECIMALS,
            )
            result.append(Observation(day, rainfall, evap, water))
        return tuple(result)

    @property
    def splits(self) -> dict[str, tuple[Observation, ...]]:
        train_end = int(self.days * 0.6)
        training_fit_end = int(train_end * 0.5)
        development_end = int(self.days * 0.8)
        return {
            # Compatibility names retained for the engineering fixture.  They
            # map to the registry's browser-visible fit/feedback windows and
            # its embargoed gate tail; restricted development rows are never
            # read by the iterative evaluator.
            "train": self.observations[:training_fit_end],
            "validation": self.observations[training_fit_end + 1 : train_end],
            "test": self.observations[development_end + 1 :],
        }

    @staticmethod
    def _parameters(parameters: Mapping[str, Any]) -> tuple[float, int, float]:
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        alpha = parameters.get("alpha", 0.35)
        window = parameters.get("window", 5)
        threshold = parameters.get("water_threshold", 0.4)
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= 30:
            raise ValueError("window must be an integer between 1 and 30")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("water_threshold must be between 0 and 1")
        return float(alpha), window, float(threshold)

    def _predictions(self, parameters: Mapping[str, Any]) -> tuple[float, ...]:
        alpha, window, threshold = self._parameters(parameters)
        predictions = [self.observations[0].soil_water]
        for index in range(1, self.days):
            start = max(0, index - window)
            history = self.observations[start:index]
            rolling = sum(item.soil_water for item in history) / len(history)
            previous = self.observations[index - 1].soil_water
            rainfall = self.observations[index].rainfall
            predicted = alpha * previous + (1.0 - alpha) * rolling + rainfall
            # A threshold is a conservative intervention rule in the toy
            # domain: add a small irrigation correction when water is low.
            if predicted < threshold:
                predicted += 0.02
            predictions.append(min(1.0, max(0.0, predicted)))
        return tuple(predictions)

    def score(self, parameters: Mapping[str, Any], split: str = "validation") -> dict[str, float]:
        if split not in self.splits:
            raise ValueError("split must be train, validation, or test")
        selected = self.splits[split]
        if not selected:
            raise ValueError("requested split is empty")
        by_day = {item.day: item for item in selected}
        predictions = self._predictions(parameters)
        errors = [predictions[day] - item.soil_water for day, item in by_day.items()]
        mae = sum(abs(error) for error in errors) / len(errors)
        rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
        balance_errors = []
        for day in by_day:
            if day == 0:
                continue
            previous = predictions[day - 1]
            current = predictions[day]
            observation = self.observations[day]
            balance_errors.append(abs(current - previous - observation.rainfall + observation.evapotranspiration))
        water_balance_error = sum(balance_errors) / len(balance_errors) if balance_errors else 0.0
        non_negative_state = all(0.0 <= value <= 1.0 for value in predictions)
        constraint_violations = float(0 if non_negative_state else 1)
        return {
            "score": max(0.0, 1.0 - rmse),
            "mae": mae,
            "rmse": rmse,
            "n": float(len(errors)),
            "water_balance_error": water_balance_error,
            "non_negative_state": 1.0 if non_negative_state else 0.0,
            "constraint_violations": constraint_violations,
        }

    def evaluate_candidate(
        self,
        run_id: str,
        candidate: Candidate,
        proposal: Proposal,
        *,
        split: str = "validation",
        evaluator_digest: str = "toy-crop-soil-water@1",
    ) -> Evaluation:
        if candidate.run_id != run_id or proposal.run_id != run_id:
            raise ValueError("candidate and proposal must belong to run_id")
        metrics = self.score(proposal.changes, split)
        predictions = self._predictions(proposal.changes)
        selected = self.splits[split]
        # Keep a small, deterministic sample-level trace for the browser
        # projection.  It is an operational prediction audit, not private
        # model reasoning and never includes the full source series.
        preview = [
            {
                "timestamp": item.day,
                "origin_timestamp": item.day - 1 if item.day > 0 else None,
                "target_timestamp": item.day,
                "horizon_hours": 24,
                "target": "soil_water",
                "unit": "fraction",
                "observed": item.soil_water,
                "predicted": predictions[item.day],
                "baseline": self.observations[item.day - 1].soil_water
                if item.day > 0
                else item.soil_water,
            }
            for item in selected[:48]
        ]
        # The threshold is intentionally modest; it makes the fixture useful
        # for both accepted and rejected candidates while remaining stable.
        passed = (
            metrics["rmse"] <= 0.12
            and metrics["water_balance_error"] <= 0.25
            and metrics["non_negative_state"] == 1.0
            and metrics["constraint_violations"] == 0.0
        )
        return Evaluation(
            evaluation_id=f"evaluation:{candidate.candidate_id}:{split}",
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            score=metrics["score"],
            passed=passed,
            metrics={
                **metrics,
                "prediction_preview": preview,
                "dataset_digest": self.dataset_digest,
                "evaluation_scope": f"visible/{split}/demo",
                "causal_interpretation": False,
            },
            partition=split,
            evaluator_digest=evaluator_digest,
        )
