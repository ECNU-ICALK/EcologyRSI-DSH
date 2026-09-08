"""Versioned native measurement context; historical digests are never rewritten."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from ..core.identity import FrozenObject, content_id


_CASE_FIELDS = ('target', 'horizon_hours', 'origin_timestamp', 'timestamp')


def case_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    key = tuple(row.get(name) for name in _CASE_FIELDS)
    if type(key[0]) is not str or not key[0] or any(type(key[i]) is not int for i in (1, 2, 3)) or key[1] < 1 or key[3] - key[2] != key[1]:
        raise ValueError('invalid_prediction_case_identity')
    return key


@dataclass(frozen=True)
class EvaluationContext:
    """No observations or predictions: identity and measurement rules only."""
    contract: FrozenObject
    cases: tuple[tuple[str, int, int, int], ...]

    def __post_init__(self):
        if not isinstance(self.contract, FrozenObject) or type(self.cases) is not tuple:
            raise ValueError('frozen_context_required')
        if len(self.cases) != len(set(self.cases)):
            raise ValueError('duplicate_prediction_case')
        for case in self.cases:
            case_key(dict(zip(_CASE_FIELDS, case, strict=True)))

    @property
    def context_id(self) -> str:
        return content_id('ecologyrsi/native-evaluation-context@1', self.to_dict())

    def to_dict(self) -> dict:
        return {**self.contract.to_dict(), 'case_grid': [list(case) for case in self.cases]}

    def summary(self) -> dict:
        return {**self.contract.to_dict(), 'context_id': self.context_id,
                'case_grid_digest': content_id('ecologyrsi/native-case-grid@1', self.cases),
                'expected_prediction_cells': len(self.cases),
                'expected_origins': len({case[2] for case in self.cases})}

    def validate_rows(self, rows: Sequence[Mapping[str, Any]], *, allow_missing: bool = False) -> int:
        actual = tuple(case_key(row) for row in rows)
        if len(actual) != len(set(actual)):
            raise ValueError('duplicate_prediction_case')
        expected = set(self.cases)
        if not set(actual).issubset(expected):
            raise ValueError('prediction_case_outside_frozen_context')
        missing = len(expected - set(actual))
        if missing and not allow_missing:
            raise ValueError('incomplete_prediction_case_grid')
        return missing

    @classmethod
    def native(cls, series, rows, *, units: Mapping[str, str], baseline_profile: Mapping,
               scales: Mapping[str, float], scoring_contract: Mapping):
        contract = {
            'schema_version': 'ecologyrsi-dsh.native-evaluation-context/1',
            'dataset_digest': series.digest,
            'split_manifest_digest': series.split_manifest_digest_sha256,
            'episode_id': series.episode_id,
            'partitions': {name: {'start': series.partitions[name].start, 'end': series.partitions[name].end}
                           for name in ('training_fit', 'training_feedback')},
            'target_units': dict(units),
            'baseline_profile_digest': baseline_profile['digest'],
            'scales': dict(scales),
            'metric': 'native_bounded_rmse_skill',
            'higher_is_better': True,
            'scoring_contract': dict(scoring_contract),
            'missing_policy': 'native_fixed_grid_coverage_and_missing_task_penalty',
            'time_policy': 'exact_observed_timestamps_no_gap_compression',
            'time_unit': 'integer_hours_since_epoch',
            'evidence_class': 'exploratory',
        }
        return cls(FrozenObject.from_mapping(contract), tuple(sorted(case_key(row) for row in rows)))
