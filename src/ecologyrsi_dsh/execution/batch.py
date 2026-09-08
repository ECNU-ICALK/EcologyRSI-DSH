"""Local numerical execution for the native greenhouse measurement contract.

The domain adapter retains timestamps, gaps, partitions and causal feature
rules. This backend produces fitted models and predictions, never scores or
promotion decisions. DSH requests continue to use their frozen agent mode.
"""
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from ..data.contracts import DatasetSeries
from ..evaluators.greenhouse_prediction import RidgeConfig, fit_predict_exogenous_ridge
from ..evaluators.sample_execution import SampleExecutionCancelledError, SampleExecutionPausedError


class NativeBatchBackend:
    version = 'ecologyrsi-dsh.native-batch/1'

    def run(self, series: DatasetSeries, *, targets: Sequence[str], horizons: Sequence[int],
            config: RidgeConfig | Mapping, evaluation_history_steps: int | None = None,
            defer_prediction_partitions: Sequence[str] = (),
            on_fit_complete: Callable[[], None] | None = None,
            control: Callable[[], str] | None = None) -> dict:
        def check():
            if control is None:
                return
            status = control()
            if status == 'paused':
                raise SampleExecutionPausedError('run paused during numerical execution')
            if status != 'running':
                raise SampleExecutionCancelledError('run closed during numerical execution')
        def fitted():
            check()
            if on_fit_complete is not None:
                on_fit_complete()
        check()
        result = fit_predict_exogenous_ridge(series, targets=targets, horizons=horizons,
            config=config, evaluation_history_steps=evaluation_history_steps,
            defer_prediction_partitions=defer_prediction_partitions, on_fit_complete=fitted, check_control=check)
        check()
        return result
