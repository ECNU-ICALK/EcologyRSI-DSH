"""Value-blind origin cursor with target-maturity purging."""
import math
from bisect import bisect_left
from .epoch_cohorts import CohortCapacityError


class _Cursor:
    def __init__(self, eligible):
        self.eligible = eligible
        self.timestamps = tuple(origin.origin_timestamp for origin in eligible)
        self.index = 0
        self.purged = 0

    def take(self, count, *, cross_day=False, span_hours=0):
        start = self.index
        if start >= len(self.eligible):
            raise CohortCapacityError(required=start + count,
                                      available=len(self.eligible), max_generations=0)
        # Small adaptation batches still span at least a complete daily cycle.
        stride = max(1, math.ceil(24 / (count - 1))) if cross_day else 1
        indices = range(start, start + count * stride, stride)
        if span_hours:
            # Spread a fixed number of requests over time, using timestamps
            # alone. More temporal coverage must not inflate sample counts.
            end = max(start + count - 1, bisect_left(
                self.timestamps, self.timestamps[start] + span_hours, lo=start))
            indices = tuple(start + i * (end - start) // (count - 1) for i in range(count))
        if indices[-1] >= len(self.eligible):
            raise CohortCapacityError(required=indices[-1] + 1,
                                      available=len(self.eligible), max_generations=0)
        result = tuple(self.eligible[i] for i in indices)
        self.index = indices[-1] + 1
        while (self.index < len(self.eligible) and
               self.eligible[self.index].origin_timestamp <= result[-1].maximum_target_timestamp):
            self.index += 1
            self.purged += 1
        return result
