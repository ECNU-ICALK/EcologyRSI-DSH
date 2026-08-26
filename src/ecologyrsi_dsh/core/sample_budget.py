"""Conversions between scoring cells and complete forecast origins."""

from __future__ import annotations


def scoring_cell_budget(origin_count: int, cells_per_origin: int) -> int:
    if isinstance(origin_count, bool) or not isinstance(origin_count, int) or origin_count < 1:
        raise ValueError("origin_count must be a positive integer")
    if (
        isinstance(cells_per_origin, bool)
        or not isinstance(cells_per_origin, int)
        or cells_per_origin < 1
    ):
        raise ValueError("cells_per_origin must be a positive integer")
    return origin_count * cells_per_origin


def complete_origin_count(scoring_cells: int, cells_per_origin: int) -> int:
    if isinstance(scoring_cells, bool) or not isinstance(scoring_cells, int) or scoring_cells < 1:
        raise ValueError("scoring_cells must be a positive integer")
    if (
        isinstance(cells_per_origin, bool)
        or not isinstance(cells_per_origin, int)
        or cells_per_origin < 1
    ):
        raise ValueError("cells_per_origin must be a positive integer")
    origin_count, remainder = divmod(scoring_cells, cells_per_origin)
    if remainder:
        nearest = origin_count * cells_per_origin
        raise ValueError(
            f"scoring-cell budget {scoring_cells:,} must be divisible by "
            f"{cells_per_origin}; nearest lower complete budget is {nearest:,}"
        )
    return origin_count
