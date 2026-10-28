"""ecogym: a reproducible training ground for ecological forecasting methods.

Layers, lowest first (a module may import only from layers below it):

1. ``kernel``     canonical JSON, digests, strict config parsing, SQLite ledger
2. ``data``       dataset specs, loaders, the dense time-grid ``Frame``
3. ``task``       task cards: targets, horizons, information set, gates
4. ``protocol``   partitions, causal views, origins, baselines, cohorts
5. ``solvers``    solver registry, ridge, feature recipes
6. ``evaluation`` scoring, block bootstrap, promotion gates, the bench
7. ``arena``      genomes, proposers, the generation loop, sealed evaluation
8. ``cli``        command line entry point

Domain packs live in ``ecogym.domains`` and register datasets, tasks and
solvers; framework code never names a dataset, column or target.
"""

__version__ = "0.1.0"
