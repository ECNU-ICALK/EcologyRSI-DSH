"""ecogym: an ecological forecasting training ground for research agents and their harnesses.

Layers, lowest first (a module may import only from its own layer or below):

1. ``kernel``     canonical JSON, digests, strict config parsing, SQLite ledger, secrets
2. ``data``       dataset specs, loaders, the dense time-grid ``Frame``
3. ``task``       task cards and suites: targets, horizons, information set, gates
4. ``protocol``   partitions, causal views, origins, baselines, cohorts
5. ``solvers``    solver registry, ridge, feature recipes
6. ``evaluation`` scoring, block bootstrap, promotion gates, multi-task comparison, the bench
7. ``harness``    content-addressed agent harnesses (instructions, skills, code, memory, DSH options)
8. ``episodes``   the inner loop: one sandboxed DSH agent episode, the judge, trajectories, export
9. ``evolution``  the outer loop: evolver sessions, harness comparison and adoption, verification
10. ``arena``, ``agents``  parameter-level method search with host-computed forecasts
11. ``cli``, ``service``, ``web``  command line, JSON API and workbench

Domain packs live in ``ecogym.domains`` and register datasets, tasks, suites
and solvers; framework code never names a dataset, column or target.
"""

__version__ = "0.3.2"
