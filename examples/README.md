# Examples

`minimal_run.py` exercises the public Python API with a deterministic fake DSH
adapter and toy evaluator. It writes one SQLite ledger, closes it, then opens a
new director and verifies replay from the persisted event stream.

Run from a source checkout:

```bash
source activate py310
PYTHONPATH=src python examples/minimal_run.py --db /tmp/ecologyrsi-example.sqlite3
```

After wheel installation, omit `PYTHONPATH`:

```bash
ecologyrsi-dsh demo --db /tmp/ecologyrsi-demo.sqlite3 --run-id run:demo
```

`task-manifest.json` documents the bounded input used by the example. The
dataset identifier and digest bind the deterministic seed-0 engineering
fixture; the task seed controls candidate search only. It is only a visible/validation demo, not a real dataset snapshot,
a scientific crop-model result, or evidence for a causal claim.

`local-config.json` is an optional relative-path configuration for the local
server:

```bash
source activate py310
PYTHONPATH=src python -m ecologyrsi_dsh serve --config examples/local-config.json
```
