"""CLI for the external AI-for-AI evolution lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import CallableBackend
from .capabilities import CapabilityRegistry, CapabilitySpec
from .controller import EvolutionController
from .evaluator import EvaluationReport
from .genome import Mutation, PluginGenome
from .store import EvolutionStore


def _registry() -> CapabilityRegistry:
    return CapabilityRegistry([
        CapabilitySpec("skill", "planner-balanced@1", ("sample-planner",), ("greenhouse",)),
        CapabilitySpec("skill", "planner-horizon-aware@1", ("sample-planner",), ("greenhouse",)),
        CapabilitySpec("tool", "prediction@1", ("sample-planner",), ("greenhouse",)),
        CapabilitySpec("tool", "repair@1", ("sample-planner",), ("greenhouse",)),
        CapabilitySpec("workflow", "sample-wave@1", ("sample-planner",), ("greenhouse",)),
        CapabilitySpec("algorithm", "greenhouse-ridge@1", (), ("greenhouse",)),
    ])


def _demo_backend(genome: PluginGenome, cohort_id: str) -> EvaluationReport:
    skill = genome.skills["sample-planner"]["capability_id"]
    tool_count = len(genome.tools["sample-planner"])
    score = 0.20 + (0.015 if skill == "planner-horizon-aware@1" else 0.0) + (0.002 if tool_count > 1 else 0.0)
    return EvaluationReport(
        cohort_id=cohort_id,
        scientific_score=score,
        per_cell_scores={"air_temperature@1h": score, "co2_concentration@24h": score},
        physical_violations=0,
        skill_success_rate=1.0,
        tool_success_rate=1.0,
        latency_ms=100.0,
        cost_units=1.0,
        stable=True,
        sample_count=64,
    )


def run_demo(args: argparse.Namespace) -> int:
    registry = _registry()
    store = EvolutionStore(args.db)
    controller = EvolutionController(store, registry)
    parent = PluginGenome(
        domain="greenhouse",
        skills={"sample-planner": {"capability_id": "planner-balanced@1", "parameters": {}}},
        tools={"sample-planner": ("prediction@1",)},
        workflow={"capability_id": "sample-wave@1", "parameters": {}},
        algorithm={"capability_id": "greenhouse-ridge@1", "parameters": {}},
    )
    if store.incumbent_digest() is None:
        controller.seed(parent)
    decision = controller.evaluate_and_record(
        parent.digest,
        Mutation("skill", "sample-planner", {"capability_id": "planner-horizon-aware@1", "parameters": {}}),
        args.cohort,
        CallableBackend(_demo_backend).evaluate,
    )
    candidate = controller.latest_candidate_digest()
    if decision.status == "certification_eligible":
        controller.promote(candidate)
    print(json.dumps({"candidate_digest": candidate, "decision": decision.to_dict(), "incumbent_digest": controller.current_incumbent().digest}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="External AI-for-AI plugin evolution lab")
    parser.add_argument("--db", default="evolution_lab.sqlite", type=Path)
    parser.add_argument("--cohort", default="demo-cohort@1")
    parser.add_argument("--demo", action="store_true", help="run a deterministic local experiment")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.demo:
        raise SystemExit("请使用 --demo，或由宿主集成 DshExperimentAdapter")
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
