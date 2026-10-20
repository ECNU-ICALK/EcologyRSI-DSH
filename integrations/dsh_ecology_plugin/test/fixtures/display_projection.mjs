export function sampleContextFixture() {
  return {
    schema_version: "ecologyrsi-dsh.sample-routing-wave/1", role: "planner",
    model_route_id: "local-display/model", wave_digest: "f".repeat(64),
    samples: Array.from({ length: 9 }, (_, index) => ({
      sample_id: `origin-fixture-cell-${index}`, candidate_id: "candidate:" + "a".repeat(40),
      dataset_digest: "d".repeat(64), algorithm_id: "greenhouse-baseline-aligned-ridge@2",
      algorithm_version: "2", origin_timestamp: "2026-01-01T00:00:00Z",
      partition: "train_feedback", attempt: 1, failure_feedback: [],
      target: ["temperature", "relative_humidity", "co2"][Math.floor(index / 3)],
      horizon_hours: [1, 6, 24][index % 3], baseline: 10 + index, minimum: 0, maximum: 2000,
      context_ref: "c".repeat(64),
    })),
    available_tools: [{ tool_id: "candidate-model", version: "1", parameters: {}, purpose: "Synthetic candidate model" }],
    allowed_reason_codes: ["model_evidence_consistent"],
    context: { candidate_agent_profile: { schema_version: "ecologyrsi-dsh.candidate-agent-profile/1",
      role: "sample-planner", skill_name: "origin-vector-forecasting", instruction_directive: "Use the available evidence." },
      origin_contexts: { ["c".repeat(64)]: { history_window: [10, 11, 12] } },
      prediction_contract: { max_prediction_tool_calls: 4 } },
  };
}

export function predictionResultFixture(context = sampleContextFixture()) {
  return {
    accepted: true, event_id: "fixture-event", output_digest: "e".repeat(64),
    tool_id: "candidate-model", call_id: "candidate-evidence", wave_digest: context.wave_digest,
    status: "completed", remaining_calls: 3, elapsed_ms: 1.25,
    outputs: context.samples.map((sample, index) => ({ sample_id: sample.sample_id, predicted: 10 + index,
      metadata: { fit_digest: "b".repeat(64), parameters: {
        ridge_alpha: 1, feature_recipe: { window: 12,
          features: Array.from({ length: 10 }, (_, feature) => ({ name: `causal_feature_${feature}`, coefficient: feature / 10 })) },
      } },
    })),
  };
}
