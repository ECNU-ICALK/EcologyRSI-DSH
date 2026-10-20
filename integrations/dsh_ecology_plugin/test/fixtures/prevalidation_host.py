"""Isolated real HTTP Host for the rc.2 loopback regression; synthetic data only."""
from contextlib import ExitStack
from copy import deepcopy
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
import threading

from ecologyrsi_dsh.api.handler import EvolutionHTTPServer
from ecologyrsi_dsh.integrations.dsh_native_runtime import DshNativeAgentRuntimeClient

os.environ['ECOLOGYRSI_SIDECAR_TOOL_TOKEN'] = 'local-only-prevalidation'
with tempfile.TemporaryDirectory(prefix='ecology-prevalidation-host-') as directory, ExitStack() as stack:
    server = EvolutionHTTPServer(('127.0.0.1', 0), Path(directory) / 'events.sqlite3')
    service = server.dsh_tools
    cases = {}
    bindings = {}
    executions = Counter()
    validation_observations = []
    for name in ('S-nine-blend-single-ref-corrected', 'two-invalid-no-third', 'persist-revoked', 'T-blend-repair-extra-tool', 'blend-only-repair-extra-tool', 'two-blend-invalid-with-tool', 'ordinary-schema-extra-tool-denied'):
        run_id = f'local-{name}'
        service.ledger.append(run_id, 'RunCreated', {'synthetic_fixture': True})
        fence = service.open_admission(run_id, 1, 1, role='sample-planner', stage='sample.plan', idempotency_key=f'idempotency-{name}')
        sample_ids = [f'origin-fixture-cell-{index}' for index in range(9)]
        def executor(tool_id, _parameters, name=name, sample_ids=sample_ids):
            executions[name] += 1
            return {sample_id: {'predicted': 10 + index} for index, sample_id in enumerate(sample_ids)}
        binding = stack.enter_context(service.bind_prediction_tool(run_id=run_id, stage_attempt=1,
            idempotency_key=f'idempotency-{name}', wave_digest='f' * 64, sample_ids=sample_ids,
            catalog=[{'tool_id': 'ridge@1', 'version': '1'}], executor=executor,
            prediction_tool_call_budget=4, allowed_methods=['blend'] if name == 'blend-only-repair-extra-tool' else ['model', 'blend', 'adjusted', 'direct'],
            identity_digests={'genome_digest':'a'*64,'compiled_behavior_digest':'b'*64,'phenotype_instance_digest':'c'*64}))
        bindings[run_id] = binding
        cases[name] = {'run_id': run_id, 'admission_id': fence.admission_id, 'idempotency_key': f'idempotency-{name}'}
    validate = service.validate_structured
    def observed_validate(envelope):
        binding = bindings[envelope['identity']['run_id']]
        before = (service.ledger.latest_seq(), deepcopy(binding.calls), deepcopy(binding._predictions),
                  deepcopy(binding._session_calls), binding._spent(), deepcopy(executions))
        result = validate(envelope)
        after = (service.ledger.latest_seq(), deepcopy(binding.calls), deepcopy(binding._predictions),
                 deepcopy(binding._session_calls), binding._spent(), deepcopy(executions))
        assert before == after, 'Dry-run changed ledger, prediction cache, budget or execution'
        validation_observations.append({'run_id': envelope['identity']['run_id'], 'valid': result['valid'],
            'diagnostic_count': len(result['diagnostics']), 'zero_state_mutation': True})
        return result
    service.validate_structured = observed_validate
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(json.dumps({'port': server.server_address[1], 'cases': cases}), flush=True)
    try:
        config = json.loads(sys.stdin.readline())
        server.dsh_native_runtime = DshNativeAgentRuntimeClient(config['native_url'], token='local-only-native-attestation')
        assert sys.stdin.readline().strip() == 'finish'
        report = {'host': 'production EvolutionHTTPServer + DshToolService', 'validations': validation_observations, 'cases': {}}
        for name, meta in cases.items():
            events = service.ledger.events(meta['run_id'])
            kinds = Counter(event.kind for event in events)
            report['cases'][name] = {'events': dict(kinds), 'tool_executions': executions[name], 'tool_budget_spent': bindings[meta['run_id']]._spent()}
        print(json.dumps(report), flush=True)
    finally:
        server.shutdown()
        thread.join(3)
        server.close()
