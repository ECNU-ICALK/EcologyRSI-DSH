import copy
import json
import unittest

from ecologyrsi_dsh.evaluators.origin_prompt import compact_origin_contexts


class OriginPromptTests(unittest.TestCase):
    def test_shared_history_preserves_every_cell_and_does_not_mutate_inputs(self):
        history = [{'time': t, 'value': 20.125 + t / 1000} for t in range(48)]
        contexts = {f'cell-{i}': {'history': history, 'horizon': i,
                                'causal_provenance': {'cutoff': 47, 'latest': 47}}
                    for i in range(9)}
        before = copy.deepcopy(contexts)
        compact, shared = compact_origin_contexts(contexts)

        def expand(value):
            if isinstance(value, dict):
                if set(value) == {'shared_origin_ref'}:
                    return shared[value['shared_origin_ref']]
                return {key: expand(child) for key, child in value.items()}
            if isinstance(value, list):
                return [expand(child) for child in value]
            return value

        self.assertEqual(expand(compact), before)
        self.assertEqual(contexts, before)
        self.assertLess(len(json.dumps([compact, shared])), len(json.dumps(contexts)) / 2)

    def test_unique_contexts_are_left_intact(self):
        contexts = {'a': {'history': [1, 2]}, 'b': {'history': [3, 4]}}
        self.assertEqual(compact_origin_contexts(contexts), (contexts, {}))
