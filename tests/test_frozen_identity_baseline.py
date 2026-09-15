"""Frozen identity baseline: every digest an archived run replays against.

These literals were captured before the multi-domain compatibility refactor.
Layer B of replay (resume, re-score, promotion) re-derives evaluator, predictor
and dataset identity from *current* code and fails closed when it differs from
the artifact, so any of these values moving means every archived greenhouse run
stops being resumable or re-scorable. Deriving a value from a new source is
allowed; producing a different value is not.

A deliberate identity change is a new id or a new schema_version alongside the
old one, never an edit in place -- so a failure here should be fixed by adding
an entry, not by refreshing the literal.
"""
import inspect
import re
import unittest

from ecologyrsi_dsh.data.adapters import dataset_adapter
from ecologyrsi_dsh.data.registry import DatasetRegistry
from ecologyrsi_dsh.evaluators.registry import EvaluatorRegistry
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry

# dataset_id -> (contract_digest, definition_digest)
DATASET_IDENTITY = {
    'agc_cucumber_2018': ('5c12279ea746f8d4e74f758afdc22fa5e33c8a349d57235ada510abc3f05ee1f',
                          '7174369d531dd9c57c0861a189aeecdb20411599888e5fd0939f2f0b8361cc59'),
    'agc_tomato_2019': ('d19f8a5ca9e779824c04dccd627924effdd827f0a96e9845fc5a1feaf5f0dc81',
                        '0c6b2e950421093312819e45d1b96e94ab2843e8d583cb5532c620c33e80b5f4'),
}

# (evaluator_id, dataset_id) -> (configuration_digest, fitness_profile_digest)
EVALUATOR_IDENTITY = {
    ('toy_time_forward@1', 'generated-toy-series@1'):
        ('cf786e77cdfd9e315c677af7b7c1a651aacfbea389cb4a42d61c45e3bf240225',
         '1f1205c26f23791de158d77fab39764191f22d8be04532a07ae1840cadd608f6'),
    ('greenhouse_time_forward@1', 'agc_cucumber_2018'):
        ('7a4bc92cc4a4dd6770e88f3d9fd514571bd0af45f0269bb9c98677f20b552d90',
         'df42b4dee75ab7d01c2853ce328d3d272e34ca662c0a7d7b1d0ac3b3823c0994'),
    ('greenhouse_time_forward@1', 'agc_tomato_2019'):
        ('6d7f45e44321f06ef367560beb7be7798ee827977ee053205995df7370e391dd',
         'df42b4dee75ab7d01c2853ce328d3d272e34ca662c0a7d7b1d0ac3b3823c0994'),
    ('greenhouse_multihorizon_time_forward@1', 'agc_cucumber_2018'):
        ('84620f5791dc93a6eaadd0d885ddd0ced3076ccf155e55c76866c69ebd582313',
         'c54c46ccc92c3a0dc5971170c10c4b193f28a52c21e3c6208d1bf10238e4fd62'),
    ('greenhouse_multihorizon_time_forward@1', 'agc_tomato_2019'):
        ('426b087dd4737f6e78849f82a5aac5ba10ccb6adfa8d8cc9ad686d6ff811c2e3',
         'c54c46ccc92c3a0dc5971170c10c4b193f28a52c21e3c6208d1bf10238e4fd62'),
    ('greenhouse_multihorizon_time_forward@2', 'agc_cucumber_2018'):
        ('5b3121a9cda0a75f1733d9db394c1cf94500cefacb25cf2b3864377bf4a4dac5',
         'c54c46ccc92c3a0dc5971170c10c4b193f28a52c21e3c6208d1bf10238e4fd62'),
    ('greenhouse_multihorizon_time_forward@2', 'agc_tomato_2019'):
        ('54a692efb63141bf9b95feca931644165b35ee2b34e8e56fc3110ff0794a6242',
         'c54c46ccc92c3a0dc5971170c10c4b193f28a52c21e3c6208d1bf10238e4fd62'),
    ('greenhouse_multihorizon_time_forward@3', 'agc_cucumber_2018'):
        ('5f2ad4470d9dbea62881dc5cb4ab08fca1237654332bfa56907b43d42d1ddc70',
         'c54c46ccc92c3a0dc5971170c10c4b193f28a52c21e3c6208d1bf10238e4fd62'),
    ('greenhouse_multihorizon_time_forward@3', 'agc_tomato_2019'):
        ('6fa491b6f4cbbf5de492c8b0fbbfb1e227d3804178c67ef4e6e7861488526eef',
         'c54c46ccc92c3a0dc5971170c10c4b193f28a52c21e3c6208d1bf10238e4fd62'),
    ('greenhouse_multihorizon_time_forward@4', 'agc_cucumber_2018'):
        ('ba992dee1dad2e9938d1addbb6418fb751d00a49b2c30fbf72f9bb4edd4dd6b6',
         'c54c46ccc92c3a0dc5971170c10c4b193f28a52c21e3c6208d1bf10238e4fd62'),
    ('greenhouse_multihorizon_time_forward@4', 'agc_tomato_2019'):
        ('ec77a3ca5fec4e0e2dbd707b7a2a3901affb88d721bb3fc1e318c35c6eb242c2',
         'c54c46ccc92c3a0dc5971170c10c4b193f28a52c21e3c6208d1bf10238e4fd62'),
    ('greenhouse_recipe_multihorizon_forward@1', 'agc_cucumber_2018'):
        ('946d241268fbe4ca8c28a1691a51d276ead2d46800dc7dae30038853288e3b0b',
         'c54c46ccc92c3a0dc5971170c10c4b193f28a52c21e3c6208d1bf10238e4fd62'),
    ('greenhouse_recipe_multihorizon_forward@1', 'agc_tomato_2019'):
        ('8479b0de650885f12e89924f0fe14771ec2c613a0343d5e22406c1185bf37723',
         'c54c46ccc92c3a0dc5971170c10c4b193f28a52c21e3c6208d1bf10238e4fd62'),
}

PREDICTOR_IDENTITY = {
    'toy-rolling-water@1': '6df7f70f37e41c10e52a4cea6bc8beb00b96120d25bfdeb1f6460f420b0793de',
    'greenhouse-rolling-residual@1': 'b9e3f4edfb2334e602c75758106671b1f3f85d370a549e1367666b21346682c2',
    'greenhouse-exogenous-ridge@1': '51ac30f421fa5df29e636f4784c20a373b21f247e2df41c87ea23e3af89ac0ff',
    'greenhouse-targetwise-ridge@1': '3292a6ceba9b13587078f9bbbbb6bcc3039c3d50334d69b1bb66d737340d8471',
    'greenhouse-horizon-targetwise-ridge@1': '5a2e55f8ccc18b0749f17ec35deaa1845f57adb5cbdc6b4bfe848e0620bb60c3',
    'greenhouse-baseline-aligned-ridge@1': '61093ff5ec72788d7e8813e16d4698e835de00681d16ba95b2fe47d87f104373',
    'greenhouse-recipe-ridge@1': '2ea9956c47224c25dd3b3dfb55752bba685e318e3fb80d1ea12e654c2b894587',
}

# Moved deliberately while connecting the evolution axes that the registry
# advertised but nothing read. Every other digest in this file must stay put:
# this one covers the program catalog, so a registry edit is exactly what is
# supposed to move it, and the run-level cost is that a paused run can no longer
# spawn candidates (`workflow_ir` hard-checks `registry_catalog_digest`).
# Rescoring archived candidates is unaffected -- it replays the archived profile.
#   bdb2ed579fc06ebdf122b6e75fd5314abf1856e992ad8b3df02f94eff91b5a62
#     -> workflow `max_attempts` minimum 1 -> 3, so no legal value is
#        unobservable under the host's own retry floor
#   bd1cccdb597073890dfa0adf25200858dd69cbc71451ca017712cd8b9a2fd44d
#     -> `confidence_threshold` became the planner's live critic-escalation
#        threshold: contract [0, 1] -> the cost window [0.5, 0.75], every
#        template's default aligned to 0.5 so template choice stays cost-neutral,
#        and the seed override aligned to the run-level 0.5 so generation 0 costs
#        what it costs today
#   dee924a60de50ff35d792bd3361ef5ca68d4fee48a477c78c156a7bddc319327
#     -> the `directive_policies` category and its first entry
#        `authored_directive@1`, so a candidate can write its own planner
#        strategy instead of selecting one of the registered templates
#   814833bfc2f2c9a2337b5aac4a0d024d41f7c2ea8168d836597e1f7235e3cd69
#     -> seed planner `preset_id` v10 -> v11, whose Skill states the real call
#        ceiling instead of the retired literal two
PROGRAM_CATALOG_DIGEST = '79bcad24da22c93e2f1e8313219d404e3654abd7f0bea6e58b1e39338c852417'

# The evaluator digest payload, field for field. dataset_ids is deliberately
# absent: the per-dataset whitelists are an admission rule, not identity, so
# widening one to admit a new ecology must not move a single digest above.
EVALUATOR_DIGEST_FIELDS = frozenset({
    'evaluator_id', 'implementation', 'evaluation_partition', 'prediction_model_ids',
    'horizons_hours', 'fitness_profile', 'scoring_contract', 'dataset_task_digest',
})


class FrozenIdentityBaselineTests(unittest.TestCase):
    def setUp(self):
        self.registry = EvaluatorRegistry(DatasetRegistry())

    def test_dataset_contract_and_definition_digests_are_unchanged(self):
        for dataset_id, (contract, definition) in DATASET_IDENTITY.items():
            with self.subTest(dataset_id):
                current = dataset_adapter(dataset_id).contract()
                self.assertEqual(current['contract_digest'], contract)
                self.assertEqual(current['definition_digest'], definition)

    def test_evaluator_configuration_and_fitness_digests_are_unchanged(self):
        catalog = {(item['id'], dataset_id) for item in self.registry.catalog()
                   for dataset_id in item['dataset_ids']}
        # Every baselined pair must still be admissible; new pairs may appear.
        self.assertLessEqual(set(EVALUATOR_IDENTITY), catalog)
        for (evaluator_id, dataset_id), (configuration, profile) in EVALUATOR_IDENTITY.items():
            with self.subTest(evaluator=evaluator_id, dataset=dataset_id):
                self.assertEqual(
                    self.registry.evaluator_configuration_digest(evaluator_id, dataset_id),
                    configuration)
                self.assertEqual(
                    self.registry.fitness_profile(evaluator_id, dataset_id).profile_digest,
                    profile)

    def test_predictor_configuration_digests_are_unchanged(self):
        catalog = {item['id'] for item in self.registry.predictor_catalog()}
        self.assertLessEqual(set(PREDICTOR_IDENTITY), catalog)
        for model_id, configuration in PREDICTOR_IDENTITY.items():
            with self.subTest(model_id):
                self.assertEqual(
                    self.registry.predictor_configuration_digest(model_id), configuration)

    def test_program_catalog_digest_is_unchanged(self):
        # workflow_ir validates the seed template against this digest, so a run
        # whose genome cites another value can no longer spawn a candidate.
        self.assertEqual(current_program_registry().catalog_digest, PROGRAM_CATALOG_DIGEST)

    def test_dataset_admission_is_not_part_of_evaluator_identity(self):
        source = inspect.getsource(EvaluatorRegistry.catalog)
        start = source.index('digest_payload = {')
        end = source.index('item["configuration_digest"] = digest(digest_payload)', start)
        keyed = re.findall(r'(?:"(\w+)":|digest_payload\["(\w+)"\] =)', source[start:end])
        assigned = {name or bracketed for name, bracketed in keyed}
        self.assertEqual(assigned, EVALUATOR_DIGEST_FIELDS)


if __name__ == '__main__':
    unittest.main()
