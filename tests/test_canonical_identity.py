import unittest
from ecologyrsi_dsh.core.identity import FrozenObject, content_id
from ecologyrsi_dsh.core.immutable import freeze_json


class IdentityTests(unittest.TestCase):
    def test_native_frozen_proposals_keep_the_same_content_identity(self):
        value = {'operations': [{'path': ['parameters', 'history_steps'], 'value': 24}]}
        frozen = freeze_json(value)
        self.assertEqual(content_id('proposal@1', value), content_id('proposal@1', frozen))
        self.assertEqual(FrozenObject.from_mapping(frozen).to_dict(), value)

    def test_detached_nested_input_and_output(self):
        value = {'x': {'a': [1, 2]}}
        frozen = FrozenObject.from_mapping(value)
        digest = frozen.digest('test@1')
        value['x']['a'].append(3)
        frozen.to_dict()['x']['a'].append(4)
        self.assertEqual(frozen.to_dict(), {'x': {'a': [1, 2]}})
        self.assertEqual(digest, frozen.digest('test@1'))

    def test_duplicate_unicode_and_nonfinite_rejected(self):
        for raw in ('{"x":1,"x":2}', '{"é":1,"e\\u0301":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                FrozenObject(raw)

    def test_identity_normalization_and_types(self):
        self.assertEqual(content_id('n', {'e\u0301': 1}), content_id('n', {'é': 1}))
        self.assertNotEqual(content_id('n', {'x': True}), content_id('n', {'x': 1}))
        self.assertNotEqual(content_id('n', {}), content_id('m', {}))
