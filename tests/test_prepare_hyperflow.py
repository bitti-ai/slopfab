"""CPU arithmetic checks for the Hyperflow companion preparation tool."""
import importlib.util
import json
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location(
    'prepare_hyperflow', Path(__file__).parents[1] / 'tools/prepare_hyperflow.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class Adapter:
    def __init__(self, values): self.values = values
    def get_tensor(self, name): return self.values[name]
    def metadata(self): return {'lora_rank': '2'}


class HyperflowPreparationTest(unittest.TestCase):
    def test_two_time_embedding_against_merged_mlp(self):
        generator = torch.Generator().manual_seed(872)
        source = {name: torch.randn(shape, generator=generator) for name, shape in
                  [('linear_1.weight', (6, 4)), ('linear_1.bias', (6,)),
                   ('linear_2.weight', (3, 6)), ('linear_2.bias', (3,))]}
        factors = {}
        for module in ('time_embedder', 'endpoint_time_embedder'):
            for layer, shape in [('linear_1', (6, 4)), ('linear_2', (3, 6))]:
                prefix = f'transformer.{module}.{layer}'
                factors[prefix + '.lora_A.weight'] = torch.randn((2, shape[1]), generator=generator)
                factors[prefix + '.lora_B.weight'] = torch.randn((shape[0], 2), generator=generator)
        adapter = Adapter(factors)
        t = torch.tensor([0, .25, .999, 1])
        r = torch.tensor([.1, .75, .999, 1])
        results = []
        for endpoint, times in [(False, t), (True, r)]:
            actual = prepare.time_embedding(source, adapter, times, endpoint, 4, -.5)
            module = 'endpoint_time_embedder' if endpoint else 'time_embedder'
            # Independent tiny sinusoid and merged-weight oracle (alpha/rank*strength=-1).
            phase = times[:, None] * torch.tensor([1., .01])[None, :]
            expected = torch.cat((phase.cos(), phase.sin()), dim=1)
            for layer in ('linear_1', 'linear_2'):
                prefix = f'transformer.{module}.{layer}'
                delta = factors[prefix + '.lora_B.weight'] @ factors[prefix + '.lora_A.weight']
                expected = F.linear(expected, source[layer + '.weight'] - delta, source[layer + '.bias'])
                if layer == 'linear_1': expected = F.silu(expected)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            results.append(actual)
        # Even pinned rows need both independently adapted embedders.
        self.assertFalse(torch.equal(results[0][-1], results[1][-1]))
        gate = .25
        mixed = F.silu(results[0] + gate * (results[1] - results[0]))
        wrong = F.silu(results[0]) + gate * (F.silu(results[1]) - F.silu(results[0]))
        self.assertGreater((mixed - wrong).abs().max().item(), .01)

    def test_schedule_pairs_and_pinned_conditions(self):
        grid = [1, .931506, .839236, .703462, .5, .296538, .160764, .068494, 0]
        metadata = {'hyperflow_sigmas': json.dumps(grid),
                    'hyperflow_video_shift': '12', 'hyperflow_audio_shift': '3'}
        pairs = prepare.time_pairs(metadata)
        self.assertEqual(tuple(pairs.shape), (18, 2))
        self.assertEqual(pairs[0, 0], pairs[8, 0])
        self.assertNotEqual(pairs[0, 1], pairs[8, 1])
        self.assertEqual(pairs[7, 1], 1)
        self.assertEqual(pairs[15, 1], 1)
        torch.testing.assert_close(pairs[-2:], torch.tensor([[.999, .999], [1, 1]]))
        metadata['hyperflow_sigmas'] = '[1,0]'
        with self.assertRaises(ValueError): prepare.time_pairs(metadata)


if __name__ == '__main__': unittest.main()
