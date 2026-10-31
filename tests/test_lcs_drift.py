"""CPU-only experiment checks; run with Python, numpy and Pillow installed."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('drift', Path(__file__).parents[1] / 'tools/lcs_drift.py')
drift = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drift)


class DriftTest(unittest.TestCase):
    def test_existing_experiment_keeps_its_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/'existing-commands.json'
            manifest.write_text('["original command"]')
            with self.assertRaises(FileExistsError):
                drift.generate(SimpleNamespace(output=root, name='existing',
                                               lcs_settings_file=None, lcs_settings='{}'))
            self.assertEqual(manifest.read_text(), '["original command"]')

    def test_fit_recovers_known_shift_without_color_or_prefix_contamination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            basis = np.eye(96, 3, dtype=np.float32)
            calibration = dict(basis=basis, latent_std=np.full(24, 2.), latent_mean=np.full(24, .25),
                               mean=np.zeros(96), anchor_lcs=np.zeros((8, 3)))
            calibration['anchor_lcs'][7] = 1
            drift.write(root/'cal.safetensors', {'slopfab_lcs': '1'}, calibration)
            early = np.zeros((7, 4, 96), np.float32)
            # Special boundary latent must not enter the reference statistics.
            early[0, :, 8] = 100
            late = np.concatenate([early, np.zeros((6, 4, 96), np.float32)])
            late[-6:, :, 0] = 10  # Color change must not enter learned residual.
            late[-6:, :, 8] = 3
            for name, rows in [('early', early), ('late', late)]:
                drift.write(root/f'{name}.safetensors', {'width': '64', 'height': '64', 'slopfab_latents': 'h3-av-v1'},
                            {'video_rows': rows.reshape(-1, 96)})
            drift.fit(SimpleNamespace(calibration=root/'cal.safetensors', early=root/'early.safetensors',
                                      late=root/'late.safetensors', output=root/'fit.safetensors'))
            _, result = drift.read(root/'fit.safetensors')
            expected = np.zeros(96)
            expected[8] = 1
            np.testing.assert_allclose(result['drift_direction'], expected, atol=1e-7)
            np.testing.assert_allclose(result['drift_reference'], [.25, .01], atol=1e-7)
            report = json.loads((root/'fit.json').read_text())
            self.assertAlmostEqual(report['raw_shift_norm'], 6.)

    def test_metrics_separate_constant_exposure_from_texture(self):
        flat = np.full((32, 32, 3), .4)
        bright = np.full((32, 32, 3), .9)
        texture = flat.copy()
        texture[::2, ::2] += .1
        self.assertEqual(drift.metrics(flat)['highlight_fraction'], 0)
        self.assertEqual(drift.metrics(bright)['highlight_fraction'], 1)
        self.assertAlmostEqual(drift.metrics(flat)['texture_rms'], 0)
        self.assertGreater(drift.metrics(texture)['texture_rms'], 0)


if __name__ == '__main__':
    unittest.main()
