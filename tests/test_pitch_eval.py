import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

SCRIPT = Path(__file__).resolve().parents[1] / 'code/evals/pitch_range/main.py'


class PitchEvalTests(unittest.TestCase):
    def test_empty_directory_needs_no_models(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-S', str(SCRIPT),
                '--input_dir', directory, '--output_dir', directory],
                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('nothing to process', result.stdout)

    @unittest.skipIf(np is None, 'NumPy is required for audio chunk tests')
    def test_padding_and_duration(self):
        spec = importlib.util.spec_from_file_location('pitch_eval', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for samples, expected_chunks in ((100, 1), (101, 2)):
            calls = []
            def infer(chunk, sr, **kwargs):
                calls.append(len(chunk))
                return [0, 440]
            fake_librosa = SimpleNamespace(load=lambda *a, **k: (np.zeros(samples), 10))
            with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,
                {'librosa': fake_librosa, 'tqdm': SimpleNamespace(tqdm=lambda x: x)}):
                output = Path(directory) / 'pitch.txt'
                _, duration = module.process_audio(SimpleNamespace(infer_from_audio=infer),
                    'unused.wav', output, 'cpu', 160, 0.03)
                self.assertEqual(calls, [100] * expected_chunks)
                self.assertEqual(duration, samples / 10)
                self.assertEqual(output.read_text().splitlines(), ['440.00'] * expected_chunks)


if __name__ == '__main__':
    unittest.main()
