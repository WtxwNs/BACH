"""Exercise the stage-two batching function without loading model assets."""
import ast
import copy
from collections import Counter
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(np is None, 'NumPy is required for stage-two array tests')
class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'stage2'
        self.output.mkdir()
        self.calls = []

        def generate(model, prompt, batch_size):
            self.assertGreater(batch_size, 0)
            self.assertGreater(prompt.shape[-1], 0)
            self.calls.append((prompt.shape[-1], batch_size))
            return np.arange(prompt.shape[-1] * 8, dtype=np.int32) % 1024

        # infer.py is a command-line entry point with heavyweight top-level
        # model loading. Compile only this reviewed function for a unit test.
        tree = ast.parse((ROOT / 'code/inference/infer.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'stage2_inference')
        namespace = dict(np=np, os=os, copy=copy, Counter=Counter, tqdm=lambda x: x,
                         stage2_generate=generate,
                         codectool_stage2=SimpleNamespace(ids2npy=lambda x: x.reshape(-1, 8).T))
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'infer.py', 'exec'), namespace)
        self.infer = namespace['stage2_inference']

    def test_short_and_partial_sequences_keep_all_frames(self):
        for frames in (1, 299, 300, 301, 600, 1499):
            with self.subTest(frames=frames):
                path = self.root / f'{frames}.npy'
                np.save(path, np.zeros((1, frames), dtype=np.int32))
                result = self.infer(object(), [str(path)], str(self.output), batch_size=4)
                self.assertEqual(len(result), 1)
                self.assertEqual(np.load(result[0]).shape, (8, frames))

    def test_cached_outputs_are_returned_for_audio_reconstruction(self):
        path = self.root / 'cached.npy'
        np.save(path, np.zeros((1, 50), dtype=np.int32))
        cached = self.output / path.name
        np.save(cached, np.ones((8, 50), dtype=np.int32))
        self.assertEqual(self.infer(object(), [str(path)], str(self.output)), [str(cached)])
        self.assertEqual(self.calls, [])

    def test_empty_prompt_fails_clearly(self):
        path = self.root / 'empty.npy'
        np.save(path, np.zeros((1, 0), dtype=np.int32))
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            self.infer(object(), [str(path)], str(self.output))


if __name__ == '__main__':
    unittest.main()
