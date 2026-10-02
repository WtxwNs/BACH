import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT / 'code/inference'
sys.path.insert(0, str(INFERENCE))
from inference_utils import (
    create_codec_model, filename_tag, prompt_count, split_lyrics, validate_prompt_args,
)


class HelperTests(unittest.TestCase):
    def args(self, **changes):
        defaults = dict(use_audio_prompt=False, audio_prompt_path='',
                        use_dual_tracks_prompt=False, vocal_track_prompt_path='',
                        instrumental_track_prompt_path='', run_n_segments=2,
                        stage2_batch_size=4, max_new_tokens=3000,
                        prompt_start_time=0.0, prompt_end_time=30.0)
        defaults.update(changes)
        return SimpleNamespace(**defaults)

    def test_single_section_and_last_section_are_not_dropped(self):
        self.assertEqual(prompt_count(split_lyrics('[verse]\nhello'), 2), 2)
        self.assertEqual(prompt_count(['one', 'two'], 2), 3)
        self.assertEqual(prompt_count(['one', 'two', 'three'], 1), 2)

    def test_empty_lyrics_fail_before_model_loading(self):
        for text in ('', 'unstructured lyrics', '[verse]\n'):
            with self.assertRaises(ValueError):
                prompt_count(split_lyrics(text), 2)

    def test_both_dual_track_paths_are_required(self):
        for vocal, instrumental in (('', ''), ('v.wav', ''), ('', 'i.wav')):
            with self.assertRaises(ValueError):
                validate_prompt_args(self.args(use_dual_tracks_prompt=True,
                    vocal_track_prompt_path=vocal, instrumental_track_prompt_path=instrumental))
        validate_prompt_args(self.args(use_dual_tracks_prompt=True,
            vocal_track_prompt_path='v.wav', instrumental_track_prompt_path='i.wav'))

    def test_invalid_generation_ranges_are_rejected(self):
        for changes in ({'run_n_segments': 0}, {'stage2_batch_size': 0},
                        {'max_new_tokens': 10}, {'max_new_tokens': 16383},
                        {'prompt_start_time': -1}, {'prompt_end_time': 0}):
            with self.assertRaises(ValueError):
                validate_prompt_args(self.args(**changes))

    def test_genre_stays_in_one_filename_component(self):
        for genre in ('/ambient/jazz', '..\\pop', '.', 'x' * 1000, '流行 音乐'):
            tag = filename_tag(genre)
            self.assertTrue(tag)
            self.assertLessEqual(len(tag), 80)
            self.assertNotIn('/', tag)
            self.assertNotIn('\\', tag)
            self.assertEqual(Path(tag).name, tag)

    def test_generator_must_be_registered(self):
        config = SimpleNamespace(name='SoundStream', config={'channels': 2})
        self.assertEqual(create_codec_model(config, {'SoundStream': dict}), {'channels': 2})
        with self.assertRaises(ValueError):
            create_codec_model(SimpleNamespace(name='OtherModel', config={}), {'SoundStream': dict})

    def test_help_needs_no_model_dependencies(self):
        result = subprocess.run([sys.executable, '-S', str(INFERENCE / 'infer.py'), '--help'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--genre_txt', result.stdout)

    def test_inference_uses_restricted_loader(self):
        tree = ast.parse((INFERENCE / 'infer.py').read_text())
        loads = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'load'
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch']
        self.assertTrue(loads)
        for node in loads:
            self.assertTrue(any(k.arg == 'weights_only' and isinstance(k.value, ast.Constant)
                                and k.value.value is True for k in node.keywords))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id == 'eval' for n in ast.walk(tree)))


class ShellTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cwd = Path(self.temp.name)
        bindir = self.cwd / 'bin'
        bindir.mkdir()
        fake = bindir / 'python'
        fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\nexit "${FAKE_EXIT:-0}"\n')
        fake.chmod(0o755)
        self.env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'])

    def run_script(self, name, *args, **changes):
        return subprocess.run(['bash', str(ROOT / 'code/finetune/scripts' / name), *args],
                              cwd=self.cwd, env=dict(self.env, **changes),
                              text=True, capture_output=True, timeout=10)

    def test_count_paths_with_spaces_and_punctuation(self):
        data = self.cwd / 'data folder'
        data.mkdir()
        path = data / 'track; notes.bin'
        path.touch()
        result = self.run_script('count_tokens.sh', str(data))
        self.assertEqual(result.returncode, 0, result.stderr)
        logs = list((self.cwd / 'count_token_logs').glob('*.log'))
        self.assertEqual(len(logs), 1)
        lines = logs[0].read_text().splitlines()
        self.assertEqual(lines[-2:], ['--mmap_path', str(path)])

    def test_count_failure_is_propagated(self):
        data = self.cwd / 'data'
        data.mkdir()
        (data / 'track.bin').touch()
        self.assertNotEqual(self.run_script('count_tokens.sh', str(data), FAKE_EXIT='7').returncode, 0)

    def test_preprocess_preserves_tokenizer_argument(self):
        path = 'models/tokenizer with spaces.model'
        for mode in ('cot', 'icl_cot'):
            result = self.run_script('preprocess_data.sh', 'dummy', mode, path, 'dual')
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            self.assertEqual(lines[lines.index('--tokenizer-model') + 1], path)

    def test_preprocess_failure_is_not_reported_as_success(self):
        result = self.run_script('preprocess_data.sh', 'dummy', 'cot', 'model', FAKE_EXIT='7')
        self.assertEqual(result.returncode, 7)
        self.assertNotIn('Preprocessing finished', result.stdout)

    def test_preprocess_requires_tokenizer(self):
        self.assertNotEqual(self.run_script('preprocess_data.sh', 'dummy', 'cot').returncode, 0)


if __name__ == '__main__':
    unittest.main()
