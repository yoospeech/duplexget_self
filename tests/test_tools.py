import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("catalog_tool", ROOT / "src/catalog_tool.py")
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)

class TestTools(unittest.TestCase):
    def test_catalog(self):
        rows = ct.records()
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({row["id"] for row in rows}), 100)
        self.assertEqual(ct.repo_id("https://huggingface.co/datasets/abc/def"), "abc/def")
        self.assertIsNone(ct.repo_id("https://example.org/dataset"))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_stereo_and_mono(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "incoming"
            source.mkdir()
            t = np.arange(24000 * 4) / 24000
            stereo = np.stack([.1 * np.sin(2 * np.pi * 220 * t),
                               .05 * np.sin(2 * np.pi * 330 * t)], axis=1)
            sf.write(source / "stereo.wav", stereo, 24000)
            sf.write(source / "mono.wav", stereo[:, 0], 24000)
            target = Path(td) / "prepared"
            run = subprocess.run([
                sys.executable, str(ROOT / "src/prepare_moshi.py"),
                "--input", str(source), "--out", str(target),
                "--chunk-sec", "3", "--min-sec", "1"
            ], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            rows = [json.loads(s) for s in (target / "train.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            for row in rows:
                wav = target / row["path"]
                audio, sr = sf.read(wav, always_2d=True)
                self.assertEqual(sr, 24000)
                self.assertEqual(audio.shape[1], 2)
                self.assertTrue(wav.with_suffix(".activity.json").exists())
            self.assertEqual(len(json.loads((target / "errors.json").read_text())), 1)

if __name__ == "__main__":
    unittest.main()
