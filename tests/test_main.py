import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class UrlParsingTests(unittest.TestCase):
    def test_markdown_urls_are_canonicalized_and_deduplicated(self):
        blob = """
        [first](https://www.instagram.com/reels/ABC123xyz9/?utm_source=test)
        https://instagram.com/reel/ABC123xyz9/
        https://www.instagram.com/p/XYZ987abcD/
        """
        urls, rejected = main.parse_urls(blob)
        self.assertEqual(rejected, [])
        self.assertEqual(urls, [
            "https://www.instagram.com/reel/ABC123xyz9/",
            "https://www.instagram.com/reel/XYZ987abcD/",
        ])

    def test_bad_url_is_reported(self):
        urls, rejected = main.parse_urls("https://example.com/not-a-reel")
        self.assertEqual(urls, [])
        self.assertEqual(len(rejected), 1)


class CompletionTests(unittest.TestCase):
    def test_only_complete_reels_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partial = root / "partial123A"
            complete = root / "complete123A"
            partial.mkdir()
            complete.mkdir()
            (partial / "transcript.txt").write_text("partial", encoding="utf-8")
            (complete / "transcript.txt").write_text("done", encoding="utf-8")
            (complete / "bundle.md").write_text("done", encoding="utf-8")
            with patch.object(main, "OUT", root):
                self.assertEqual(main.existing_codes(), {"complete123A"})

    def test_transcript_duration_uses_segment_end(self):
        segments, _ = main.parse_transcript("[ 0.0- 2.5] hello\n[ 2.5- 7.0] world")
        self.assertEqual(segments[-1]["end"], 7.0)


class LocalModelTests(unittest.TestCase):
    def test_local_model_directory_needs_core_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.bin").write_bytes(b"test")
            self.assertEqual(main.ensure_model_downloaded(str(model)),
                             str(model.resolve()))


if __name__ == "__main__":
    unittest.main()
