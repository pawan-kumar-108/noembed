"""
Track D requires proving the storage layer "survives a basic crash or
concurrent-access test." This module simulates a process dying mid-write
and asserts the on-disk index is never left in a corrupted, half-written
state -- see src/index.py's save() docstring for the exact guarantee this
is testing.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.index import NoEmbedIndex


class CrashRecoveryTests(unittest.TestCase):
    def test_failed_replace_leaves_original_index_untouched_and_valid(self):
        """
        Simulate a crash that happens after the temp file is fully written
        and fsynced, but before os.replace() swaps it into place (i.e. the
        one moment where a naive "write directly to the real path" approach
        would leave a truncated, corrupt file). Because we only ever write
        to a temp file and swap it in atomically, a failure at this point
        must leave the real index path exactly as it was before the save
        attempt.
        """
        docs_v1 = {"a.txt": "original content here"}
        docs_v2 = {"a.txt": "updated content that would replace the original"}

        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "index.json"

            # A real, valid save -- this is the state that must survive.
            index_v1 = NoEmbedIndex()
            index_v1.build(docs_v1)
            index_v1.save(index_path)

            original_bytes = index_path.read_bytes()

            index_v2 = NoEmbedIndex()
            index_v2.build(docs_v2)

            # Simulate the crash: os.replace() itself fails, as it would if
            # the process were killed in the instant between the fsync
            # completing and the rename syscall landing.
            with patch("src.index.os.replace", side_effect=OSError("simulated crash")):
                with self.assertRaises(OSError):
                    index_v2.save(index_path)

            # The real index file must be byte-identical to the pre-crash
            # version -- not truncated, not partially the new version.
            self.assertEqual(index_path.read_bytes(), original_bytes)

            # And it must still load and search correctly, proving this
            # isn't just "the bytes happen to match" but a genuinely intact,
            # valid index.
            recovered = NoEmbedIndex()
            recovered.load(index_path)
            results = recovered.search("original content", k=5)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0], "a.txt")

    def test_no_leftover_temp_file_after_failed_save(self):
        """
        A failed save() should clean up its own temp file rather than
        littering the index directory with `.index.json.<random>.tmp`
        files -- see the except block in save().
        """
        docs = {"a.txt": "some content"}

        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "index.json"

            index = NoEmbedIndex()
            index.build(docs)

            with patch("src.index.os.replace", side_effect=OSError("simulated crash")):
                with self.assertRaises(OSError):
                    index.save(index_path)

            leftover_tmp_files = list(Path(tmp).glob(".index.json.*.tmp"))
            self.assertEqual(leftover_tmp_files, [])

    def test_first_ever_save_failing_leaves_no_index_at_all(self):
        """
        If the very first save() for a fresh index path fails, there must
        be no index file at all afterward -- not an empty or corrupt one.
        This matters because the CLI treats "index file doesn't exist" as
        a clean, well-defined error state (see cli.py's `exists()` checks).
        """
        docs = {"a.txt": "brand new content"}

        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "index.json"
            self.assertFalse(index_path.exists())

            index = NoEmbedIndex()
            index.build(docs)

            with patch("src.index.os.replace", side_effect=OSError("simulated crash")):
                with self.assertRaises(OSError):
                    index.save(index_path)

            self.assertFalse(index_path.exists())
            self.assertFalse(NoEmbedIndex.exists(index_path))


if __name__ == "__main__":
    unittest.main()