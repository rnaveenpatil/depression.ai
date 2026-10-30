"""Regression tests for session task checkpoint persistence."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from agent.context.checkpoint import MAX_BYTES, TaskCheckpointStore


class TaskCheckpointTests(unittest.TestCase):
    def test_roundtrip_and_goal_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskCheckpointStore(Path(tmp))
            record = {
                "version": 1, "goal": "fix tests",
                "completed_tools": ["edit"], "modified_files": ["app.py"],
                "failures": [], "pending_tasks": ["run tests"],
                "next_step": "run tests", "unverified_changes": 1,
            }
            store.save("session:abc", record)
            self.assertEqual(store.load("session:abc", "fix tests"), record)
            self.assertIsNone(store.load("session:abc", "different goal"))
            self.assertEqual(
                store._path("session:abc").name, "sessionabc.json"
            )
            if os.name == "posix":
                self.assertEqual(store._path("session:abc").stat().st_mode & 0o777, 0o600)

    def test_reject_oversized_checkpoint_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskCheckpointStore(Path(tmp))
            store.save("x", {"goal": "safe"})
            with self.assertRaises(ValueError):
                store.save("x", {"goal": "x" * MAX_BYTES})
            self.assertEqual(store.load("x", "safe"), {"goal": "safe"})

    def test_malformed_file_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskCheckpointStore(Path(tmp))
            store.root.mkdir()
            store._path("x").write_text("{invalid", encoding="utf-8")
            self.assertIsNone(store.load("x", "goal"))


if __name__ == "__main__":
    unittest.main()
