"""Guard against preflight classifier/planner calls returning to the main loop."""
from pathlib import Path


def test_main_loop_has_no_separate_preflight_decision():
    source = Path("src/agent/agent/loop.py").read_text(encoding="utf-8")
    run_block = source.split("    async def _classify_intent(", 1)[0]
    assert "intent = await self._classify_intent(query)" not in run_block
    assert "await self._should_plan(query, intent=intent)" not in run_block
    assert 'intent = "unknown"' in run_block
    assert 'self.enable_intent_classification = False' in run_block


def test_progress_guidance_and_batch_verification_remain():
    source = Path("src/agent/agent/loop.py").read_text(encoding="utf-8")
    assert 'implementation_guidance_sent' in source
    assert 'make the targeted change now instead of browsing more' in source
    assert 'if self._should_batch_verify(response_text):' in source
    assert 'self._root_cause_gate(name, args)' in source


def test_instant_verification_is_removed_from_execution_path():
    source = Path("src/agent/agent/loop.py").read_text(encoding="utf-8")
    assert "_auto_verify_single_file_change" not in source
    assert 'self.verification_mode == "per_step"' not in source
    assert 'self.enable_planning = False' in source
    assert 'if self._should_batch_verify(response_text):' in source
