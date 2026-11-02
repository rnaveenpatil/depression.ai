"""Guard against preflight classifier/planner calls returning to the main loop."""
from pathlib import Path


def test_main_loop_has_no_separate_preflight_decision():
    source = Path("src/agent/agent/loop.py").read_text(encoding="utf-8")
    run_block = source.split("    async def _classify_intent(", 1)[0]
    assert "intent = await self._classify_intent(query)" not in run_block
    assert "await self._should_plan(query, intent=intent)" not in run_block
    assert 'intent = "unknown"' in run_block
    assert 'self.enable_intent_classification = False' in run_block
