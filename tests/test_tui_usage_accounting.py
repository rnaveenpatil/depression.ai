"""Regression tests for TUI usage accounting without mounting Textual."""
from types import SimpleNamespace

from agent.tui.app import _query_usage, _session_usage


def _usage(**kw):
    return SimpleNamespace(input_tokens=kw.get("input_tokens", 0),
                           output_tokens=kw.get("output_tokens", 0),
                           tokens_used=kw.get("tokens_used", 0),
                           llm_calls=kw.get("llm_calls", 0),
                           cost=kw.get("cost", 0.0),
                           duration=kw.get("duration", 0.0),
                           start_time=kw.get("start_time", 0),
                           actions_taken=[])


def test_query_ignores_old_loop_context():
    old = _usage(input_tokens=99, tokens_used=99, start_time=100)
    agent = SimpleNamespace(loop=SimpleNamespace(context=old),
                            context=_usage(tokens_used=200),
                            _query_started_at=0)
    coord = SimpleNamespace(plan_agent=None, build_agent=agent)
    query = _query_usage(coord, 200)
    assert query["tokens_used"] == 0


def test_live_session_does_not_double_count_stale_loop():
    historic = _usage(input_tokens=50, output_tokens=10, tokens_used=60)
    old = _usage(input_tokens=50, output_tokens=10, tokens_used=60, start_time=100)
    agent = SimpleNamespace(loop=SimpleNamespace(context=old),
                            context=historic, _query_started_at=200)
    coord = SimpleNamespace(plan_agent=None, build_agent=agent)
    assert _session_usage(coord, now=210)["tokens_used"] == 60


def test_session_includes_current_run_once():
    historic = _usage(tokens_used=60, input_tokens=50, output_tokens=10)
    live = _usage(tokens_used=15, input_tokens=10, output_tokens=5, start_time=201)
    agent = SimpleNamespace(loop=SimpleNamespace(context=live),
                            context=historic, _query_started_at=200)
    coord = SimpleNamespace(plan_agent=None, build_agent=agent)
    assert _session_usage(coord, now=210)["tokens_used"] == 75
