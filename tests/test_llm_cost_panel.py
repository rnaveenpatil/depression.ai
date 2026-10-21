from types import SimpleNamespace as NS

from agent.tui.app import _query_elapsed, _query_usage, _session_usage


def ctx(start, inp, out, calls=1, cost=0.0, tools=0, **extra):
    data = {
        "start_time": start,
        "input_tokens": inp,
        "output_tokens": out,
        "tokens_used": inp + out,
        "llm_calls": calls,
        "cost": cost,
        "actions_taken": [{} for _ in range(tools)],
    }
    data.update(extra)
    return NS(**data)


def agent(context, loop_ctx=None, started_at=0.0):
    return NS(context=context, loop=NS(context=loop_ctx), _query_started_at=started_at)


def coord(plan=None, build=None, mode="build"):
    c = NS(plan_agent=plan, build_agent=build, current_mode=mode)
    c.get_current_agent = lambda: build if mode != "plan" else plan
    return c


def test_query_sums_plan_and_build_after_marker():
    plan = agent({}, ctx(100.0, 10, 5, calls=2, cost=0.1, tools=3))
    build = agent({}, ctx(101.0, 7, 3, calls=1, cost=0.2, tools=2))
    q = _query_usage(coord(plan, build, mode="auto"), 99.0)
    assert q["input_tokens"] == 17
    assert q["output_tokens"] == 8
    assert q["tokens_used"] == 25
    assert q["llm_calls"] == 3
    assert q["tool_calls"] == 5
    assert abs(q["cost"] - 0.3) < 1e-9


def test_query_ignores_previous_turn_contexts():
    plan = agent({}, ctx(50.0, 100, 100, calls=9))
    build = agent({}, ctx(60.0, 100, 100, calls=9))
    q = _query_usage(coord(plan, build, mode="auto"), 70.0)
    assert q["tokens_used"] == 0
    assert q["llm_calls"] == 0


def test_query_live_phase_of_auto():
    plan = agent({}, ctx(100.0, 10, 5, calls=2))
    build = agent({}, ctx(101.0, 4, 1, calls=1))
    q = _query_usage(coord(plan, build, mode="auto"), 99.5)
    assert q["tokens_used"] == 20
    assert q["llm_calls"] == 3


def test_session_sums_both_agents_and_metrics():
    plan_ctx = {"input_tokens": 10, "output_tokens": 5, "tokens_used": 15,
                "llm_calls": 2, "cost": 0.1, "duration": 3.0,
                "tools_used": {"read": 2}}
    build_ctx = {"input_tokens": 7, "output_tokens": 3, "tokens_used": 10,
                 "llm_calls": 1, "cost": 0.2, "duration": 2.0,
                 "tools_used": {"bash": 1}}
    plan = agent(plan_ctx, ctx(50.0, 10, 5), started_at=0.0)
    build = agent(build_ctx, ctx(60.0, 7, 3), started_at=0.0)
    s = _session_usage(coord(plan, build, mode="auto"), now=200.0)
    assert s["input_tokens"] == 17
    assert s["output_tokens"] == 8
    assert s["tokens_used"] == 25
    assert s["llm_calls"] == 3
    assert s["tool_calls"] == 3
    assert abs(s["cost"] - 0.3) < 1e-9
    assert abs(s["duration"] - 5.0) < 1e-9


def test_session_adds_only_in_flight_run_without_double_count():
    base = {"input_tokens": 7, "output_tokens": 3, "tokens_used": 10,
            "llm_calls": 1, "cost": 0.2, "duration": 2.0,
            "tools_used": {"bash": 1}}
    live = ctx(100.0, 4, 1, calls=1, cost=0.1, tools=2)
    build = agent(base, live, started_at=100.0)
    s = _session_usage(coord(agent({}), build), now=102.0)
    assert s["tokens_used"] == 15
    assert s["llm_calls"] == 2
    assert s["tool_calls"] == 3
    assert abs(s["cost"] - 0.3) < 1e-9
    assert abs(s["duration"] - 4.0) < 1e-9

    # Once the run is folded, _query_started_at is cleared and the live
    # context must not be added a second time.
    build._query_started_at = 0.0
    folded = dict(base)
    folded.update({"input_tokens": 11, "output_tokens": 4, "tokens_used": 15,
                   "llm_calls": 2, "cost": 0.3, "duration": 4.0})
    build.context = folded
    s2 = _session_usage(coord(agent({}), build), now=102.0)
    assert s2["tokens_used"] == 15
    assert s2["llm_calls"] == 2


def test_query_duration_only_runs_while_active_or_frozen():
    assert _query_elapsed(False, 100.0, 0.0, now=200.0) == 0.0
    assert _query_elapsed(True, 100.0, 0.0, now=102.5) == 2.5
    assert _query_elapsed(False, 100.0, 2.5, now=200.0) == 2.5


def test_intent_classification_usage_is_recorded():
    import asyncio

    from agent.agent.loop import AgentLoop, LoopContext

    class FakeLLM:
        def get_current_model(self):
            return "test-model"

        async def complete(self, **kwargs):
            return NS(
                content="question",
                usage={"prompt_tokens": 4, "completion_tokens": 1,
                       "total_tokens": 5},
            )

    loop = object.__new__(AgentLoop)
    loop.context = LoopContext()
    loop.enable_intent_classification = True
    loop.llm = FakeLLM()

    intent = asyncio.run(loop._classify_intent("compose a poem about rivers"))

    assert intent == "question"
    assert loop.context.llm_calls == 1
    assert loop.context.input_tokens == 4
    assert loop.context.output_tokens == 1
    assert loop.context.tokens_used == 5
