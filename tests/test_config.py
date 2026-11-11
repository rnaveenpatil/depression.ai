from agent.config import Config


def test_loop_guardrail_settings_are_configurable():
    cfg = Config.from_dict({
        "loop": {
            "no_progress_streak_cap": 4,
            "no_progress_warn_at": 2,
            "reserve_output_tokens": 6000,
            "verification_mode": "batch",
            "max_batch_verify_attempts": 3,
        }
    })
    cfg.validate()
    assert cfg.loop.no_progress_streak_cap == 4
    assert cfg.loop.no_progress_warn_at == 2
    assert cfg.loop.reserve_output_tokens == 6000
    assert cfg.loop.verification_mode == "batch"
    assert cfg.loop.max_batch_verify_attempts == 3
