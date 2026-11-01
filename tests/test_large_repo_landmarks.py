"""Focused coverage for bounded, relevant file knowledge."""
from agent.context.read_cache import ReadCache, _hash_str


def _add(cache, path, body):
    path.write_text(body)
    st = path.stat()
    cache.put_content(str(path), body, st.st_mtime, st.st_size)
    cache.record_read(str(path), _hash_str(body), len(body) // 4, st.st_size)


def test_landmarks_reach_late_symbols(tmp_path):
    cache = ReadCache()
    p = tmp_path / "large.py"
    body = "\n".join(f"def f{i}(): pass" for i in range(50)) + "\n"
    _add(cache, p, body)
    result = cache.knowledge_summary()
    assert "def f0()" in result
    assert "def f49()" in result
    assert len(result) <= 1800


def test_prefers_relevant_file_and_discards_stale_cache(tmp_path):
    cache = ReadCache()
    preferred = tmp_path / "target.py"
    other = tmp_path / "other.py"
    _add(cache, preferred, "def target(): pass\n")
    _add(cache, other, "def other(): pass\n")
    summary = cache.knowledge_summary(max_entries=1, relevant_paths=[str(preferred)])
    assert "def target()" in summary
    assert "def other()" not in summary
    preferred.write_text("def replaced(): pass\n")
    assert "def target()" not in cache.knowledge_summary(relevant_paths=[str(preferred)])
