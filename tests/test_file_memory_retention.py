"""Regression tests for file landmarks retained through compaction."""
import os
from types import SimpleNamespace

import pytest

from agent.context.read_cache import ReadCache
from agent.context.runtime import add_tool_result


def test_landmarks_follow_original_full_file_and_invalidate(tmp_path):
    path = tmp_path / "sample.py"
    body = "import os\nclass Sample:\n    pass\n\ndef handle():\n    return 1\n"
    path.write_text(body)
    cache = ReadCache()
    st = path.stat()
    cache.put_content(str(path), body, st.st_mtime, st.st_size)
    from agent.context.read_cache import _hash_str
    cache.record_read(str(path), _hash_str(body), len(body) // 4, st.st_size)
    assert "class Sample:" in cache.knowledge_summary()
    path.write_text("def updated():\n    pass\n")
    assert "class Sample:" not in cache.knowledge_summary()
    cache.invalidate(str(path))
    assert not cache.knowledge_summary()


@pytest.mark.asyncio
async def test_runtime_keeps_original_hash_when_tool_payload_truncates(tmp_path):
    path = tmp_path / "large.py"
    body = "def important():\n    return 1\n" + ("x" * 16000)
    path.write_text(body)
    cache = ReadCache()
    output = []
    class Context:
        async def add_message(self, **kwargs):
            output.append(kwargs)
    call = SimpleNamespace(id="call1", name="read_file", arguments={"filePath": str(path)})
    await add_tool_result(Context(), call, {"success": True, "content": body}, context_window=4096, read_cache=cache)
    from agent.context.read_cache import _hash_str
    assert cache.get_index()[str(path)].sha == _hash_str(body)
    assert "_truncated" in output[0]["content"]
    assert "important" in cache.knowledge_summary()
