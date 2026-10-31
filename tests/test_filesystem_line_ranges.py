"""Line-range filesystem read regression coverage."""
import pytest
from agent.tools.filesystem import FileSystemTool


@pytest.mark.asyncio
async def test_line_range_reads_exact_lines(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    tool = FileSystemTool()
    result = await tool._read(path, {"start_line": 2, "end_line": 3})
    assert result["success"]
    assert result["content"] == "two\nthree\n"
    assert result["next_line"] == 4
    assert not result["truncated"]


@pytest.mark.asyncio
async def test_line_range_small_cap_resumes_at_whole_line(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("first\nsecond\nthird\n", encoding="utf-8")
    tool = FileSystemTool()
    result = await tool._read(path, {"start_line": 1, "end_line": 3, "max_bytes": 7})
    assert result["success"]
    assert result["content"] == "first\n"
    assert result["truncated"]
    assert result["next_line"] == 2


@pytest.mark.asyncio
async def test_invalid_line_range_rejected(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("one\n", encoding="utf-8")
    tool = FileSystemTool()
    result = await tool._read(path, {"start_line": 4, "end_line": 2})
    assert not result["success"]


@pytest.mark.asyncio
async def test_line_range_binary_file_rejected(tmp_path):
    path = tmp_path / "binary.dat"
    path.write_bytes(b"ok\x00bad\n")
    tool = FileSystemTool()
    result = await tool._read(path, {"start_line": 1, "end_line": 1})
    assert result["success"] is False
    assert result.get("binary") is True
