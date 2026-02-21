"""Unit tests for ToolLoopDetector."""

from src.agent.loop_detector import ToolLoopDetector


def test_ok_on_first_call():
    """First call to any tool should return 'ok'."""
    d = ToolLoopDetector()
    assert d.check_before("exec", {"command": "ls"}) == "ok"


def test_ok_on_different_params():
    """Same tool with different arguments should not trigger detection."""
    d = ToolLoopDetector()
    for i in range(5):
        d.record("exec", {"command": f"cmd_{i}"}, '{"ok": true}')
    assert d.check_before("exec", {"command": "cmd_new"}) == "ok"


def test_ok_on_different_tools():
    """Different tools with the same arguments should not trigger detection."""
    d = ToolLoopDetector()
    args = {"query": "test"}
    result = '{"found": true}'
    for _ in range(5):
        d.record("web_search", args, result)
    # Different tool name, same args — should be ok
    assert d.check_before("http_request", args) == "ok"


def test_warn_after_threshold():
    """2 prior identical calls should trigger 'warn'."""
    d = ToolLoopDetector()
    args = {"query": "stuck"}
    result = '{"error": "not found"}'
    d.record("web_search", args, result)
    d.record("web_search", args, result)
    assert d.check_before("web_search", args) == "warn"


def test_block_after_threshold():
    """4 prior identical calls should trigger 'block'."""
    d = ToolLoopDetector()
    args = {"query": "stuck"}
    result = '{"error": "not found"}'
    for _ in range(4):
        d.record("web_search", args, result)
    assert d.check_before("web_search", args) == "block"


def test_terminate_after_threshold():
    """9 prior calls with same tool+params (any result) should trigger 'terminate'."""
    d = ToolLoopDetector()
    args = {"query": "stuck"}
    for i in range(9):
        d.record("web_search", args, f'{{"attempt": {i}}}')
    assert d.check_before("web_search", args) == "terminate"


def test_exempt_tools_always_ok():
    """memory_search and memory_recall should always return 'ok'."""
    d = ToolLoopDetector()
    args = {"query": "test"}
    result = '{"results": []}'
    for _ in range(10):
        d.record("memory_search", args, result)
        d.record("memory_recall", args, result)
    assert d.check_before("memory_search", args) == "ok"
    assert d.check_before("memory_recall", args) == "ok"


def test_sliding_window_eviction():
    """Entries beyond window_size are evicted, reducing counts."""
    d = ToolLoopDetector(window_size=5)
    args = {"q": "a"}
    result = '{"r": 1}'
    # Fill window with 5 identical calls
    for _ in range(5):
        d.record("search", args, result)
    # At this point, 5 identical calls → check should be "block" (>=4)
    assert d.check_before("search", args) == "block"

    # Now add 3 different calls, pushing 3 old entries out
    for i in range(3):
        d.record("other_tool", {"x": i}, '{"ok": true}')
    # Only 2 identical calls remain → warn
    assert d.check_before("search", args) == "warn"

    # Add 2 more different calls, pushing the remaining old entries out
    for i in range(2):
        d.record("other_tool", {"y": i}, '{"ok": true}')
    # 0 identical calls remain → ok
    assert d.check_before("search", args) == "ok"


def test_different_results_no_accumulate():
    """Same tool+params but all different results should not trigger warn.

    _count_identical uses the most-frequent result hash. If every result is
    different, the max count is 1, which is below the warn threshold.
    """
    d = ToolLoopDetector()
    args = {"q": "test"}
    for i in range(4):
        d.record("search", args, f'{{"unique_result": {i}}}')
    # 4 calls, but all different results → most_common count = 1 → ok
    assert d.check_before("search", args) == "ok"


def test_reset_clears_window():
    """reset() should clear all recorded calls."""
    d = ToolLoopDetector()
    args = {"q": "stuck"}
    result = '{"error": "fail"}'
    for _ in range(5):
        d.record("search", args, result)
    assert d.check_before("search", args) == "block"
    d.reset()
    assert d.check_before("search", args) == "ok"


def test_block_still_counts_toward_terminate():
    """Blocked calls (different result hash) still progress the terminate counter.

    When a call is blocked, the result is an error message (different hash from
    the original failing result). _count_any counts ALL entries regardless of
    result, so blocked calls still count toward the terminate threshold.
    """
    d = ToolLoopDetector()
    args = {"q": "stuck"}
    original_result = '{"error": "not found"}'
    blocked_result = '{"error": "Tool loop detected"}'

    # 4 identical calls → would trigger block
    for _ in range(4):
        d.record("search", args, original_result)

    # 5 blocked calls recorded with different result hash
    for _ in range(5):
        d.record("search", args, blocked_result)

    # Total 9 calls with same tool+params → terminate
    assert d.check_before("search", args) == "terminate"


def test_would_terminate_true():
    """would_terminate returns True when terminate threshold is met."""
    d = ToolLoopDetector()
    args = {"q": "stuck"}
    for i in range(9):
        d.record("search", args, f'{{"attempt": {i}}}')
    assert d.would_terminate("search", args) is True


def test_would_terminate_false_below_threshold():
    """would_terminate returns False below the threshold."""
    d = ToolLoopDetector()
    args = {"q": "stuck"}
    for i in range(8):
        d.record("search", args, f'{{"attempt": {i}}}')
    assert d.would_terminate("search", args) is False


def test_would_terminate_exempt_tools():
    """would_terminate returns False for exempt tools even above threshold."""
    d = ToolLoopDetector()
    args = {"query": "test"}
    for _ in range(10):
        d.record("memory_search", args, '{"results": []}')
    assert d.would_terminate("memory_search", args) is False
