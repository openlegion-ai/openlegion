"""Display helpers: tool formatting, styled output, response rendering."""

from __future__ import annotations

import json
import sys

import click

# ── Tool formatters ─────────────────────────────────────────


def _format_tool_summary(name: str, inp: dict, out: dict) -> str:
    """One-line summary of tool INPUT for the REPL."""
    if name == "exec":
        cmd = inp.get("command", "")
        if len(cmd) > 60:
            cmd = cmd[:57] + "..."
        return f"`{cmd}`"
    elif name in ("read_file", "write_file", "list_files"):
        return inp.get("path", inp.get("directory", ""))
    elif name == "http_request":
        return f"{inp.get('method', 'GET')} {inp.get('url', '')}"
    elif name == "browser_navigate":
        return inp.get("url", "")
    elif name == "browser_click":
        return inp.get("selector", "")
    elif name == "browser_type":
        sel = inp.get("selector", "")
        text = inp.get("text", "")
        if len(text) > 30:
            text = text[:27] + "..."
        return f'{sel} \u2190 "{text}"'
    elif name == "browser_evaluate":
        script = inp.get("script", "")
        for line in script.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("//"):
                if len(line) > 60:
                    line = line[:57] + "..."
                return line
        return script[:60]
    elif name == "browser_screenshot":
        return inp.get("filename", "screenshot.png")
    elif name == "web_search":
        return inp.get("query", "")
    elif name == "memory_save":
        return inp.get("key", inp.get("content", ""))[:60]
    elif name == "memory_search":
        return inp.get("query", "")
    else:
        text = json.dumps(inp, default=str)
        if len(text) > 80:
            text = text[:77] + "..."
        return text


def _format_tool_result_hint(name: str, out: dict) -> str:
    """Concise result hint shown after tool completion."""
    if "error" in out:
        err = str(out["error"])
        if len(err) > 60:
            err = err[:57] + "..."
        return f"error: {err}"
    if name == "exec":
        code = out.get("exit_code", 0)
        return f"exit {code}" if code != 0 else ""
    elif name == "browser_navigate":
        title = out.get("title", "")
        status = out.get("status", "")
        return f"{status} {title}".strip()[:60] if title else ""
    elif name == "browser_click":
        return out.get("url", "")[:60] if out.get("url") else ""
    elif name == "browser_type":
        return ""
    elif name == "browser_screenshot":
        return out.get("path", "")
    elif name == "web_search":
        results = out.get("results", [])
        return f"{len(results)} results" if results else ""
    return ""


# ── Unified tool output display ─────────────────────────────


def display_tool_line(name: str, inp: dict, out: dict, index: int) -> None:
    """Render a single tool call line with optional result hint."""
    summary = _format_tool_summary(name, inp, out)
    hint = _format_tool_result_hint(name, out)
    idx = click.style(f"  [{index}]", fg="bright_black")
    tool = click.style(name, fg="cyan")
    line = f"{idx} {tool}: {summary}"
    if hint:
        is_error = hint.startswith("error:") or hint.startswith("exit ")
        hint_color = "red" if is_error else "bright_black"
        line += click.style(f" \u2192 {hint}", fg=hint_color)
    else:
        line += click.style(" \u2713", fg="green")
    click.echo(line)


# ── Styled output helpers ───────────────────────────────────


def echo_header(text: str) -> None:
    """Section header during startup."""
    click.echo(click.style(f"\n  {text}", bold=True))


def echo_ok(text: str) -> None:
    """Success status line."""
    click.echo(click.style("  \u2713 ", fg="green") + text)


def echo_warn(text: str) -> None:
    """Warning."""
    click.echo(click.style("  \u26a0 ", fg="yellow") + text)


def echo_fail(text: str) -> None:
    """Error."""
    click.echo(click.style("  \u2717 ", fg="red") + text, err=True)


def echo_dim(text: str) -> None:
    """Dim/secondary info."""
    click.echo(click.style(f"  {text}", fg="bright_black"))


def agent_prompt(agent_name: str) -> str:
    """Styled agent response prefix."""
    return click.style(f"{agent_name}> ", fg="green", bold=True)


def user_prompt(agent_name: str) -> str:
    """REPL input prompt showing active agent."""
    return click.style(f"[{agent_name}] ", fg="cyan") + "You> "


# ── Unified response display ────────────────────────────────


def display_response(agent: str, data: dict) -> None:
    """Display tool outputs + response from a non-streaming /chat reply."""
    for i, tool_out in enumerate(data.get("tool_outputs", []), 1):
        display_tool_line(
            tool_out.get("tool", "unknown"),
            tool_out.get("input", {}),
            tool_out.get("output", {}) if isinstance(tool_out.get("output"), dict) else {},
            i,
        )
    resp = data.get("response", "(no response)")
    click.echo(f"\n{agent_prompt(agent)}{resp}\n")


# ── Streaming display ───────────────────────────────────────


def display_stream_tool_start(name: str, inp: dict, tool_count: int) -> None:
    """Show tool start line during streaming (no newline)."""
    summary = _format_tool_summary(name, inp, {})
    idx = click.style(f"  [{tool_count}]", fg="bright_black")
    tool = click.style(name, fg="cyan")
    click.echo(f"{idx} {tool}: {summary}", nl=False)
    sys.stdout.flush()


def display_stream_tool_result(name: str, out: dict) -> None:
    """Complete the tool line with result hint."""
    result_hint = _format_tool_result_hint(name, out if isinstance(out, dict) else {})
    if result_hint:
        is_error = result_hint.startswith("error:") or result_hint.startswith("exit ")
        hint_color = "red" if is_error else "bright_black"
        click.echo(click.style(f" \u2192 {result_hint}", fg=hint_color))
    else:
        click.echo(click.style(" \u2713", fg="green"))


def display_stream_text_delta(agent: str, content: str, is_first: bool) -> None:
    """Write a text delta during streaming."""
    if is_first:
        sys.stdout.write(f"\n{agent_prompt(agent)}")
    sys.stdout.write(content)
    sys.stdout.flush()
