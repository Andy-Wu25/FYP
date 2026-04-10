"""Interactive curses-based result browser for code similarity checks.

Launched after the standard report prints.  Users navigate with arrow keys,
press Enter to expand/collapse stored source code, and q to quit.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ResultEntry:
    """One similarity hit with optional stored code."""
    query_name: str
    query_file: str
    rank: int
    distance: float
    meta: Dict
    code: str       # stored source text; empty string if not available
    hit_type: str   # "private", "public", or "self"

    @property
    def has_code(self) -> bool:
        return bool(self.code and self.code.strip())

    @property
    def label(self) -> str:
        repo = self.meta.get("repo_name", "?")
        file = self.meta.get("file_path", "?")
        func = self.meta.get("function_name", "?")
        sl = self.meta.get("start_line", "?")
        el = self.meta.get("end_line", "?")
        lic = self.meta.get("license", "")
        lic_str = f" [{lic}]" if lic and lic != "<unknown>" else ""
        return f"#{self.rank}  {self.distance:.4f}{lic_str}  {repo}  >  {file}:{sl}-{el}  >  {func}"


_MAX_CODE_LINES = 50


def launch_interactive_viewer(entries: List[ResultEntry]) -> None:
    """Launch the interactive viewer if appropriate, otherwise no-op."""
    if not entries:
        return
    if not sys.stdout.isatty():
        return

    has_any_code = any(e.has_code for e in entries)
    if not has_any_code:
        return

    try:
        import curses  # noqa: F811
    except ImportError:
        return

    print(f"\n  {len(entries)} result(s) with stored code available.")
    print("  Press Enter to browse interactively, or q to skip... ", end="", flush=True)

    # Read a single keypress without requiring Enter
    try:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except (ImportError, OSError):
        # Fallback: just read a line
        ch = input()
        ch = ch.strip()[:1] if ch.strip() else "\r"

    print()  # newline after the prompt

    if ch in ("q", "Q", "\x03"):  # q or Ctrl-C
        return

    curses.wrapper(_curses_main, entries)


def _curses_main(stdscr, entries: List[ResultEntry]) -> None:
    import curses

    curses.curs_set(0)
    curses.use_default_colors()

    # Define colour pairs
    try:
        curses.init_pair(1, curses.COLOR_CYAN, -1)     # headers
        curses.init_pair(2, curses.COLOR_GREEN, -1)     # expanded marker
        curses.init_pair(3, curses.COLOR_YELLOW, -1)    # distance highlight
        curses.init_pair(4, curses.COLOR_WHITE, -1)     # code
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_WHITE)  # selected row
        curses.init_pair(6, curses.COLOR_RED, -1)       # close distance
    except curses.error:
        pass

    CYAN = curses.color_pair(1)
    GREEN = curses.color_pair(2)
    YELLOW = curses.color_pair(3)
    CODE_ATTR = curses.color_pair(4) | curses.A_DIM
    SELECTED = curses.color_pair(5)
    RED = curses.color_pair(6) | curses.A_BOLD

    expanded: set[int] = set()
    cursor = 0
    scroll_offset = 0

    def _build_lines():
        """Build virtual line list: each line is (entry_idx | None, text, attr, is_code)."""
        lines = []
        prev_query = None
        for i, e in enumerate(entries):
            query_key = (e.query_name, e.query_file)
            if query_key != prev_query:
                lines.append((None, f"  [{e.query_name}  {e.query_file}]", CYAN | curses.A_BOLD, False))
                prev_query = query_key

            marker = "v" if i in expanded else ">"
            code_tag = "" if e.has_code else "  [no code stored]"
            lines.append((i, f"  {marker} {e.label}{code_tag}", 0, False))

            if i in expanded and e.has_code:
                code_lines = e.code.splitlines()
                shown = code_lines[:_MAX_CODE_LINES]
                for cl in shown:
                    lines.append((None, f"     | {cl}", CODE_ATTR, True))
                if len(code_lines) > _MAX_CODE_LINES:
                    lines.append((None, f"     | ... {len(code_lines) - _MAX_CODE_LINES} more lines", CODE_ATTR, True))
                lines.append((None, "", 0, False))

        return lines

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()

        # Header bar
        title = " CODE SIMILARITY BROWSER "
        help_text = " q=quit  Up/Down=navigate  Enter=toggle code "
        header = title + " " * max(0, width - len(title) - len(help_text)) + help_text
        stdscr.addnstr(0, 0, header[:width], width, curses.A_REVERSE)

        lines = _build_lines()
        total_lines = len(lines)
        content_height = height - 2  # minus header and footer

        # Find the line index of the cursor entry
        cursor_line = 0
        for li, (entry_idx, _, _, _) in enumerate(lines):
            if entry_idx == cursor:
                cursor_line = li
                break

        # Adjust scroll so cursor is visible
        if cursor_line < scroll_offset:
            scroll_offset = cursor_line
        elif cursor_line >= scroll_offset + content_height:
            scroll_offset = cursor_line - content_height + 1
        scroll_offset = max(0, min(scroll_offset, max(0, total_lines - content_height)))

        # Render lines
        for row_i in range(content_height):
            line_idx = scroll_offset + row_i
            if line_idx >= total_lines:
                break
            entry_idx, text, attr, is_code = lines[line_idx]
            screen_row = row_i + 1
            if entry_idx is not None and entry_idx == cursor:
                attr = SELECTED
            try:
                stdscr.addnstr(screen_row, 0, text[:width - 1], width - 1, attr)
            except curses.error:
                pass

        # Footer
        footer = f" {cursor + 1}/{len(entries)} results  |  scroll: {scroll_offset + 1}-{min(scroll_offset + content_height, total_lines)}/{total_lines} "
        try:
            stdscr.addnstr(height - 1, 0, footer[:width], width, curses.A_REVERSE)
        except curses.error:
            pass

        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):  # q, Q, Esc
            break
        elif key == curses.KEY_UP or key == ord("k"):
            if cursor > 0:
                cursor -= 1
        elif key == curses.KEY_DOWN or key == ord("j"):
            if cursor < len(entries) - 1:
                cursor += 1
        elif key in (curses.KEY_ENTER, 10, 13):  # Enter
            if entries[cursor].has_code:
                if cursor in expanded:
                    expanded.discard(cursor)
                else:
                    expanded.add(cursor)
        elif key == curses.KEY_PPAGE:  # Page Up
            cursor = max(0, cursor - content_height)
        elif key == curses.KEY_NPAGE:  # Page Down
            cursor = min(len(entries) - 1, cursor + content_height)
        elif key == curses.KEY_HOME or key == ord("g"):
            cursor = 0
            scroll_offset = 0
        elif key == curses.KEY_END or key == ord("G"):
            cursor = len(entries) - 1
        elif key == curses.KEY_RESIZE:
            pass  # will re-render on next iteration
