"""Terminal output. Uses `rich` when it is installed, plain text otherwise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _console: Console | None = Console(highlight=False)
except ImportError:  # pragma: no cover - exercised only without rich
    _console = None


def banner(title: str, subtitle: str = "") -> None:
    if _console:
        body = f"[bold cyan]{title}[/]" + (f"\n[dim]{subtitle}[/]" if subtitle else "")
        _console.print(Panel.fit(body, border_style="cyan"))
    else:
        line = "=" * max(len(title), len(subtitle), 40)
        print(f"\n{line}\n{title}" + (f"\n{subtitle}" if subtitle else "") + f"\n{line}")


def stage(index: int, total: int, name: str) -> None:
    if _console:
        _console.rule(f"[bold magenta]Stage {index}/{total} · {name}[/]")
    else:
        print(f"\n---- Stage {index}/{total} · {name} " + "-" * 30)


def section(title: str) -> None:
    if _console:
        _console.rule(f"[bold]{title}[/]", style="dim")
    else:
        print(f"\n-- {title} --")


def _numeric(cell: str) -> bool:
    head = cell.split(" ")[0].rstrip("%")
    try:
        float(head)
        return True
    except ValueError:
        return head in ("—", "-", "?", "")


def table(title: str, columns: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    rows = [[str(c) for c in r] for r in rows]
    if _console:
        t = Table(title=title, title_style="bold", header_style="bold cyan", show_lines=False)
        for i, col in enumerate(columns):
            numeric = i > 0 and bool(rows) and all(_numeric(r[i]) for r in rows)
            t.add_column(col, justify="right" if numeric else "left")
        for r in rows:
            t.add_row(*r)
        _console.print(t)
        return
    widths = [max(len(str(c)), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(columns)]
    print(f"\n{title}")
    print("  " + "  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))


def kv(title: str, pairs: Iterable[tuple[str, object]]) -> None:
    table(title, ("Setting", "Value"), [(k, v) for k, v in pairs])


def _say(rich_prefix: str, plain_prefix: str, msg: str) -> None:
    if _console:
        _console.print(f"{rich_prefix} {msg}")
    else:
        print(f"{plain_prefix}{msg}")


def info(msg: str) -> None:
    _say("[cyan]•[/]", "", msg)


def success(msg: str) -> None:
    _say("[bold green]✓[/]", "OK  ", msg)


def warn(msg: str) -> None:
    _say("[bold yellow]![/]", "WARN  ", msg)


def error(msg: str) -> None:
    _say("[bold red]✗[/]", "ERROR  ", msg)
