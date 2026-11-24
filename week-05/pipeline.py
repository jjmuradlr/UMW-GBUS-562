"""pipeline.py - Brookfield's response to one supplier email, in a fixed order.

Runs the specialists from specialists.py, always in this order:
    1. The supply analyst (an agent with the supply MCP server's tools).
    2. A customer notice for each order at risk (skipped if there are none).
    3. A reply to the supplier.
Every draft goes through the review step (review, at most 2 revisions, a name check in
code) and is saved in week-05/drafts, with "pipeline" in its file name. At the end it prints a summary: drafts saved,
drafts marked for a person, decisions for a person, and total tokens.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-05\\pipeline.py --email week-05\\brookfield-bikes\\supplier-emails\\01-northgate-controller-delay.eml
    ... --save run.txt      also writes everything printed to run.txt (UTF-8)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import specialists as sp


class Tee:
    """Writes to the console and to a file at once."""

    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def __getattr__(self, name):            # encoding, isatty, ... come from the console
        return getattr(self.console, name)


def step(n: int, title: str) -> None:
    print(f"\n{'=' * 70}\nStep {n}: {title}\n{'=' * 70}")


def print_summary(label: str, drafts: list[sp.Draft], brief: dict, error: str | None,
                  extra: list[str] = ()) -> None:
    """The end-of-run summary: drafts saved, drafts marked for a person, decisions, tokens.
    extra: lines printed after the email name (the orchestrator adds its own summary)."""
    saved = [d for d in drafts if d.path is not None]
    marked = [d for d in saved if d.needs_person]
    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    print(f"Email: {label}")
    if error:
        print(f"Stopped early: {error}")
    for line in extra:
        print(line)

    print(f"\nDrafts saved: {len(saved)}")
    for d in saved:
        print(f"  {d.recipient:<8} {sp.shown_path(d.path)}  "
              f"(reviewer: {d.verdict}, {d.revisions} revision(s))")
    if any(d.path is None for d in drafts):
        print("  (the supplier writer found no reply needed; nothing saved for the supplier)")

    print(f"\nDrafts marked for a person: {len(marked)}")
    for d in marked:
        print(f"  {sp.shown_path(d.path)}: {'; '.join(d.reasons)}")

    decisions = brief.get("decisions_for_a_person") or []
    print(f"\nDecisions for a person: {len(decisions)}")
    for i, text in enumerate(decisions, 1):
        print(f"  {i}. {text}")

    sp.print_token_totals()


async def run(email_path: Path) -> int:
    sp.reset_tokens()
    print(f"Email: {sp.shown_path(email_path)}")
    print(f"Model: {sp.MODEL}")
    email = sp.load_email(email_path)
    label = email_path.name
    drafts: list[sp.Draft] = []
    brief: dict = {}
    error = None

    try:
        step(1, "supply analyst")
        brief = await sp.run_analyst(email)

        orders = brief.get("orders_at_risk") or []
        if orders:
            step(2, f"customer notices ({len(orders)} order(s) at risk)")
            for order in orders:
                drafts.append(await sp.reviewed_customer_notice(order, label, script="pipeline"))
        else:
            step(2, "customer notices - skipped: no orders at risk")

        step(3, "reply to the supplier")
        drafts.append(await sp.reviewed_supplier_reply(email, brief, label, script="pipeline"))
    except Exception as e:                  # report what was done before the failure
        error = f"{type(e).__name__}: {e}"
        print(f"\nPipeline stopped: {error}")

    print_summary(label, drafts, brief, error)
    return 1 if error else 0


def cli(description: str, run_fn) -> None:
    """--email and --save handling shared by pipeline.py and orchestrator.py.
    run_fn(email_path) is an async function that returns the exit code."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--email", required=True, metavar="PATH", help="path of one supplier email (.eml)")
    parser.add_argument("--save", metavar="FILE", help="also write everything printed to this text file (UTF-8)")
    args = parser.parse_args()

    email_path = Path(args.email)
    if not email_path.is_file():
        parser.error(f"no such email file: {args.email}")

    for stream in (sys.stdout, sys.stderr):  # don't crash on characters the console can't show
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    if not args.save:
        sys.exit(asyncio.run(run_fn(email_path)))

    save_path = Path(args.save)
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out = save_path.open("w", encoding="utf-8")
    except OSError as e:
        parser.error(f"can't write {args.save}: {e.strerror}")
    console_out, console_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(console_out, out), Tee(console_err, out)
    try:
        code = asyncio.run(run_fn(email_path))
        print(f"\nSaved this output to {sp.shown_path(save_path)}")
    finally:
        sys.stdout, sys.stderr = console_out, console_err
        out.close()
    sys.exit(code)


def main() -> None:
    cli("Run Brookfield's specialists on one supplier email, in a fixed order.", run)


if __name__ == "__main__":
    main()
