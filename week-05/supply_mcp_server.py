"""supply_mcp_server.py - an MCP server with Brookfield Bicycle Company's supply data tools.

Built with the official MCP Python SDK (`mcp` 2.x, `MCPServer`), like week-04's
hr_mcp_server.py. It speaks MCP over stdio, so an MCP client (Claude Code, or
the week-05 supply analyst) starts it as a subprocess.

Every tool is read-only. The server reads the CSV files in the data folder
(--data, default week-05/brookfield-bikes) on every call, so edits to them take
effect at once:
    parts.csv              part_id, description, on_hand, supplier
    inbound.csv            po_number, part_id, supplier, qty, arrives_in_days
    bill-of-materials.csv  model, part_id, qty_per_bike
    orders.csv             order_id, customer, contact_email, model, qty, ship_in_days, late_penalty

Days are counted from today: day 0 is today. A "unit" is one part, so an order
needs units = bikes x parts per bike.

Tools
-----
get_part(part_id)
    The part's stock on hand, its scheduled shipments, and the models that use
    it (with parts per bike).

get_orders_for_part(part_id)
    The open orders that need the part, in ship-day order, with units needed,
    ship day, and late penalty.

check_coverage(part_id, arrivals)
    arrivals (a list of quantity + day) REPLACES all of the part's scheduled
    shipments. Orders are filled complete, in ship-day order (ties: order_id):
    an order gets its units only after every order ahead of it has all of
    its units, so a short order holds up the ones after it. For each order:
    covered, or units short on its ship day and the first day it can ship
    (null if these arrivals never cover it).

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-05\\supply_mcp_server.py
    .venv\\Scripts\\python.exe week-05\\supply_mcp_server.py --data week-05\\brookfield-bikes

Everything this server logs goes to stderr: stdout carries the MCP protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent              # .../UMW-GBUS-562/week-05
DEFAULT_DATA = HERE / "brookfield-bikes"
DATA_FILES = ("parts.csv", "inbound.csv", "bill-of-materials.csv", "orders.csv")
DATA_DIR = DEFAULT_DATA                             # set from --data in main()


def shown_path(p: Path) -> str:
    """A path for output: relative to the repository (like week-05\\...), never the full
    C:\\Users\\<name>\\... path."""
    p = Path(p).resolve()
    repo = HERE.parent
    return str(p.relative_to(repo)) if p.is_relative_to(repo) else p.name


def log(msg: str) -> None:
    """Log to stderr; stdout is reserved for MCP messages."""
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Data loading (re-read on every call)
# ---------------------------------------------------------------------------

def read_csv(name: str) -> list[dict]:
    path = DATA_DIR / name
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]
    except OSError as e:
        raise ToolError(f"The data file {name} can't be read ({e.strerror}). Do not estimate any numbers.")


def to_int(value: str, what: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise ToolError(f"The data has a non-numeric {what} ({value!r}). Do not estimate any numbers.")


def find_part(part_id: str) -> dict:
    wanted = (part_id or "").strip().upper()
    parts = read_csv("parts.csv")
    for row in parts:
        if row["part_id"].upper() == wanted:
            return row
    known = ", ".join(r["part_id"] for r in parts) or "none"
    raise ToolError(f"No part {part_id!r} in parts.csv. Known parts: {known}.")


def models_using(part_id: str) -> dict[str, int]:
    """{model: parts per bike} for every model whose bill of materials includes the part."""
    return {row["model"]: to_int(row["qty_per_bike"], "qty_per_bike")
            for row in read_csv("bill-of-materials.csv") if row["part_id"].upper() == part_id.upper()}


def shipments_for(part_id: str) -> list[dict]:
    rows = [r for r in read_csv("inbound.csv") if r["part_id"].upper() == part_id.upper()]
    out = [{"po_number": r["po_number"], "supplier": r["supplier"],
            "qty": to_int(r["qty"], "inbound qty"),
            "arrives_in_days": to_int(r["arrives_in_days"], "arrives_in_days")} for r in rows]
    return sorted(out, key=lambda s: s["arrives_in_days"])


def orders_for(part_id: str) -> list[dict]:
    """Open orders that need the part, in ship-day order (ties: order_id)."""
    per_bike = models_using(part_id)
    out = []
    for r in read_csv("orders.csv"):
        if r["model"] not in per_bike:
            continue
        bikes = to_int(r["qty"], "order qty")
        out.append({
            "order_id": r["order_id"],
            "customer": r["customer"],
            "model": r["model"],
            "bikes": bikes,
            "parts_per_bike": per_bike[r["model"]],
            "units_needed": bikes * per_bike[r["model"]],
            "ship_day": to_int(r["ship_in_days"], "ship_in_days"),
            "late_penalty": r["late_penalty"].lower() == "yes",
        })
    out.sort(key=lambda o: (o["ship_day"], o["order_id"]))
    return out


def as_json(data) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Coverage arithmetic (all of it here, never left to Claude)
# ---------------------------------------------------------------------------

def coverage(on_hand: int, arrivals: list[tuple[int, int]], orders: list[dict]) -> list[dict]:
    """Fill orders complete, in ship-day order. arrivals: [(day, qty), ...].

    Because orders are filled complete and in order, an order can take units only
    once every order ahead of it is full. So on day t, the units available to an
    order are supply(t) minus everything the orders ahead of it need, and it can
    ship on the first day supply covers its own need plus theirs.
    """
    arrivals = sorted(arrivals)

    def supply_by(day: int) -> int:
        return on_hand + sum(q for d, q in arrivals if d <= day)

    def first_day_covering(total_needed: int) -> int | None:
        if on_hand >= total_needed:
            return 0
        running = on_hand
        for day, qty in arrivals:
            running += qty
            if running >= total_needed:
                return day
        return None

    results, ahead = [], 0                  # ahead = units needed by the orders before this one
    for o in orders:
        need = o["units_needed"]
        available = max(0, supply_by(o["ship_day"]) - ahead)
        short = max(0, need - available)
        row = {
            "order_id": o["order_id"],
            "customer": o["customer"],
            "model": o["model"],
            "units_needed": need,
            "ship_day": o["ship_day"],
            "late_penalty": o["late_penalty"],
            "covered": short == 0,
        }
        if short:
            first = first_day_covering(ahead + need)
            row["units_short"] = short
            row["first_ship_day"] = first
            row["days_late"] = None if first is None else first - o["ship_day"]
            if first is None:
                row["note"] = "These arrivals never cover this order."
        results.append(row)
        ahead += need
    return results


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class Arrival(BaseModel):
    qty: int = Field(ge=1, description="Number of units (parts) arriving.")
    day: int = Field(ge=0, description="Day they arrive, counted from today (0 = today).")


def get_part(part_id: str) -> str:
    """Return a part's stock on hand, its scheduled shipments (purchase order, supplier,
    quantity, and arrival day counted from today), and the bike models that use it with
    how many of the part each bike needs."""
    part = find_part(part_id)
    pid = part["part_id"]
    return as_json({
        "part_id": pid,
        "description": part["description"],
        "supplier": part["supplier"],
        "on_hand": to_int(part["on_hand"], "on_hand"),
        "scheduled_shipments": shipments_for(pid),
        "used_in": [{"model": m, "parts_per_bike": q} for m, q in models_using(pid).items()],
    })


def get_orders_for_part(part_id: str) -> str:
    """Return the open orders that need a part, in ship-day order: order_id, customer,
    model, bikes, units_needed (bikes times parts per bike), ship_day (days from today),
    and late_penalty (true if shipping late carries a penalty)."""
    part = find_part(part_id)
    return as_json({"part_id": part["part_id"], "orders": orders_for(part["part_id"])})


def check_coverage(part_id: str, arrivals: list[Arrival]) -> str:
    """Check whether a part's open orders are covered under a new list of arrivals.

    arrivals (each a qty and a day counted from today) REPLACES all of the part's
    scheduled shipments; stock on hand is added to it. Pass an empty list if nothing
    will arrive. To compare a supplier's options, call this once per option.

    Orders are filled complete, in ship-day order, so a short order holds up the ones
    after it. For each order it returns covered: true, or covered: false with
    units_short (units missing on its ship day), first_ship_day (null if never
    covered), and days_late. Report these numbers as given; do not recalculate them."""
    part = find_part(part_id)
    pid = part["part_id"]
    on_hand = to_int(part["on_hand"], "on_hand")
    results = coverage(on_hand, [(a.day, a.qty) for a in arrivals], orders_for(pid))
    return as_json({
        "part_id": pid,
        "on_hand": on_hand,
        "arrivals_checked": sorted(({"qty": a.qty, "day": a.day} for a in arrivals), key=lambda a: a["day"]),
        "replaced_shipments": shipments_for(pid),
        "fill_rule": "orders filled complete, in ship-day order",
        "all_covered": all(r["covered"] for r in results),
        "orders": results,
    })


def build_server() -> MCPServer:
    server = MCPServer(
        name="brookfield-supply",
        instructions=(
            "Read-only supply data for Brookfield Bicycle Company: parts, scheduled shipments, "
            "open orders, and coverage checks. Days are counted from today (day 0); a unit is one "
            "part. Report numbers exactly as the tools return them."
        ),
    )
    server.add_tool(get_part, name="get_part")
    server.add_tool(get_orders_for_part, name="get_orders_for_part")
    server.add_tool(check_coverage, name="check_coverage")
    return server


def main() -> None:
    global DATA_DIR
    parser = argparse.ArgumentParser(description="Brookfield supply data MCP server (stdio).")
    parser.add_argument("--data", metavar="FOLDER",
                        help="folder holding parts.csv, inbound.csv, bill-of-materials.csv, orders.csv "
                             "(default: week-05/brookfield-bikes)")
    args = parser.parse_args()
    data = Path(args.data).resolve() if args.data else DEFAULT_DATA
    missing = [f for f in DATA_FILES if not (data / f).is_file()]
    if missing:
        parser.error(f"{shown_path(data)} is missing: {', '.join(missing)}")
    DATA_DIR = data
    log(f"Supply data: {shown_path(DATA_DIR)}")
    build_server().run("stdio")


if __name__ == "__main__":
    main()
