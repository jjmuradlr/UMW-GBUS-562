"""supply_mcp_server.py - an MCP server with a manufacturer's supply data tools.

Built with the official MCP Python SDK (`mcp` 2.x, `MCPServer`), like week-04's
hr_mcp_server.py. It speaks MCP over stdio, so an MCP client (Claude Code, or
the week-05 supply analyst) starts it as a subprocess.

Works for any company folder (--org; default week-05/brookfield-bikes) laid out as:
    parts.csv              part, description, stock on hand, supplier
    inbound.csv            purchase order, part, supplier, quantity, arrival day
    bill-of-materials.csv  product, part, parts per unit of product
    orders.csv             order, customer, contact email, product, quantity, ship day, late penalty
    suppliers.csv          supplier, contact name, contact email
    prompts/               the specialists' system prompts
    supplier-emails/       sample supplier emails
Column names may differ between companies (Brookfield's orders.csv has model, qty,
ship_in_days; Copperline's has product, units, ship_by_day). column_map() works out
from each header row which column holds what, and fails with a clear message if it
can't tell.

Every tool is read-only. The server reads the CSV files on every call, so edits to
them take effect at once. Days are counted from today: day 0 is today. A "part unit"
is one part, so an order needs part units = products ordered x parts per product.

Tools
-----
get_part(part_id)
    The part's stock on hand, its scheduled shipments, and the products that use it
    (with parts per unit).

get_orders_for_part(part_id)
    The open orders that need the part, in ship-day order, with part units needed,
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
    .venv\\Scripts\\python.exe week-05\\supply_mcp_server.py --org copperline-coffee
    (--org takes a folder name under week-05 or a path; --data is the old name for it)

Everything this server logs goes to stderr: stdout carries the MCP protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

# ---------------------------------------------------------------------------
# Paths and company folders
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent              # .../UMW-GBUS-562/week-05
DEFAULT_ORG = HERE / "brookfield-bikes"
DEFAULT_DATA = DEFAULT_ORG                          # old name, kept for other scripts
DATA_FILES = ("parts.csv", "inbound.csv", "bill-of-materials.csv", "orders.csv", "suppliers.csv")
PROMPT_FILES = ("analyst.txt", "supplier-writer.txt", "customer-writer.txt", "reviewer.txt", "orchestrator.txt")
DATA_DIR = DEFAULT_ORG                              # set from --org in main()


def shown_path(p: Path) -> str:
    """A path for output: relative to the repository (like week-05\\...), never the full
    C:\\Users\\<name>\\... path."""
    p = Path(p).resolve()
    repo = HERE.parent
    return str(p.relative_to(repo)) if p.is_relative_to(repo) else p.name


def log(msg: str) -> None:
    """Log to stderr; stdout is reserved for MCP messages."""
    print(msg, file=sys.stderr, flush=True)


def resolve_org(value: str | Path | None) -> Path:
    """A company folder: a path, or a folder name under week-05. None means brookfield-bikes.
    Raises ValueError if it isn't a complete company folder."""
    if value is None or str(value).strip() == "":
        org = DEFAULT_ORG
    else:
        p = Path(value)
        org = p if p.is_dir() else HERE / str(value)
    org = org.resolve()
    if not org.is_dir():
        raise ValueError(f"No company folder {value!r} (give a folder under week-05, "
                         f"like brookfield-bikes, or a path).")
    missing = [f for f in DATA_FILES if not (org / f).is_file()]
    missing += [f"prompts/{f}" for f in PROMPT_FILES if not (org / "prompts" / f).is_file()]
    if missing:
        raise ValueError(f"{shown_path(org)} is missing: {', '.join(missing)}")
    return org


# ---------------------------------------------------------------------------
# Working out the columns from each header row
# ---------------------------------------------------------------------------

_PART = [r"^part(_?(id|no|num|number|code))?$", r"part.*(id|no|num|number|code)", r"^sku$"]
_SUPPLIER = [r"^(supplier|vendor)(_?name)?$", r"supplier|vendor|manufacturer"]
_PRODUCT = [r"^(product|model|item|machine)(_?(name|id|code))?$", r"product|model|item|machine|bike"]
_QTY = [r"^(qty|quantity|units|count|amount|pieces|pcs)$", r"qty|quantit|units|count|amount"]

# For each file: (role, patterns in order of preference, required). Roles are assigned in
# this order and a column is used once, so specific roles (email) come before broad ones.
COLUMN_ROLES: dict[str, list[tuple[str, list[str], bool]]] = {
    "parts.csv": [
        ("part_id", _PART, True),
        ("supplier", _SUPPLIER, True),
        ("on_hand", [r"on_?hand|in_?stock|stock|inventory|available"], True),
        ("description", [r"desc|name|title"], False),
    ],
    "inbound.csv": [
        ("po_number", [r"^po(_|$)", r"purchase_?order|^po"], True),
        ("part_id", _PART, True),
        ("supplier", _SUPPLIER, False),
        ("arrives_in_days", [r"arriv|eta", r"deliver|due|day"], True),
        ("qty", _QTY, True),
    ],
    "bill-of-materials.csv": [
        ("part_id", _PART, True),
        ("parts_per_unit", [r"(^|_)per(_|$)", r"qty|quantit|count|units|amount|usage"], True),
        ("product", _PRODUCT, True),
    ],
    "orders.csv": [
        ("order_id", [r"^order(_?(id|no|num|number|ref))?$", r"order"], True),
        ("contact_email", [r"e_?mail"], False),
        ("customer", [r"customer|client|account|buyer", r"company|name"], True),
        ("late_penalty", [r"penalt", r"late"], True),
        ("ship_day", [r"(ship|deliver|due|dispatch).*(day|date)|(day|date).*(ship|deliver|due)",
                      r"ship|deliver|due|dispatch"], True),
        ("product", _PRODUCT, True),
        ("quantity", _QTY, True),
    ],
    "suppliers.csv": [
        ("contact_email", [r"e_?mail"], False),
        ("contact_name", [r"contact|rep(resentative)?|person"], False),
        ("supplier", _SUPPLIER + [r"company|name"], True),
    ],
}


def _norm(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", header.strip().lower()).strip("_")


def column_map(data_dir: Path, name: str) -> dict[str, str]:
    """{role: header} for one CSV file, worked out from its header row.
    Raises ValueError if a required column can't be found, or two columns fit equally."""
    path = Path(data_dir) / name
    with path.open(newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f), [])
    header = [h.strip() for h in header if h.strip()]
    free = list(header)
    found: dict[str, str] = {}
    for role, patterns, required in COLUMN_ROLES[name]:
        for pat in patterns:
            hits = [h for h in free if re.search(pat, _norm(h))]
            if len(hits) == 1:
                found[role] = hits[0]
                free.remove(hits[0])
                break
            if len(hits) > 1:
                raise ValueError(f"{name}: can't tell which column holds {role.replace('_', ' ')}: "
                                 f"{', '.join(hits)} all fit. Header: {', '.join(header)}")
        else:
            if required:
                raise ValueError(f"{name}: no column for {role.replace('_', ' ')}. "
                                 f"Header: {', '.join(header)}")
    return found


def read_table(data_dir: Path, name: str) -> list[dict]:
    """Rows of one CSV file, keyed by role (part_id, product, quantity, ship_day, ...)
    whatever the file's own column names are."""
    cols = column_map(data_dir, name)
    with (Path(data_dir) / name).open(newline="", encoding="utf-8-sig") as f:
        rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]
    return [{role: row.get(col, "") for role, col in cols.items()} for row in rows]


def is_yes(value: str) -> bool:
    return value.strip().lower() in {"yes", "y", "true", "1", "x"}


# ---------------------------------------------------------------------------
# Data loading (re-read on every call)
# ---------------------------------------------------------------------------

def read_csv(name: str) -> list[dict]:
    try:
        return read_table(DATA_DIR, name)
    except OSError as e:
        raise ToolError(f"The data file {name} can't be read ({e.strerror}). Do not estimate any numbers.")
    except ValueError as e:
        raise ToolError(f"{e}. Do not estimate any numbers.")


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


def products_using(part_id: str) -> dict[str, int]:
    """{product: parts per unit} for every product whose bill of materials includes the part."""
    return {row["product"]: to_int(row["parts_per_unit"], "parts per unit")
            for row in read_csv("bill-of-materials.csv") if row["part_id"].upper() == part_id.upper()}


def shipments_for(part_id: str) -> list[dict]:
    rows = [r for r in read_csv("inbound.csv") if r["part_id"].upper() == part_id.upper()]
    out = [{"po_number": r["po_number"], "supplier": r.get("supplier", ""),
            "qty": to_int(r["qty"], "inbound quantity"),
            "arrives_in_days": to_int(r["arrives_in_days"], "arrival day")} for r in rows]
    return sorted(out, key=lambda s: s["arrives_in_days"])


def orders_for(part_id: str) -> list[dict]:
    """Open orders that need the part, in ship-day order (ties: order_id)."""
    per_unit = products_using(part_id)
    out = []
    for r in read_csv("orders.csv"):
        if r["product"] not in per_unit:
            continue
        quantity = to_int(r["quantity"], "order quantity")
        out.append({
            "order_id": r["order_id"],
            "customer": r["customer"],
            "product": r["product"],
            "quantity": quantity,
            "parts_per_unit": per_unit[r["product"]],
            "units_needed": quantity * per_unit[r["product"]],
            "ship_day": to_int(r["ship_day"], "ship day"),
            "late_penalty": is_yes(r["late_penalty"]),
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
            "product": o["product"],
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
    quantity, and arrival day counted from today), and the products that use it with
    how many of the part each unit of product needs."""
    part = find_part(part_id)
    pid = part["part_id"]
    return as_json({
        "part_id": pid,
        "description": part.get("description", ""),
        "supplier": part["supplier"],
        "on_hand": to_int(part["on_hand"], "stock on hand"),
        "scheduled_shipments": shipments_for(pid),
        "used_in": [{"product": m, "parts_per_unit": q} for m, q in products_using(pid).items()],
    })


def get_orders_for_part(part_id: str) -> str:
    """Return the open orders that need a part, in ship-day order: order_id, customer,
    product, quantity (units of product ordered), units_needed (quantity times parts per
    unit), ship_day (days from today), and late_penalty (true if shipping late carries a
    penalty)."""
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
    on_hand = to_int(part["on_hand"], "stock on hand")
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


def build_server(company: str) -> MCPServer:
    server = MCPServer(
        name="supply-data",
        instructions=(
            f"Read-only supply data for {company}: parts, scheduled shipments, open orders, and "
            "coverage checks. Days are counted from today (day 0); a unit is one part. Report "
            "numbers exactly as the tools return them."
        ),
    )
    server.add_tool(get_part, name="get_part")
    server.add_tool(get_orders_for_part, name="get_orders_for_part")
    server.add_tool(check_coverage, name="check_coverage")
    return server


def main() -> None:
    global DATA_DIR
    parser = argparse.ArgumentParser(description="Supply data MCP server (stdio) for one company folder.")
    parser.add_argument("--org", "--data", dest="org", metavar="FOLDER",
                        help="company folder: a name under week-05 or a path (default: brookfield-bikes)")
    args = parser.parse_args()
    try:
        DATA_DIR = resolve_org(args.org)
        for name in COLUMN_ROLES:               # fail now, not on the first tool call
            column_map(DATA_DIR, name)
    except ValueError as e:
        parser.error(str(e))
    log(f"Supply data: {shown_path(DATA_DIR)}")
    build_server(DATA_DIR.name.replace("-", " ").title()).run("stdio")


if __name__ == "__main__":
    main()
