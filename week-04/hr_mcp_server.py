"""hr_mcp_server.py - an MCP server with HR tools for any organization.

Built with the official MCP Python SDK (`mcp` 2.x, `MCPServer`). It speaks MCP
over stdio, so Claude Code (or any MCP client) starts it as a subprocess.

Organization folder (--org PATH, default week-04/hartwell-hr) holds:
    hr-docs/            HTML policy documents
    employees.csv       one row per employee; column names can differ between
                        organizations (see "Employee records" below)
    sample-emails/      example emails (used by agent.py, not by this server)
    system_prompt.txt   the agent's instructions (used by agent.py)
A bare name like "ridgeline-hotels" is looked up inside week-04/.

Tools
-----
search_policy(query)
    Searches the organization's HR policy documents and returns the 4 closest
    chunks, each with its source file. The documents are split into one chunk
    per section and stored in a Chroma database in week-04/chroma_db/<org>/
    (one per organization, all ignored by Git), embedded with Chroma's built-in
    model (all-MiniLM-L6-v2).

get_leave_balance
    Reads leave balances from the organization's employees.csv. A mode must be
    chosen; there is no default:
    - Employee mode (--as-employee ADDRESS): takes no inputs and returns only
      that one employee's balance. There is no way to ask for anyone else's.
      agent.py always uses this mode, with the address from the email's From:
      header (worked out in code, never chosen by Claude).
    - HR staff mode (--hr-staff): takes any employee's work email. Only for HR
      staff's own tools, never for answering employee email.
    If no record matches, if two rows share the address, or if the balances are
    blank, the tool returns an error telling Claude not to state a balance. A
    single blank balance is reported as NOT ON RECORD, never as an empty value.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --hr-staff           # HR staff mode, Hartwell
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --org ridgeline-hotels --hr-staff
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --as-employee dana.whitfield@hartwellfoods.example
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --org ridgeline-hotels --rebuild   # rebuild the index, then exit

The index is also rebuilt automatically when a policy document is added,
removed, or changed.

Everything this server logs goes to stderr: stdout carries the MCP protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent              # .../UMW-GBUS-562/week-04
DEFAULT_ORG = HERE / "hartwell-hr"
DB_ROOT = HERE / "chroma_db"                        # Git-ignored (".gitignore": chroma_db/)
ORG_ITEMS = ("hr-docs", "employees.csv", "sample-emails", "system_prompt.txt")
COLLECTION = "hr_policies"

# Set by use_org(); the defaults keep the old Hartwell behavior.
DATA_DIR = DEFAULT_ORG
DOCS_DIR = DATA_DIR / "hr-docs"
EMPLOYEES_CSV = DATA_DIR / "employees.csv"
DB_DIR = DB_ROOT / DATA_DIR.name                    # one Chroma database per organization


def shown_path(p: Path) -> str:
    """A path for output: relative to the repository (like week-04\\...), never the full
    C:\\Users\\<name>\\... path. Files beside the repo show as ..\\name; anything else as its name."""
    p = Path(p).resolve()
    repo = HERE.parent
    if p.is_relative_to(repo):
        return str(p.relative_to(repo))
    if p.parent == repo.parent:
        return str(Path("..") / p.name)
    return p.name


def resolve_org(value: str | Path | None) -> Path:
    """Turn --org into an organization folder. A bare name is looked up in week-04/.

    Raises ValueError naming any of the four expected items that is missing.
    """
    if value is None:
        org = DEFAULT_ORG
    else:
        org = Path(value)
        if not org.is_dir() and not org.is_absolute() and (HERE / org).is_dir():
            org = HERE / org
    org = org.resolve()
    if not org.is_dir():
        raise ValueError(f"organization folder not found: {value}")
    missing = [item for item in ORG_ITEMS if not (org / item).exists()]
    if missing:
        raise ValueError(f"{org.name} is missing: {', '.join(missing)} "
                         f"(an organization folder must hold {', '.join(ORG_ITEMS)} directly)")
    return org


def use_org(org: Path) -> None:
    """Point the server at one organization folder."""
    global DATA_DIR, DOCS_DIR, EMPLOYEES_CSV, DB_DIR, _collection
    DATA_DIR = org
    DOCS_DIR = org / "hr-docs"
    EMPLOYEES_CSV = org / "employees.csv"
    DB_DIR = DB_ROOT / org.name
    _collection = None

TOP_K = 4
MAX_CHUNK_WORDS = 400       # sections longer than this are split at paragraph boundaries
CHUNKER_VERSION = "1"       # bump when the chunking logic changes, so the index rebuilds
SUPPORTED = {".html", ".htm"}


def log(msg: str) -> None:
    """Log to stderr; stdout is reserved for MCP messages."""
    print(msg, file=sys.stderr, flush=True)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Chunking: one chunk per <h2>/<h3> section, prefixed with "<title> > <heading>"
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def parse_html(path: Path) -> tuple[str, str, list[tuple[str, list[str]]]]:
    """Return (title, note, [(heading, paragraphs), ...]) for one HTML policy page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    page_title = clean(soup.title.get_text()) if soup.title else ""
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "template"]):
        tag.decompose()
    for tag in soup.select(".bar"):
        tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    h1 = root.find("h1")
    title = clean(h1.get_text()) if h1 else (page_title.split("|")[0].strip() or path.stem)
    if h1:
        h1.decompose()
    meta = root.select_one(".meta")
    note = clean(meta.get_text()) if meta else ""
    if meta:
        meta.decompose()

    headings = {"h2", "h3", "h4", "h5", "h6"}
    blocks = {"p", "li", "tr", "pre", "blockquote", "dt", "dd"}
    sections: list[tuple[str, list[str]]] = [("Introduction", [])]
    for el in root.find_all(headings | blocks):
        if el.name in headings:
            sections.append((clean(el.get_text()), []))
            continue
        if el.find_parent(list(blocks)):        # already captured by an enclosing block
            continue
        if el.name == "tr":
            text = " | ".join(t for t in (clean(c.get_text()) for c in el.find_all(["th", "td"])) if t)
        else:
            text = clean(el.get_text(" "))
        if text:
            sections[-1][1].append("- " + text if el.name == "li" else text)
    return title, note, sections


def split_long(paragraphs: list[str], limit: int) -> list[list[str]]:
    """Group paragraphs into parts of at most `limit` words without splitting a paragraph
    or separating a list from the sentence that introduces it."""
    parts: list[list[str]] = [[]]
    words = 0
    for p in paragraphs:
        n = len(p.split())
        if parts[-1] and words + n > limit and not p.startswith("- "):
            parts.append([])
            words = 0
        parts[-1].append(p)
        words += n
    return [p for p in parts if p]


def chunk_document(path: Path) -> list[Chunk]:
    title, note, sections = parse_html(path)
    chunks: list[Chunk] = []
    for heading, paragraphs in sections:
        if not paragraphs:
            continue
        parts = split_long(paragraphs, MAX_CHUNK_WORDS)
        for n, part in enumerate(parts, start=1):
            h = heading if len(parts) == 1 else f"{heading} (part {n} of {len(parts)})"
            chunks.append(Chunk(
                id=f"{path.name}::{len(chunks):03d}",
                text=f"{title} > {h}\n\n" + "\n".join(part),
                metadata={"source_file": path.name, "doc_title": title, "section": h, "doc_note": note},
            ))
    return chunks


# ---------------------------------------------------------------------------
# Chroma index
# ---------------------------------------------------------------------------

def policy_files() -> list[Path]:
    return sorted(p for p in DOCS_DIR.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def fingerprint(files: list[Path]) -> str:
    """Hash of every document's name, size and modification time, plus the chunker version."""
    h = hashlib.sha256(CHUNKER_VERSION.encode())
    for p in files:
        st = p.stat()
        h.update(f"{p.relative_to(DOCS_DIR)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def open_db():
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))
    return client, embedding_functions.DefaultEmbeddingFunction()


def build_index(client, embed, files: list[Path], fp: str):
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass                                    # nothing to delete yet
    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embed,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"fingerprint": fp},
    )
    all_chunks: list[Chunk] = []
    for path in files:
        try:
            chunks = chunk_document(path)
        except Exception as e:                  # one bad file shouldn't stop the index
            log(f"  Could not read {path.name}: {e}")
            continue
        log(f"  {path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    if all_chunks:
        collection.add(
            ids=[c.id for c in all_chunks],
            documents=[c.text for c in all_chunks],
            metadatas=[c.metadata for c in all_chunks],
        )
    log(f"Indexed {len(all_chunks)} chunks from {len(files)} files into chroma_db/{DB_DIR.name}/.")
    return collection


def get_collection(force_rebuild: bool = False):
    if not DOCS_DIR.is_dir():
        raise RuntimeError(f"Policy folder not found: {shown_path(DOCS_DIR)}")
    files = policy_files()
    fp = fingerprint(files)
    client, embed = open_db()
    if not force_rebuild:
        try:
            collection = client.get_collection(COLLECTION, embedding_function=embed)
            if (collection.metadata or {}).get("fingerprint") == fp:
                return collection
            log("Policy documents changed since the last index; rebuilding...")
        except Exception:
            log("No index yet; building it (the first run downloads Chroma's embedding model)...")
    else:
        log("Rebuilding the policy index...")
    return build_index(client, embed, files, fp)


_collection = None


def collection_for_search():
    """Open the index on first use (and re-check it if the documents changed since)."""
    global _collection
    if _collection is None or (_collection.metadata or {}).get("fingerprint") != fingerprint(policy_files()):
        _collection = get_collection()
    return _collection


# ---------------------------------------------------------------------------
# Employee records
# ---------------------------------------------------------------------------

# Column names differ between organizations (Hartwell: "email", "vacation_days_left";
# Ridgeline: "Email Address", "PTO Days Remaining"), so the columns are worked out
# from the header row, checked against the values underneath.

EMAIL_RE = re.compile(r"[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+")
NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")
BALANCE_WORDS = {"left", "remaining", "balance", "available", "accrued", "remain"}
LEAVE_WORDS = {"vacation", "sick", "personal", "pto", "leave", "holiday", "holidays", "floating", "days", "hours"}


@dataclass
class EmployeeColumns:
    email: str
    name: str | None
    first_name: str | None
    last_name: str | None
    balances: list[str]           # header names, in file order


def header_words(header: str) -> set[str]:
    """'PTO Days Remaining' / 'vacation_days_left' -> {'pto','days','remaining'} / {...}."""
    return set(re.sub(r"[^a-z0-9]+", " ", header.lower()).split())


def balance_label(header: str) -> str:
    """Human label: 'vacation_days_left' -> 'Vacation days left'; 'PTO Days Remaining' stays."""
    text = re.sub(r"[_\s]+", " ", header).strip()
    return text[:1].upper() + text[1:] if text.islower() else text


def detect_columns(headers: list[str], rows: list[dict]) -> EmployeeColumns:
    """Find the email, name, and balance columns. Raises ValueError if it can't."""
    def values(h):
        return [(r.get(h) or "").strip() for r in rows if (r.get(h) or "").strip()]

    # Email: a header mentioning email/mail whose values look like addresses;
    # otherwise any column where every value is an address.
    looks_like_email = [h for h in headers if values(h) and all(EMAIL_RE.fullmatch(v) for v in values(h))]
    named_email = [h for h in looks_like_email if header_words(h) & {"email", "mail", "e"}]
    candidates = named_email or looks_like_email
    if len(candidates) > 1:                                        # e.g. "Work Email" and "Personal Email"
        candidates = [h for h in candidates if "work" in header_words(h)] or candidates
        candidates = [h for h in candidates if "personal" not in header_words(h)] or candidates
    if not candidates:
        raise ValueError("no column of email addresses found in the header row")
    email = candidates[0]

    # Balances: numeric columns whose header says it's a leave amount
    # ("left"/"remaining"/... or a leave type like "vacation"/"pto").
    balances = []
    for h in headers:
        words = header_words(h)
        if h == email or not (words & BALANCE_WORDS or words & (LEAVE_WORDS - {"days", "hours"})):
            continue
        vals = values(h)
        if vals and all(NUMBER_RE.fullmatch(v) for v in vals):
            balances.append(h)
    if not balances:
        raise ValueError("no leave-balance columns found (numeric columns named like "
                         "'... days left', '... remaining', 'PTO ...')")

    # Name: a full-name column, or first + last name columns.
    def find(pred):
        return next((h for h in headers if h != email and pred(header_words(h))), None)
    name = find(lambda w: "name" in w and not w & {"first", "last", "given", "family", "sur", "middle", "user",
                                                   "supervisor", "manager", "hotel", "department"})
    first = find(lambda w: w & {"first", "given"} and "name" in w)
    last = find(lambda w: w & {"last", "family", "surname"} and ("name" in w or "surname" in w))
    return EmployeeColumns(email=email, name=name, first_name=first, last_name=last, balances=balances)


def load_employees() -> tuple[EmployeeColumns, list[dict]]:
    """Read employees.csv fresh on each call, so edits to the file take effect immediately."""
    with EMPLOYEES_CSV.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = [h for h in (reader.fieldnames or []) if h]
    try:
        return detect_columns(headers, rows), rows
    except ValueError as e:
        raise ValueError(f"{EMPLOYEES_CSV.name} in {DATA_DIR.name}: {e}") from None


def normalize_email(email: str) -> str:
    return email.strip().strip("<>").strip().lower()


class DuplicateRecordError(Exception):
    """More than one row has this email address, so we can't tell which balance is right."""


def find_employee(email: str) -> tuple[EmployeeColumns, dict] | None:
    """Exact (case-insensitive) match on the email column only. Never a partial or fuzzy match."""
    wanted = normalize_email(email)
    if not wanted:
        return None
    cols, rows = load_employees()
    matches = [row for row in rows if normalize_email(row.get(cols.email) or "") == wanted]
    if len(matches) > 1:
        raise DuplicateRecordError(f"{len(matches)} rows in {EMPLOYEES_CSV.name} share this address")
    return (cols, matches[0]) if matches else None


def no_record_error(email: str) -> ToolError:
    return ToolError(
        f"No employee record found for this address ({email}). Do NOT state, estimate, or guess "
        "any leave balance for this person. Say that their balance couldn't be found and that "
        "someone from HR will check it."
    )


def format_balance(found: tuple[EmployeeColumns, dict]) -> str:
    cols, row = found
    if cols.name:
        name = (row.get(cols.name) or "").strip()
    else:
        name = " ".join((row.get(c) or "").strip() for c in (cols.first_name, cols.last_name) if c).strip()
    who = f"{name} ({row[cols.email].strip()})" if name else row[cols.email].strip()
    values = {h: (row.get(h) or "").strip() for h in cols.balances}
    if not any(values.values()):
        raise ToolError(f"The employee record for {row[cols.email].strip()} has no balances filled in. "
                        "Do NOT state, estimate, or guess any balance; say someone from HR will check it.")
    lines = [f"Leave balance for {who}:"]
    for h, v in values.items():
        # A blank cell is reported as missing, never shown as an empty value Claude might fill in.
        lines.append(f"- {balance_label(h)}: {v}" if v else
                     f"- {balance_label(h)}: NOT ON RECORD (do not state or estimate this balance)")
    lines.append(f"Source: {EMPLOYEES_CSV.name}")
    return "\n".join(lines)


def balance_kinds() -> str:
    """For tool descriptions, e.g. 'PTO Days Remaining, Sick Days Remaining'."""
    try:
        return ", ".join(balance_label(h) for h in load_employees()[0].balances)
    except (OSError, ValueError):
        return "leave balances"


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

def search_policy(query: str) -> str:
    """Search the organization's HR policy documents and return the 4 closest passages,
    each labeled with its source file. Use this for any question about HR policy
    (time off, leave, benefits, scheduling, reporting concerns, and so on). Cite the
    source file for each policy fact you use."""
    query = query.strip()
    if not query:
        raise ToolError("The search query is empty. Pass a short description of the policy question.")
    try:
        collection = collection_for_search()
    except Exception as e:
        raise ToolError(f"The policy index is unavailable ({e}). Do not answer policy questions from memory.")
    res = collection.query(query_texts=[query], n_results=TOP_K,
                           include=["documents", "metadatas", "distances"])
    hits = zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    out = [f"Top {TOP_K} policy passages for: {query}"]
    for rank, (text, meta, dist) in enumerate(hits, start=1):
        note = f"\nDocument note: {meta['doc_note']}" if meta.get("doc_note") else ""
        out.append(
            f"--- Result {rank} | source: {meta['source_file']} | similarity {1.0 - dist:.3f}{note}\n{text}"
        )
    return "\n\n".join(out)


def lookup(email: str) -> str:
    try:
        found = find_employee(email)
    except DuplicateRecordError as e:
        raise ToolError(f"More than one employee record matches this address ({e}). Do NOT state, "
                        "estimate, or guess any balance; say someone from HR will check it.")
    except (OSError, ValueError) as e:
        raise ToolError(f"The employee file can't be read ({e}). Do not state any balance.")
    if found is None:
        raise no_record_error(email)
    return format_balance(found)


def build_server(as_employee: str | None) -> MCPServer:
    org = DATA_DIR.name
    kinds = balance_kinds()
    if as_employee:
        instructions = (
            f"HR tools for {org}, employee mode. get_leave_balance returns only the balance of "
            f"{as_employee}; it cannot look up any other employee."
        )
    else:
        instructions = f"HR tools for {org}, for HR staff: policy search and leave balances by work email."
    server = MCPServer(name=f"{org}-hr", instructions=instructions)
    server.add_tool(search_policy, name="search_policy")

    if as_employee:
        def get_leave_balance() -> str:
            return lookup(as_employee)

        server.add_tool(
            get_leave_balance,
            name="get_leave_balance",
            description=(
                f"Return the leave balance ({kinds}) of the employee this session belongs to. "
                "Takes no inputs and cannot look up anyone else. If it returns an error, do not "
                "state any balance."
            ),
        )
    else:
        def get_leave_balance(email: str) -> str:
            if not normalize_email(email):
                raise ToolError("Pass the employee's work email address.")
            return lookup(email)

        server.add_tool(
            get_leave_balance,
            name="get_leave_balance",
            description=(
                f"For HR staff: return an employee's leave balance ({kinds}), looked up by their "
                "work email address. If it returns an error, do not state any balance."
            ),
        )
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="HR MCP server (stdio) for one organization folder.")
    parser.add_argument("--org", metavar="FOLDER",
                        help="organization folder holding hr-docs, employees.csv, sample-emails, "
                             "system_prompt.txt (default: week-04/hartwell-hr)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--as-employee", metavar="EMAIL",
                      help="employee mode: get_leave_balance takes no inputs and returns only this employee's balance")
    mode.add_argument("--hr-staff", action="store_true",
                      help="HR staff mode: get_leave_balance looks up ANY employee by address. Only for HR "
                           "staff's own tools; never for answering employee email")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the organization's Chroma policy index in week-04/chroma_db/<org>/, then exit")
    args = parser.parse_args()
    if not (args.rebuild or args.as_employee is not None or args.hr_staff):
        # The any-address mode must be asked for by name; it is never the default.
        parser.error("choose a mode: --as-employee EMAIL (one employee's balance only) or --hr-staff")

    try:
        use_org(resolve_org(args.org))
        cols, _ = load_employees()
    except (ValueError, OSError) as e:
        parser.error(str(e))
    log(f"Organization: {DATA_DIR.name}. Employee file columns: email = '{cols.email}', "
        f"balances = {', '.join(repr(h) for h in cols.balances)}.")

    if args.rebuild:
        get_collection(force_rebuild=True)
        return

    as_employee = normalize_email(args.as_employee) if args.as_employee is not None else None
    if args.as_employee is not None and not as_employee:
        parser.error("--as-employee needs an email address")
    try:
        if as_employee and find_employee(as_employee) is None:
            log(f"Warning: no employee record found for {as_employee}; get_leave_balance will return an error.")
    except DuplicateRecordError as e:
        log(f"Warning: {e}; get_leave_balance will return an error.")

    build_server(as_employee).run("stdio")


if __name__ == "__main__":
    main()
