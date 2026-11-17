"""hr_mcp_server.py - an MCP server with Hartwell Foods HR tools.

Built with the official MCP Python SDK (`mcp` 2.x, `MCPServer`). It speaks MCP
over stdio, so Claude Code (or any MCP client) starts it as a subprocess.

Tools
-----
search_policy(query)
    Searches the HR policy documents in hartwell-hr/hr-docs/ and returns the
    4 closest chunks, each with its source file. The documents are split into
    one chunk per section and stored in a Chroma database in week-04/chroma_db/
    (ignored by Git), embedded with Chroma's built-in model (all-MiniLM-L6-v2).

get_leave_balance
    Reads leave balances from hartwell-hr/employees.csv. Two modes:
    - HR staff mode (default): takes an employee's work email.
    - Employee mode (--as-employee ADDRESS): takes no inputs and returns only
      that one employee's balance. There is no way to ask for anyone else's.
    If no record matches, the tool returns an error telling Claude not to state
    a balance.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py                      # HR staff mode
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --as-employee dana.whitfield@hartwellfoods.example
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --rebuild            # rebuild the Chroma index, then exit

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
DATA_DIR = HERE / "hartwell-hr"
DOCS_DIR = DATA_DIR / "hr-docs"
EMPLOYEES_CSV = DATA_DIR / "employees.csv"
DB_DIR = HERE / "chroma_db"                         # Git-ignored (".gitignore": chroma_db/)
COLLECTION = "hr_policies"

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
    log(f"Indexed {len(all_chunks)} chunks from {len(files)} files into {DB_DIR.name}/.")
    return collection


def get_collection(force_rebuild: bool = False):
    if not DOCS_DIR.is_dir():
        raise RuntimeError(f"Policy folder not found: {DOCS_DIR}")
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

def normalize_email(email: str) -> str:
    return email.strip().strip("<>").strip().lower()


def find_employee(email: str) -> dict | None:
    """Read employees.csv fresh on each call, so edits to the file take effect immediately."""
    wanted = normalize_email(email)
    if not wanted:
        return None
    with EMPLOYEES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if normalize_email(row.get("email", "")) == wanted:
                return row
    return None


def no_record_error(email: str) -> ToolError:
    return ToolError(
        f"No employee record matches '{email}'. Do NOT state, estimate, or guess any leave "
        "balance (vacation, sick, or personal days) for this person. Say that their balance "
        "couldn't be found and that someone from HR will check it."
    )


def format_balance(row: dict) -> str:
    return (
        f"Leave balance for {row['name']} ({row['email']}), {row['department']}:\n"
        f"- Vacation days left: {row['vacation_days_left']}\n"
        f"- Sick days left: {row['sick_days_left']}\n"
        f"- Personal days left: {row['personal_days_left']}\n"
        f"Source: employees.csv"
    )


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

def search_policy(query: str) -> str:
    """Search Hartwell Foods' HR policy documents and return the 4 closest passages,
    each labeled with its source file. Use this for any question about HR policy
    (vacation, sick days, leave, benefits, remote work, reporting concerns). Cite the
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


def build_server(as_employee: str | None) -> MCPServer:
    if as_employee:
        instructions = (
            f"Hartwell Foods HR tools, employee mode. get_leave_balance returns only the balance of "
            f"{as_employee}; it cannot look up any other employee."
        )
    else:
        instructions = "Hartwell Foods HR tools for HR staff: policy search and leave balances by work email."
    server = MCPServer(name="hartwell-hr", instructions=instructions)
    server.add_tool(search_policy, name="search_policy")

    if as_employee:
        def get_leave_balance() -> str:
            row = find_employee(as_employee)
            if row is None:
                raise no_record_error(as_employee)
            return format_balance(row)

        server.add_tool(
            get_leave_balance,
            name="get_leave_balance",
            description=(
                "Return the leave balance (vacation, sick, and personal days left) of the employee "
                "this session belongs to. Takes no inputs and cannot look up anyone else. If it "
                "returns an error, do not state any balance."
            ),
        )
    else:
        def get_leave_balance(email: str) -> str:
            if not normalize_email(email):
                raise ToolError("Pass the employee's work email, e.g. name@hartwellfoods.example.")
            row = find_employee(email)
            if row is None:
                raise no_record_error(email)
            return format_balance(row)

        server.add_tool(
            get_leave_balance,
            name="get_leave_balance",
            description=(
                "For HR staff: return an employee's leave balance (vacation, sick, and personal "
                "days left), looked up by their work email address. If it returns an error, do not "
                "state any balance."
            ),
        )
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Hartwell Foods HR MCP server (stdio).")
    parser.add_argument("--as-employee", metavar="EMAIL",
                        help="employee mode: get_leave_balance takes no inputs and returns only this employee's balance")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the Chroma policy index in week-04/chroma_db/, then exit")
    args = parser.parse_args()

    if args.rebuild:
        get_collection(force_rebuild=True)
        return

    as_employee = normalize_email(args.as_employee) if args.as_employee is not None else None
    if args.as_employee is not None and not as_employee:
        parser.error("--as-employee needs an email address")
    if as_employee and find_employee(as_employee) is None:
        log(f"Warning: no employee record matches {as_employee}; get_leave_balance will return an error.")

    build_server(as_employee).run("stdio")


if __name__ == "__main__":
    main()
