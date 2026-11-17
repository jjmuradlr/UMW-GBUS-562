"""Load the sample HR emails into a Gmail sandbox inbox over IMAP.

Each .eml file in hartwell-hr/sample-emails is placed directly into the INBOX
as an unread message dated today (IMAP APPEND). Nothing is sent.

The Gmail login is read at run time from Agent-Email.txt in the folder above
the repository. Accepted formats:

    EMAIL_ADDRESS=someone@gmail.com   (or email= / user= / address=)
    APP_PASSWORD=abcdefghijklmnop     (or password= / pass=)

or simply the address on the first line and the password on the second.
Gmail needs an app password (Google Account > Security > App passwords).

The password is never printed or written anywhere.
"""

import email
import email.policy
import email.utils
import imaplib
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
LOGIN_FILE = REPO_ROOT.parent / "Agent-Email.txt"
EMAIL_DIR = HERE / "hartwell-hr" / "sample-emails"
IMAP_HOST = "imap.gmail.com"
MAILBOX = "INBOX"

USER_KEYS = {"email_address", "email", "user", "username", "address", "login", "account"}
PASS_KEYS = {"password", "pass", "app_password", "apppassword", "app password"}


def load_login():
    """Return (address, password) from the login file without echoing either."""
    if not LOGIN_FILE.exists():
        sys.exit(f"Login file not found: {LOGIN_FILE}")
    lines = [ln.strip() for ln in LOGIN_FILE.read_text(encoding="utf-8-sig").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]

    user = password = None
    for ln in lines:
        m = re.match(r"^([A-Za-z _-]+?)\s*[:=]\s*(.+)$", ln)
        if not m:
            continue
        key = m.group(1).strip().lower().replace("-", "_")
        value = m.group(2).strip().strip('"').strip("'")
        if key in USER_KEYS:
            user = value
        elif key in PASS_KEYS or key.replace("_", " ") in PASS_KEYS:
            password = value

    if user is None and password is None and len(lines) >= 2:
        user, password = lines[0], lines[1]

    if not user or not password:
        sys.exit(f"Could not find an address and password in {LOGIN_FILE.name}. "
                 "See the format notes at the top of load_inbox.py.")

    # Gmail shows app passwords as four groups of four letters; spaces are optional.
    if re.fullmatch(r"[a-zA-Z]{4}( [a-zA-Z]{4}){3}", password):
        password = password.replace(" ", "")
    return user, password


def prepare(path):
    """Read an .eml file and stamp it with today's date and a Message-ID."""
    msg = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)
    del msg["Date"]
    msg["Date"] = email.utils.formatdate(localtime=True)
    if msg["Message-ID"] is None:
        msg["Message-ID"] = email.utils.make_msgid(domain="hartwellfoods.example")
    return msg.as_bytes(policy=email.policy.SMTP)


def main():
    files = sorted(EMAIL_DIR.glob("*.eml"))
    if not files:
        sys.exit(f"No .eml files found in {EMAIL_DIR}")

    user, password = load_login()
    try:
        imap = imaplib.IMAP4_SSL(IMAP_HOST)
        imap.login(user, password)
    except imaplib.IMAP4.error:
        sys.exit("IMAP login failed. Check the address and app password in "
                 f"{LOGIN_FILE.name}, and that IMAP is enabled in Gmail settings.")
    finally:
        password = None

    try:
        for path in files:
            # Empty flag list = no \Seen flag, so the message arrives unread.
            status, _ = imap.append(MAILBOX, "", imaplib.Time2Internaldate(time.time()),
                                    prepare(path))
            print(f"{'Added' if status == 'OK' else 'FAILED'}: {path.name}")
    finally:
        imap.logout()


if __name__ == "__main__":
    main()
