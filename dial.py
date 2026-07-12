#!/usr/bin/env python3
"""
Outbound dialer — pick a contact from a CSV and have the phonebot call them.

Reads a contacts CSV (default contacts.csv), shows the list, places the call
through the server's /outbound-call endpoint (the answered call streams into
the same bot pipeline as an inbound call), then live-tracks the Twilio call
status. A ringing or in-progress call can be terminated with [h].

Before dialing you pick the end-of-turn / interruption mode ('smart' semantic
turn detection, the server default, or 'vad' fixed-silence); it's sent per call
and changeable from the menu with 'm'.

CSV columns: phone (E.164, required), doctor_name, patient_name, patient_dob,
patient_phone, insurance, notes. doctor_name replaces "Dr. Smith" and
patient_name replaces "John Doe" in that call's prompt and greeting; the other
patient fields are passed as background context the bot only shares if the
office explicitly asks.

Usage:
    python dial.py                    # use contacts.csv
    python dial.py mylist.csv
    python dial.py --server http://localhost:8000

Config: MONITOR_TOKEN from .env (must match the server's), server URL from
--server / PHONEBOT_SERVER env (default https://phonebotpearai.online).

Requirements (local only, not in requirements.txt):
    pip install httpx python-dotenv
"""

import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import httpx
except ImportError:
    sys.exit("Missing dependency. Run: pip install httpx")

try:
    import msvcrt  # Windows hotkeys
except ImportError:
    msvcrt = None

DEFAULT_SERVER = os.getenv("PHONEBOT_SERVER", "https://phonebotpearai.online")
TERMINAL_STATUSES = {"completed", "busy", "failed", "no-answer", "canceled"}
POLL_SECS = 2.0

# End-of-turn / interruption strategy sent to the server per call (see
# build_turn_detection on the server). "smart" is the server default.
TURN_MODES = {
    "smart": "smart-turn v3 (semantic end-of-turn; tolerates natural pauses)",
    "vad": "VAD only (fixed 0.8s silence gap; snappier, may cut off pauses)",
}


def _headers(token: str) -> dict:
    return {"X-Monitor-Token": token}


# ── contacts ──────────────────────────────────────────────────────────────────

def load_contacts(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} not found — create it (see contacts.example.csv).")
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = [
            {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(f)
        ]
    contacts = [r for r in rows if r.get("phone")]
    if not contacts:
        sys.exit(f"No rows with a 'phone' value in {path}.")
    return contacts


def show_menu(contacts: list[dict], path: Path, turn_mode: str) -> None:
    print(f"\nContacts ({path})   [interruption mode: {turn_mode}]:")
    for i, c in enumerate(contacts, 1):
        doctor = f"Dr. {c['doctor_name']}" if c.get("doctor_name") else "(no doctor name)"
        patient = c.get("patient_name") or "-"
        print(f"  [{i}] {c['phone']:<16} {doctor:<26} patient: {patient}")


def choose_turn_mode(current: str) -> str:
    """Prompt for the end-of-turn / interruption strategy; Enter keeps current."""
    keys = list(TURN_MODES)
    print("\nInterruption / end-of-turn mode:")
    for i, k in enumerate(keys, 1):
        marker = "*" if k == current else " "
        print(f" {marker}[{i}] {k:<6} {TURN_MODES[k]}")
    choice = input(f"Select [1-{len(keys)}], Enter to keep '{current}': ").strip()
    if not choice:
        return current
    if choice.isdigit() and 1 <= int(choice) <= len(keys):
        return keys[int(choice) - 1]
    print("  (unrecognized — keeping current)")
    return current


def build_call_context(c: dict) -> str:
    parts = []
    if c.get("patient_name"):
        parts.append(f"Patient name: {c['patient_name']}")
    if c.get("patient_dob"):
        parts.append(f"Patient date of birth: {c['patient_dob']}")
    if c.get("patient_phone"):
        parts.append(f"Patient callback number: {c['patient_phone']}")
    if c.get("insurance"):
        parts.append(f"Insurance: {c['insurance']}")
    if c.get("notes"):
        parts.append(f"Notes: {c['notes']}")
    return "\n".join(parts)


# ── server API ────────────────────────────────────────────────────────────────

async def place_call(client: httpx.AsyncClient, server: str, token: str,
                     contact: dict, turn_mode: str) -> str | None:
    payload = {
        "to": contact["phone"],
        "doctor_name": contact.get("doctor_name", ""),
        "patient_name": contact.get("patient_name", ""),
        "call_context": build_call_context(contact),
        "turn_mode": turn_mode,
    }
    r = await client.post(f"{server}/outbound-call", json=payload, headers=_headers(token))
    if r.status_code == 403:
        sys.exit("Server rejected MONITOR_TOKEN (403) — check .env on both ends.")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        print(f"  Call failed ({r.status_code}): {detail}")
        return None
    data = r.json()
    print(f"  Dialing {data['to']} from {data['from']}  (sid {data['call_sid']})")
    return data["call_sid"]


async def hangup(client: httpx.AsyncClient, server: str, token: str, call_sid: str) -> None:
    r = await client.post(f"{server}/calls/{call_sid}/hangup", headers=_headers(token))
    if r.status_code == 200:
        print("\n  Hangup sent — waiting for Twilio to confirm…")
    else:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        print(f"\n  Hangup failed ({r.status_code}): {detail}")


async def fetch_status(client: httpx.AsyncClient, server: str, token: str, call_sid: str) -> str:
    try:
        r = await client.get(f"{server}/calls/{call_sid}/status", headers=_headers(token))
    except httpx.HTTPError:
        return "unreachable"
    if r.status_code != 200:
        return "unknown"
    return r.json().get("status", "unknown")


# ── call watcher ──────────────────────────────────────────────────────────────

async def watch_call(client: httpx.AsyncClient, server: str, token: str, call_sid: str) -> None:
    print(f"  Tip: run  python listen.py {call_sid}  in another terminal to hear the call.")
    if msvcrt:
        print("  [h] hang up   [q] back to menu (call continues)")
    else:
        print("  (hotkeys unavailable on this platform — Ctrl+C to quit)")
    started = time.monotonic()
    status = "queued"
    next_poll = 0.0
    while True:
        now = time.monotonic()
        if now >= next_poll:
            status = await fetch_status(client, server, token, call_sid)
            next_poll = now + POLL_SECS
            if status in TERMINAL_STATUSES:
                elapsed = int(now - started)
                print(f"\r  Call ended: {status}  ({elapsed // 60:02d}:{elapsed % 60:02d})"
                      + " " * 30)
                return
        if msvcrt:
            while msvcrt.kbhit():
                ch = msvcrt.getwch().lower()
                if ch == "h":
                    await hangup(client, server, token, call_sid)
                    next_poll = 0.0  # re-check status right away
                elif ch == "q":
                    print("\n  Back to menu — the call continues on the server.")
                    return
        elapsed = int(time.monotonic() - started)
        print(f"\r  {status:<12} {elapsed // 60:02d}:{elapsed % 60:02d}  ", end="", flush=True)
        await asyncio.sleep(0.1)


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="Place bot calls to contacts from a CSV.")
    parser.add_argument("csv", nargs="?", default="contacts.csv", help="contacts CSV (default contacts.csv)")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"server base URL (default {DEFAULT_SERVER})")
    args = parser.parse_args()

    server = args.server.rstrip("/")
    token = os.getenv("MONITOR_TOKEN", "")
    if not token:
        sys.exit("MONITOR_TOKEN is not set in .env — it must match the server's.")

    path = Path(args.csv)
    contacts = load_contacts(path)

    # Session default; picked once up front, changeable from the menu with 'm'.
    turn_mode = choose_turn_mode("smart")

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            show_menu(contacts, path, turn_mode)
            choice = input(
                f"Call [1-{len(contacts)}], m to change mode, r to reload, q to quit: "
            ).strip().lower()
            if choice == "q":
                return
            if choice == "m":
                turn_mode = choose_turn_mode(turn_mode)
                continue
            if choice == "r":
                contacts = load_contacts(path)
                continue
            if not (choice.isdigit() and 1 <= int(choice) <= len(contacts)):
                continue
            contact = contacts[int(choice) - 1]
            doctor = f"Dr. {contact['doctor_name']}" if contact.get("doctor_name") else "the default doctor"
            confirm = input(
                f"Call {contact['phone']} about {doctor} [mode: {turn_mode}]? [y/N]: "
            ).strip().lower()
            if confirm != "y":
                continue
            call_sid = await place_call(client, server, token, contact, turn_mode)
            if call_sid:
                await watch_call(client, server, token, call_sid)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye.")
