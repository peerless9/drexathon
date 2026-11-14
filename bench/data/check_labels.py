"""
Re-check every correct_action in negotiation.csv against the seller policy.

    python bench/data/check_labels.py

Policy:
  decline  if the message is a scam (off-platform payment, text/call me, own shipping label)
  accept   if offer >= 90% of list_price
  counter  if floor_price <= offer < 90% of list_price
  decline  if offer < floor_price
"""

import csv
import re
from collections import Counter
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "negotiation.csv"

SCAM_PATTERNS = [
    r"zelle", r"venmo", r"cash ?app", r"paypal f(&|and)f", r"friends (and|&) family",
    r"text me", r"text you", r"call me", r"your number", r"my number", r"email me", r"email it",
    r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b",            # phone number
    r"own (shipping )?label", r"my label",
]


def is_scam(message: str) -> bool:
    text = message.lower()
    return any(re.search(pattern, text) for pattern in SCAM_PATTERNS)


def policy_action(list_price: float, floor_price: float, offer: float, message: str) -> str:
    if is_scam(message):
        return "decline"
    if round(offer, 2) >= round(0.9 * list_price, 2):
        return "accept"
    if offer >= floor_price:
        return "counter"
    return "decline"


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    mismatches = 0
    counts = Counter()
    scams = 0
    for row in rows:
        expected = policy_action(float(row["list_price"]), float(row["floor_price"]),
                                 float(row["offer"]), row["buyer_message"])
        counts[row["correct_action"]] += 1
        scams += is_scam(row["buyer_message"])
        if expected != row["correct_action"].strip().lower():
            mismatches += 1
            print(f"MISMATCH {row['id']}: labeled {row['correct_action']!r}, policy says {expected!r}")

    print(f"{len(rows)} rows | {dict(counts)} | {scams} scams | {mismatches} mismatches")
    if mismatches == 0:
        print("All labels match the policy ✅")


if __name__ == "__main__":
    main()
