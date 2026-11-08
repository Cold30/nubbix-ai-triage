"""Protect the expanded classification dataset and the original labeled records."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from evals.runner import load_dataset


DATASET_PATH = Path(__file__).resolve().parents[1] / "evals" / "dataset.jsonl"
EXPECTED_DISTRIBUTION = {"billing": 13, "technical": 13, "account": 12, "other": 12}
ORIGINAL_FINGERPRINT = "3dfaf24ea0231cf4a0cb475ef7a801629006db50ee171c0978024a7adce41e67"


def test_dataset_has_fifty_unique_sequential_ticket_ids():
    tickets = load_dataset(DATASET_PATH)
    assert len(tickets) == 50
    assert [ticket["id"] for ticket in tickets] == [f"T{index:03d}" for index in range(1, 51)]


def test_dataset_category_distribution():
    tickets = load_dataset(DATASET_PATH)
    assert Counter(ticket["expected_category"] for ticket in tickets) == EXPECTED_DISTRIBUTION
    assert Counter(ticket["expected_category"] for ticket in tickets[30:]) == {
        category: 5 for category in EXPECTED_DISTRIBUTION
    }


def test_original_thirty_tickets_are_unchanged():
    tickets = load_dataset(DATASET_PATH)
    serialized = json.dumps(
        tickets[:30], sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(serialized).hexdigest() == ORIGINAL_FINGERPRINT


def test_dataset_has_no_normalized_duplicate_queries():
    normalized = []
    for ticket in load_dataset(DATASET_PATH):
        decomposed = unicodedata.normalize("NFKD", ticket["query"].casefold())
        unaccented = "".join(character for character in decomposed if not unicodedata.combining(character))
        normalized.append(" ".join(re.findall(r"\w+", unaccented)))
    assert len(set(normalized)) == 50
