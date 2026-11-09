"""Test offline scoring independently of provider behavior."""

import json
from pathlib import Path

import pytest

from evals.adversarial import canonical_source_hash, detection_scores, evaluate_cases, load_cases, main


DATASET_PATH = Path(__file__).resolve().parents[1] / "evals" / "adversarial.jsonl"


def test_dataset_covers_attacks_and_benign_quoted_tickets():
    cases = load_cases(DATASET_PATH)
    assert sum(case["expected_block"] for case in cases) >= 10
    assert {"prompt_injection", "role_spoofing", "output_manipulation", "benign_quote"} <= {case["attack_type"] for case in cases}
    assert sum(not case["expected_block"] for case in cases) >= 2


def test_confusion_matrix_and_rates():
    results = [
        {"expected_block": True, "blocked": True},
        {"expected_block": True, "blocked": False},
        {"expected_block": False, "blocked": True},
        {"expected_block": False, "blocked": False},
    ]
    scores = detection_scores(results)
    assert all(scores[key] == 1 for key in ("true_positives", "false_negatives", "false_positives", "true_negatives"))
    assert scores["attack_detection_rate"] == scores["false_positive_rate"] == 0.5
    assert scores["false_negative_rate"] == scores["precision"] == scores["accuracy"] == 0.5
    assert detection_scores([])["attack_detection_rate"] is None


def test_actual_guardrail_report_exposes_known_failures():
    report = evaluate_cases(load_cases(DATASET_PATH))
    assert report["mode"] == "offline_input_guardrail"
    assert report["scores"]["false_negatives"] > 0
    assert report["scores"]["false_positives"] > 0
    assert len(report["dataset_hash"]) == len(report["guardrail_hash"]) == 64
    assert sum(row["total"] for row in report["per_attack_type"].values()) == report["scores"]["total"]


def test_saved_evidence_matches_current_offline_measurement():
    path = DATASET_PATH.parents[1] / "docs" / "evidence" / "adversarial_report.json"
    assert json.loads(path.read_text(encoding="utf-8")) == evaluate_cases(load_cases(DATASET_PATH))


def test_source_hash_is_independent_of_line_endings():
    source = "# Código de prueba\nvalue = 1\n".encode("utf-8")
    assert canonical_source_hash(source) == canonical_source_hash(source.replace(b"\n", b"\r\n"))
    assert canonical_source_hash(source) == canonical_source_hash(source.replace(b"\n", b"\r"))
    assert canonical_source_hash(source) != canonical_source_hash(source.replace(b"1", b"2"))


def test_complete_report_matches_for_lf_and_crlf_source(monkeypatch):
    from src import guardrails

    source_path = Path(guardrails.__file__)
    source = source_path.read_bytes().replace(b"\r\n", b"\n")
    original_read_bytes = Path.read_bytes
    reports = []
    for ending in (b"\n", b"\r\n"):
        content = source.replace(b"\n", ending)
        monkeypatch.setattr(Path, "read_bytes", lambda path: content if path == source_path else original_read_bytes(path))
        reports.append(evaluate_cases(load_cases(DATASET_PATH)))
    assert reports[0] == reports[1]
    assert reports[0]["guardrail_hash_scheme"] == "sha256-utf8-lf-v1"


def test_input_detection_scores_remain_unchanged():
    scores = evaluate_cases(load_cases(DATASET_PATH))["scores"]
    assert {key: scores[key] for key in ("true_positives", "false_negatives", "false_positives", "true_negatives")} == {
        "true_positives": 5, "false_negatives": 7, "false_positives": 2, "true_negatives": 2,
    }


def test_rejects_duplicate_case_ids(tmp_path):
    case = load_cases(DATASET_PATH)[0]
    path = tmp_path / "cases.jsonl"
    path.write_text((json.dumps(case) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicado"):
        load_cases(path)


def test_cli_is_offline_and_refuses_overwriting(tmp_path, monkeypatch, capsys):
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["adversarial", "--dataset", str(DATASET_PATH), "--output", str(output)])
    main()
    assert json.loads(capsys.readouterr().out)["scores"]["total"] >= 10
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        main()
    assert output.read_bytes() == original
