"""Regression coverage for dataset integrity and atomic resume checkpoints."""

import json
from types import SimpleNamespace

import pytest

from evals.runner import (
    atomic_write_report, build_experiment_identity, load_dataset,
    run_evaluation, save_checkpoint, validate_dataset,
)
from src.configuration import GenerationSettings
from src.llm_client import PermanentLLMError


def dataset():
    return [{"id": "T1", "query": "No puedo ingresar.", "expected_category": "account"}]


def identity(rows=None, **kwargs):
    return build_experiment_identity(rows or dataset(), "triage_v1", "gemini", "test-model", **kwargs)


@pytest.mark.parametrize("row", [
    {}, {"id": 1, "query": "Consulta", "expected_category": "account"},
    {"id": " ", "query": "Consulta", "expected_category": "account"},
    {"id": "T1", "query": " ", "expected_category": "account"},
    {"id": "T1", "query": "Consulta", "expected_category": "invalid"},
    {"id": "T1", "query": "Consulta", "expected_category": "account", "extra": True},
    [], None,
])
def test_rejects_invalid_dataset_rows(row):
    with pytest.raises(ValueError, match="Ticket inválido"):
        validate_dataset([row])


def test_rejects_duplicate_ids_and_empty_dataset():
    with pytest.raises(ValueError, match="duplicado"):
        validate_dataset(dataset() * 2)
    with pytest.raises(ValueError, match="al menos"):
        validate_dataset([])


def test_dataset_json_errors_have_line_numbers_without_content(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text('\n{"private": broken}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="línea 2") as caught:
        load_dataset(path)
    assert "private" not in str(caught.value)


def test_loads_valid_dataset_with_blank_lines(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n" + json.dumps(dataset()[0]) + "\n\n", encoding="utf-8")
    assert load_dataset(path) == dataset()


def test_failed_atomic_replace_preserves_original_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "results.json"
    path.write_text('{"original":true}', encoding="utf-8")

    def fail_replace(*args):
        raise OSError("simulated interruption")

    monkeypatch.setattr("evals.runner.os.replace", fail_replace)
    with pytest.raises(OSError):
        atomic_write_report({"replacement": True}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"original": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_produces_complete_report(tmp_path):
    path = tmp_path / "nested" / "results.json"
    atomic_write_report({"answer": "Español", "results": []}, path)
    assert json.loads(path.read_text(encoding="utf-8"))["answer"] == "Español"


@pytest.mark.parametrize("settings", [
    GenerationSettings(0.2, 512, 30), GenerationSettings(0.1, 256, 30), GenerationSettings(0.1, 512, 10),
])
def test_resume_rejects_changed_generation_policy(tmp_path, settings):
    path = tmp_path / "results.json"
    save_checkpoint([], path, identity())
    service = SimpleNamespace(classify=lambda **kwargs: pytest.fail("Unexpected generation"))
    with pytest.raises(ValueError, match="configuración"):
        run_evaluation(service, dataset(), "triage_v1", path, identity(settings=settings), resume=True)


def test_validation_retry_count_changes_identity():
    assert identity(max_retries=0) != identity(max_retries=2)


@pytest.mark.parametrize("report", [
    "not json", "[]", '{}', '{"experiment":null,"results":[]}',
])
def test_rejects_corrupt_or_legacy_checkpoints(tmp_path, report):
    path = tmp_path / "checkpoint.json"
    path.write_text(report, encoding="utf-8")
    with pytest.raises(ValueError):
        run_evaluation(SimpleNamespace(), dataset(), "triage_v1", path, identity(), resume=True)


@pytest.mark.parametrize("change", [
    {"id": "unknown"}, {"expected_category": "billing"},
    {"predicted_category": []}, {"correct": False}, {"error": {}},
])
def test_rejects_inconsistent_checkpoint_rows(tmp_path, change):
    row = {"id": "T1", "expected_category": "account", "predicted_category": "account", "correct": True, "error": None}
    row.update(change)
    path = tmp_path / "checkpoint.json"
    save_checkpoint([row], path, identity())
    with pytest.raises(ValueError, match="resultados inválidos"):
        run_evaluation(SimpleNamespace(), dataset(), "triage_v1", path, identity(), resume=True)


def test_duplicate_checkpoint_ids_are_rejected(tmp_path):
    row = {"id": "T1", "expected_category": "account", "predicted_category": "account", "correct": True, "error": None}
    path = tmp_path / "checkpoint.json"
    save_checkpoint([row, row], path, identity())
    with pytest.raises(ValueError, match="duplicados"):
        run_evaluation(SimpleNamespace(), dataset(), "triage_v1", path, identity(), resume=True)


def test_provider_failure_preserves_completed_checkpoint(tmp_path):
    rows = dataset() + [{"id": "T2", "query": "No puedo acceder.", "expected_category": "account"}]
    calls = 0

    def classify(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermanentLLMError("Fallo simulado")
        return SimpleNamespace(category="account")

    path = tmp_path / "checkpoint.json"
    with pytest.raises(PermanentLLMError):
        run_evaluation(SimpleNamespace(classify=classify), rows, "triage_v1", path, identity(rows))
    assert [row["id"] for row in json.loads(path.read_text(encoding="utf-8"))["results"]] == ["T1"]
