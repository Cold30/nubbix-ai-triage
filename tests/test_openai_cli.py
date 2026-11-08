"""Exercise native OpenAI through the CLI and evaluation runner with mocked responses."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def configure_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6-luna")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "512")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.responses.create.return_value = SimpleNamespace(
        status="completed", error=None, model="gpt-6-luna", usage=None,
        output_text=json.dumps({"answer": "Un agente puede revisar el acceso.", "confidence": 0.9,
                                "category": "account", "actions": ["Revisar permisos"]}),
    )
    monkeypatch.setattr("src.openai_provider.OpenAI", lambda **kwargs: client)
    return client


def test_classification_cli_accepts_openai(tmp_path, monkeypatch, capsys):
    from src import main

    configure_openai(monkeypatch)
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setenv("METRICS_PATH", str(tmp_path / "metrics.jsonl"))
    monkeypatch.setattr("sys.argv", ["main", "classify", "No puedo acceder.", "--provider", "openai"])
    main.main()
    assert json.loads(capsys.readouterr().out)["category"] == "account"


def test_evaluation_cli_accepts_openai_and_resumes_only_compatible_effort(tmp_path, monkeypatch, capsys):
    from evals import runner

    client = configure_openai(monkeypatch)
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    monkeypatch.setenv("METRICS_PATH", str(tmp_path / "metrics.jsonl"))
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps({"id": "T1", "query": "No puedo acceder.", "expected_category": "account"}), encoding="utf-8")
    output = tmp_path / "report.json"
    arguments = ["runner", "--provider", "openai", "--prompt-version", "triage_v1", "--dataset", str(dataset_path),
                 "--output", str(output), "--delay-seconds", "0"]
    monkeypatch.setattr("sys.argv", arguments + ["--fresh"])
    runner.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["experiment"]["provider"] == "openai"
    assert report["experiment"]["generation"]["reasoning"] == {"effort": "low"}
    assert "temperature" not in report["experiment"]["generation"]
    monkeypatch.setattr("sys.argv", arguments + ["--resume"])
    runner.main()
    assert client.responses.create.call_count == 1
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "medium")
    with pytest.raises(ValueError, match="configuración"):
        runner.main()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    capsys.readouterr()
