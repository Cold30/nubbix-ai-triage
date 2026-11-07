"""Exercise both application entry points with a deterministic fake provider."""

import json
from types import SimpleNamespace

from src.configuration import GenerationSettings
from src.llm_client import LLMResponse


def fake_provider():
    def generate(prompt):
        text = json.dumps({
            "answer": "Un agente puede revisar el acceso.", "confidence": 0.9,
            "category": "account", "actions": ["Revisar los permisos"],
        })
        return LLMResponse(text, 10, 5, 15, 0.01, "fake", "fake-model")

    return SimpleNamespace(model="fake-model", settings=GenerationSettings(0.4, 128, 5), generate=generate)


def test_classification_cli_produces_json_and_timestamped_metrics(tmp_path, monkeypatch, capsys):
    from src import main

    path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setattr(main, "create_provider", lambda _: fake_provider())
    monkeypatch.setenv("METRICS_PATH", str(path))
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setattr("sys.argv", ["main", "classify", "No puedo acceder."])
    main.main()
    assert json.loads(capsys.readouterr().out)["category"] == "account"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    assert metrics["total_tokens"] == 15
    assert metrics["timestamp"].endswith("+00:00")


def test_evaluation_cli_fresh_and_resume_use_actual_settings(tmp_path, monkeypatch, capsys):
    from evals import runner

    provider = fake_provider()
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    monkeypatch.setattr(runner, "create_provider", lambda _: provider)
    monkeypatch.setenv("METRICS_PATH", str(tmp_path / "metrics.jsonl"))
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    data_path = tmp_path / "dataset.jsonl"
    data_path.write_text(json.dumps({"id": "T1", "query": "No puedo acceder.", "expected_category": "account"}), encoding="utf-8")
    output = tmp_path / "results.json"
    common = ["runner", "--prompt-version", "triage_v1", "--dataset", str(data_path), "--output", str(output), "--delay-seconds", "0"]
    monkeypatch.setattr("sys.argv", common + ["--fresh"])
    runner.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scores"]["accuracy"] == 1
    assert report["experiment"]["generation"] == provider.settings.generation_parameters()
    assert report["experiment"]["request_policy"]["validation_retries"] == 0

    def unexpected_generation(prompt):
        raise AssertionError("Resume must skip completed rows")

    provider.generate = unexpected_generation
    monkeypatch.setattr("sys.argv", common + ["--resume"])
    runner.main()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    capsys.readouterr()
