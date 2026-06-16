import importlib.util
import json
from pathlib import Path


def _load_monitor():
    path = Path(__file__).resolve().parents[1] / "scripts" / "monitor_training_step_gate.py"
    spec = importlib.util.spec_from_file_location("monitor_training_step_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_monitor = _load_monitor()
latest_training_status = _monitor.latest_training_status
monitor_step_gate = _monitor.monitor_step_gate


def test_latest_training_status_parses_tqdm_steps_and_fatal(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text("\rTropicalGT-I train:   0%|          | 42/5000 [00:10<?, ?it/s, loss=1.2, nll=1.1]\nRuntimeError: no\n", encoding="utf-8")
    status = latest_training_status(log)
    assert status["available"] is True
    assert status["latest_step"] == 42
    assert status["latest_loss_nll"] == ["1.2", "1.1"]
    assert status["fatal"] is True



def test_latest_training_status_does_not_treat_inference_as_inf(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text("\rTropicalGT-I train:   0%|          | 44/5000 [00:10<?, ?it/s, loss=1.2, nll=1.1]\nperiodic inference audit complete\n", encoding="utf-8")
    status = latest_training_status(log)
    assert status["latest_step"] == 44
    assert status["fatal"] is False


def test_step_gate_once_records_monitoring_without_kill(tmp_path: Path):
    log = tmp_path / "train.log"
    record = tmp_path / "record.json"
    log.write_text("\rTropicalGT-I train:   0%|          | 12/5000 [00:10<?, ?it/s, loss=1.2, nll=1.1]\n", encoding="utf-8")
    payload = monitor_step_gate(
        pid=__import__("os").getpid(),
        log_path=log,
        target_step=5000,
        record_path=record,
        poll_seconds=1,
        once=True,
        dry_run=True,
    )
    assert payload["action"] == "monitoring"
    assert json.loads(record.read_text(encoding="utf-8"))["status"]["latest_step"] == 12


def test_step_gate_reaches_target_in_dry_run(tmp_path: Path):
    log = tmp_path / "train.log"
    record = tmp_path / "record.json"
    log.write_text("\rTropicalGT-I train:   0%|          | 5000/5000 [00:10<?, ?it/s, loss=1.2, nll=1.1]\n", encoding="utf-8")
    payload = monitor_step_gate(
        pid=__import__("os").getpid(),
        log_path=log,
        target_step=5000,
        record_path=record,
        poll_seconds=1,
        once=True,
        dry_run=True,
    )
    assert payload["action"] == "target_reached_terminate"
    assert json.loads(record.read_text(encoding="utf-8"))["dry_run"] is True
