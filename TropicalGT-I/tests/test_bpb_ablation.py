import json
from pathlib import Path

from tropicalgt.ablation import build_bpb_ablation_report, write_bpb_ablation_artifacts


def _write_report(path: Path, offset: float = 0.0) -> Path:
    history = []
    for step in range(6):
        history.append(
            {
                "step": step + 1,
                "bpb": 6.0 - 0.2 * step + offset,
                "graph_bpb": 9.0 - 0.1 * step + offset,
                "margin_mean": 0.1 + 0.2 * step,
                "support_entropy": 2.0 - 0.1 * step,
                "graphcg_loss": 0.5 + 0.05 * step,
            }
        )
    report = {
        "checkpoint": str(path.with_suffix(".pt")),
        "final_step": len(history),
        "device": "cpu",
        "history": history,
        "metrics": history[-1],
        "eval": {
            "bpb": history[-1]["bpb"] + 0.1,
            "graph_bpb": history[-1]["graph_bpb"] + 0.2,
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_bpb_ablation_report_ranks_correlated_metrics(tmp_path: Path):
    first = _write_report(tmp_path / "baseline_train_report.json", offset=0.0)
    second = _write_report(tmp_path / "variant_train_report.json", offset=-0.3)
    third = _write_report(tmp_path / "third_train_report.json", offset=0.4)
    report = build_bpb_ablation_report([first, second, third], top_k=20)
    assert report["baseline"].endswith("baseline_train_report.json")
    assert len(report["runs"]) == 3
    assert report["deltas_vs_baseline"][1]["deltas"]["delta_bpb"] < 0
    ranked = {(row["target"], row["metric"]): row for row in report["aggregate_metric_rankings"]}
    assert ("bpb", "margin_mean") in ranked
    assert ranked[("bpb", "margin_mean")]["mean_spearman"] < 0
    assert any(row["target"] == "eval_bpb" and row.get("scope") == "final" for row in report["history_correlations"])


def test_bpb_ablation_artifacts_write_json_markdown_and_html(tmp_path: Path):
    report_path = _write_report(tmp_path / "train_report.json")
    paths = write_bpb_ablation_artifacts([report_path], tmp_path / "ablation", render_html=True)
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith("# TropicalGT-I BPB")
    assert Path(paths["html"]).exists()
