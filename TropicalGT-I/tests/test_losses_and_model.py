import torch
from tropicalgt.data import FixtureGraphDataset, encode_bytes
from tropicalgt.losses import GFlowNetPolicy, GraphCGLoss
from tropicalgt.model import TropicalGTConfig, TropicalGTModel, tropical_certificate_objective, tropical_certificate_targets
from tropicalgt.tokenizer import TokenGTTokenizer


def test_losses_backpropagate():
    z = torch.randn(4, 16, requires_grad=True)
    gfn = GFlowNetPolicy(16)
    actions = torch.zeros(4, 2, dtype=torch.long)
    states = z[:, None, :].repeat(1, 2, 1)
    loss = gfn.trajectory_balance_loss(states, actions, torch.ones(4))
    graphcg, _ = GraphCGLoss(16)(z)
    total = loss + graphcg
    total.backward()
    assert z.grad is not None
    assert torch.isfinite(total)


def test_graphcg_loss_reports_full_rank_terms():
    z = torch.randn(5, 16, requires_grad=True)
    loss, metrics = GraphCGLoss(16, num_directions=4)(z)
    assert torch.isfinite(loss)
    for key in [
        "graphcg_full_rank",
        "graphcg_full_rank_penalty",
        "graphcg_effective_rank",
        "graphcg_numerical_rank",
        "graphcg_rank_target",
        "graphcg_min_singular_value",
        "graphcg_max_singular_value",
        "graphcg_direction_effective_rank",
        "graphcg_direction_numerical_rank",
        "graphcg_direction_rank_target",
        "graphcg_direction_singular_min",
        "graphcg_direction_singular_max",
        "graphcg_direction_svd_condition_proxy",
    ]:
        assert key in metrics
        assert torch.isfinite(metrics[key])
    assert metrics["graphcg_direction_rank_target"].item() == 4.0
    loss.backward()
    assert z.grad is not None


def test_graphcg_full_rank_penalty_detects_collapsed_directions():
    z = torch.randn(5, 16, requires_grad=True)
    graphcg = GraphCGLoss(16, num_directions=4)
    with torch.no_grad():
        graphcg.directions.fill_(0.0)
        graphcg.directions[:, 0] = 1.0
    _loss, metrics = graphcg(z)
    assert metrics["graphcg_full_rank_penalty"] > 0
    assert metrics["graphcg_numerical_rank"] < metrics["graphcg_rank_target"]


def test_model_forward_fixture():
    ds = FixtureGraphDataset(2)
    records = [ds[0], ds[1]]
    tok = TokenGTTokenizer(feature_dim=48)
    gb = tok.batch_encode(records)
    xs, ys = zip(*(encode_bytes(r.text, 32) for r in records))
    model = TropicalGTModel(TropicalGTConfig(dim=32, hidden_dim=32, graph_feature_dim=48))
    out = model(torch.stack(xs), gb, torch.stack(ys))
    assert out["logits"].shape[:2] == (2, 32)
    assert torch.isfinite(out["loss"])
    for key in [
        "certificate_loss",
        "certificate_agreement",
        "certificate_edge_agreement",
        "certificate_coverage",
        "wall_hit_rate",
        "support_boundary_hit_rate",
        "margin_min",
        "margin_p05",
        "node_edge_ratio",
        "loss_regularizer_total",
        "loss_regularizer_ratio",
        "loss_certificate_weighted",
        "gflownet_tb_residual_abs_mean",
        "gflownet_log_z",
        "graphcg_full_rank",
        "graphcg_direction_effective_rank",
        "graphcg_direction_numerical_rank",
        "graphcg_direction_singular_min",
    ]:
        assert key in out
        assert torch.isfinite(out[key])


def test_tropical_certificate_targets_allow_edge_endpoints():
    graph = {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a", "target": "b"}]}
    record = FixtureGraphDataset(1)[0]
    record.graph_json = graph
    tok = TokenGTTokenizer(feature_dim=48)
    gb = tok.batch_encode([record])
    targets = tropical_certificate_targets(gb)
    # Token order is graph, node a, node b, edge a->b.
    assert targets[0, 3, 1]
    assert targets[0, 3, 2]
    assert targets[0, 3, 3]
    scores = torch.zeros(1, 4, 4)
    support = torch.tensor([[0, 1, 2, 1]])
    loss, metrics = tropical_certificate_objective(scores, support, gb)
    assert torch.isfinite(loss)
    assert torch.isclose(metrics["certificate_edge_agreement"], torch.tensor(1.0))
