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
        "graphcg_active_full_rank",
        "graphcg_raw_full_rank",
        "graphcg_active_rank_fraction",
        "graphcg_raw_active_rank_fraction",
        "graphcg_active_full_rank_penalty",
        "graphcg_full_rank_penalty",
        "graphcg_effective_rank",
        "graphcg_numerical_rank",
        "graphcg_rank_target",
        "graphcg_requested_num_directions",
        "graphcg_effective_num_directions",
        "graphcg_embedding_span_rank_target",
        "graphcg_embedding_span_full_rank",
        "graphcg_direction_bank_clamped_to_embedding_dim",
        "graphcg_min_singular_value",
        "graphcg_max_singular_value",
        "graphcg_direction_effective_rank",
        "graphcg_direction_numerical_rank",
        "graphcg_direction_rank_target",
        "graphcg_direction_singular_min",
        "graphcg_direction_singular_max",
        "graphcg_direction_svd_condition_number",
        "graphcg_raw_full_rank_penalty",
        "graphcg_raw_numerical_rank",
        "graphcg_raw_effective_rank",
        "graphcg_full_rank_possible",
    ]:
        assert key in metrics
        assert torch.isfinite(metrics[key])
    assert metrics["graphcg_direction_rank_target"].item() == 4.0
    assert metrics["graphcg_requested_num_directions"].item() == 4.0
    assert metrics["graphcg_effective_num_directions"].item() == 4.0
    assert metrics["graphcg_embedding_dim"].item() == 16.0
    assert metrics["graphcg_embedding_span_rank_target"].item() == 16.0
    assert metrics["graphcg_embedding_span_full_rank"].item() == 0.0
    assert metrics["graphcg_direction_bank_clamped_to_embedding_dim"].item() == 0.0
    assert metrics["graphcg_numerical_rank"].item() == metrics["graphcg_active_rank_target"].item()
    assert metrics["graphcg_full_rank"].item() == 1.0
    assert metrics["graphcg_active_full_rank"].item() == 1.0
    assert metrics["graphcg_active_rank_fraction"].item() == 1.0
    assert metrics["graphcg_active_full_rank_penalty"].item() == 0.0
    assert metrics["graphcg_min_singular_value"].item() > 0.99
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
    assert metrics["graphcg_raw_full_rank"] == 0
    assert metrics["graphcg_raw_numerical_rank"] < metrics["graphcg_rank_target"]
    assert metrics["graphcg_numerical_rank"] == metrics["graphcg_active_rank_target"]
    assert metrics["graphcg_full_rank"] == 1


def test_model_forward_fixture():
    ds = FixtureGraphDataset(2)
    records = [ds[0], ds[1]]
    tok = TokenGTTokenizer(feature_dim=48)
    gb = tok.batch_encode(records)
    xs, ys = zip(*(encode_bytes(r.text, 32) for r in records))
    model = TropicalGTModel(TropicalGTConfig(dim=32, hidden_dim=32, graph_feature_dim=48))
    out = model(torch.stack(xs), gb, torch.stack(ys))
    assert out["logits"].shape[:2] == (2, 32)
    assert model.graphcg.directions.shape == (32, 32)
    assert torch.isfinite(out["loss"])
    for key in [
        "certificate_loss",
        "certificate_objective_loss",
        "certificate_diagnostic_penalty",
        "certificate_loss_reconstruction_error",
        "certificate_valid_token_count",
        "certificate_allowed_target_count_mean",
        "certificate_allowed_target_count_min",
        "tropical_margin_loss",
        "tropical_margin_signed_loss",
        "tropical_margin_signed_objective",
        "tropical_margin_reward",
        "tropical_margin_shortfall_loss",
        "tropical_margin_shortfall_rate",
        "certificate_agreement",
        "certificate_edge_agreement",
        "certificate_coverage",
        "wall_hit_rate",
        "strict_wall_hit_rate",
        "near_wall_hit_rate",
        "wall_margin_threshold",
        "near_wall_margin_threshold",
        "support_boundary_hit_rate",
        "margin_min",
        "margin_p05",
        "node_edge_ratio",
        "loss_regularizer_total",
        "loss_regularizer_ratio",
        "loss_certificate_weighted",
        "loss_certificate_objective_weighted",
        "loss_certificate_diagnostic_penalty_weighted",
        "gflownet_tb_residual_abs_mean",
        "gflownet_log_z",
        "graphcg_full_rank",
        "graphcg_direction_effective_rank",
        "graphcg_direction_numerical_rank",
        "graphcg_direction_singular_min",
        "graphcg_num_directions",
        "graphcg_requested_num_directions",
        "graphcg_effective_num_directions",
        "graphcg_embedding_dim",
        "graphcg_embedding_span_rank_target",
        "graphcg_embedding_span_full_rank",
        "graphcg_direction_bank_clamped_to_embedding_dim",
        "graphcg_active_directions",
        "sequence_tropical_tokens_mean",
        "sequence_tropical_margin_mean",
        "sequence_tropical_support_entropy",
        "bundle_transport_l1",
        "bundle_cocycle_defect",
        "bundle_flat_rank_defect",
        "bundle_monomial_projection_one_hotness",
        "bundle_chart_confidence_mean",
        "bundle_chart_count",
        "bundle_overlap_pair_count",
        "bundle_overlap_triple_count",
        "bundle_atom_stability_gap",
        "bundle_atom_stability_available",
        "toric_activation_cell_margin_loss",
        "toric_normal_fan_loss",
        "toric_active_row_count",
        "graphcg_toric_cell_agreement",
        "graphcg_toric_cell_agreement_available",
        "chart_bpb_consistency",
        "chart_bpb_consistency_available",
        "chart_bpb_global",
        "chart_bpb_min",
        "chart_bpb_max",
        "chart_bpb_spread",
        "chart_bpb_active_count",
        "loss_tropical_margin_signed_weighted",
        "loss_tropical_margin_shortfall_weighted",
        "loss_bundle_transport_weighted",
        "loss_bundle_cocycle_weighted",
        "loss_bundle_flat_rank_weighted",
        "loss_toric_activation_cell_margin_weighted",
        "loss_toric_normal_fan_weighted",
        "loss_graphcg_toric_cell_agreement_weighted",
        "loss_chart_bpb_consistency_weighted",
        "loss_bundle_atom_stability_weighted",
        "loss_bundle_toric_regularizer_total",
    ]:
        assert key in out
        assert torch.isfinite(out[key])
    assert out["graphcg_num_directions"].item() == 32.0
    assert out["graphcg_requested_num_directions"].item() == 32.0
    assert out["graphcg_effective_num_directions"].item() == 32.0
    assert out["graphcg_embedding_dim"].item() == 32.0
    assert out["graphcg_embedding_span_rank_target"].item() == 32.0
    assert out["graphcg_embedding_span_full_rank"].item() == 1.0
    assert out["graphcg_direction_bank_clamped_to_embedding_dim"].item() == 0.0
    assert torch.allclose(out["certificate_loss"], out["certificate_objective_loss"])
    assert out["certificate_diagnostic_penalty"].item() == 0.0
    assert out["certificate_loss_reconstruction_error"].item() == 0.0
    assert torch.allclose(out["loss_certificate_weighted"], out["loss_certificate_objective_weighted"])
    assert out["loss_certificate_diagnostic_penalty_weighted"].item() == 0.0
    assert out["certificate_valid_token_count"].item() > 0.0
    assert out["certificate_allowed_target_count_mean"].item() >= 1.0
    assert torch.allclose(out["tropical_margin_loss"], out["tropical_margin_shortfall_loss"])
    assert out["tropical_margin_loss"].item() >= 0.0
    assert torch.allclose(out["tropical_margin_signed_objective"], out["tropical_margin_signed_loss"])
    assert torch.allclose(out["tropical_margin_reward"], -out["tropical_margin_signed_objective"])
    assert out["tropical_margin_shortfall_loss"].item() >= 0.0
    assert torch.allclose(out["loss_margin_weighted"], out["loss_tropical_margin_signed_weighted"])
    assert out["loss_tropical_margin_shortfall_weighted"].item() >= 0.0
    assert 0.0 <= out["tropical_margin_shortfall_rate"].item() <= 1.0
    assert torch.allclose(out["wall_hit_rate"], out["strict_wall_hit_rate"])
    assert out["near_wall_hit_rate"].item() >= out["strict_wall_hit_rate"].item()
    assert out["near_wall_margin_threshold"].item() >= out["wall_margin_threshold"].item()
    assert out["chart_bundle_transport_metadata"]["available"] is False
    disabled_cert = out["chart_bundle_transport_metadata"]["toric_embedding_certificate"]
    assert disabled_cert["available"] is False
    assert disabled_cert["safe_to_render_as_toric_embedding"] is False
    assert disabled_cert["safe_to_render_as_tropical_variety_embedding"] is False
    assert disabled_cert["safe_to_render_as_global_toric_variety_embedding"] is False
    assert disabled_cert["embedding_scope"] == "uncertified_activation_chart_not_tropical_variety_embedding"
    assert "not toric embeddings" in disabled_cert["no_proxy_policy"]
    assert "tropical-variety embeddings" in disabled_cert["no_proxy_policy"]


def _fixture_batch(batch_size: int = 2, seq_len: int = 32):
    ds = FixtureGraphDataset(batch_size)
    records = [ds[idx] for idx in range(batch_size)]
    tok = TokenGTTokenizer(feature_dim=48)
    gb = tok.batch_encode(records)
    xs, ys = zip(*(encode_bytes(r.text, seq_len) for r in records))
    return torch.stack(xs), gb, torch.stack(ys)


def test_chart_bundle_auxiliary_zero_weights_do_not_change_logits_or_loss():
    input_ids, graph_batch, target_ids = _fixture_batch()
    base = TropicalGTModel(TropicalGTConfig(dim=32, hidden_dim=32, graph_feature_dim=48, graphcg_num_directions=32))
    aux = TropicalGTModel(
        TropicalGTConfig(
            dim=32,
            hidden_dim=32,
            graph_feature_dim=48,
            graphcg_num_directions=32,
            enable_chart_bundle_auxiliary=True,
            bundle_num_charts=3,
            toric_num_active_rows=5,
            bundle_transport_weight=0.0,
            bundle_cocycle_weight=0.0,
            bundle_flat_rank_weight=0.0,
            toric_normal_fan_weight=0.0,
            graphcg_toric_cell_agreement_weight=0.0,
            chart_bpb_consistency_weight=0.0,
            bundle_atom_stability_weight=0.0,
        )
    )
    aux.load_state_dict(base.state_dict(), strict=False)
    base_out = base(input_ids, graph_batch, target_ids)
    aux_out = aux(input_ids, graph_batch, target_ids)
    assert torch.allclose(aux_out["logits"], base_out["logits"], atol=0.0, rtol=0.0)
    assert torch.allclose(aux_out["loss"], base_out["loss"], atol=1e-6, rtol=1e-6)
    assert aux_out["loss_bundle_toric_regularizer_total"].item() == 0.0
    assert aux_out["bundle_monomial_projection_one_hotness"].item() == 1.0
    assert aux_out["bundle_chart_count"].item() == 3.0
    assert aux_out["bundle_overlap_pair_count"].item() == 6.0
    assert aux_out["bundle_overlap_triple_count"].item() == 6.0
    assert aux_out["toric_active_row_count"].item() == 5.0
    metadata = aux_out["chart_bundle_transport_metadata"]
    assert metadata["available"] is True
    assert metadata["chart_ids"] == ["chart_00", "chart_01", "chart_02"]
    assert {row["id"] for row in metadata["overlap_pairs"]} >= {"chart_00__to__chart_01", "chart_01__to__chart_02"}
    assert metadata["overlap_triples"][0]["pair_ids"]
    toric_cert = metadata["toric_embedding_certificate"]
    assert toric_cert["available"] is False
    assert toric_cert["status"] == "uncertified_activation_chart"
    assert toric_cert["safe_to_render_as_toric_embedding"] is False
    assert toric_cert["safe_to_render_as_tropical_variety_embedding"] is False
    assert toric_cert["safe_to_render_as_global_toric_variety_embedding"] is False
    assert toric_cert["embedding_scope"] == "uncertified_activation_chart_not_tropical_variety_embedding"
    assert toric_cert["metric_scope"] == "activation_margin_diagnostic_not_tool_backed_toric_embedding"
    assert "Chart-bundle logits" in toric_cert["no_proxy_policy"]
    assert "tropical-variety embeddings" in toric_cert["no_proxy_policy"]
    assert aux_out["chart_bpb_consistency_available"].item() == 1.0
    assert aux_out["chart_bpb_active_count"].item() == 3.0
    assert aux_out["chart_bpb_spread"].item() >= 0.0
    assert aux_out["chart_bpb_min"].item() <= aux_out["chart_bpb_global"].item() <= aux_out["chart_bpb_max"].item()
    assert aux_out["bundle_atom_stability_available"].item() == 1.0


def test_chart_bundle_bpb_partition_unavailable_without_targets():
    input_ids, graph_batch, _target_ids = _fixture_batch()
    model = TropicalGTModel(
        TropicalGTConfig(
            dim=32,
            hidden_dim=32,
            graph_feature_dim=48,
            graphcg_num_directions=32,
            enable_chart_bundle_auxiliary=True,
            bundle_num_charts=3,
            toric_num_active_rows=5,
        )
    )
    out = model(input_ids, graph_batch)
    assert out["chart_bpb_consistency_available"].item() == 0.0
    assert out["chart_bpb_global"].item() == 0.0
    assert out["chart_bpb_active_count"].item() == 0.0


def test_chart_bundle_auxiliary_positive_weights_change_loss_not_logits():
    input_ids, graph_batch, target_ids = _fixture_batch()
    model = TropicalGTModel(
        TropicalGTConfig(
            dim=32,
            hidden_dim=32,
            graph_feature_dim=48,
            graphcg_num_directions=32,
            enable_chart_bundle_auxiliary=True,
            bundle_num_charts=4,
            toric_num_active_rows=6,
            bundle_transport_weight=0.0,
            bundle_cocycle_weight=0.0,
            bundle_flat_rank_weight=0.0,
            toric_normal_fan_weight=0.0,
            graphcg_toric_cell_agreement_weight=0.0,
            chart_bpb_consistency_weight=0.0,
            bundle_atom_stability_weight=0.0,
        )
    )
    zero_out = model(input_ids, graph_batch, target_ids)
    model.config.bundle_transport_weight = 0.05
    model.config.bundle_cocycle_weight = 0.02
    model.config.toric_normal_fan_weight = 0.03
    model.config.chart_bpb_consistency_weight = 0.04
    model.config.bundle_atom_stability_weight = 0.01
    weighted_out = model(input_ids, graph_batch, target_ids)
    assert torch.allclose(weighted_out["logits"], zero_out["logits"], atol=0.0, rtol=0.0)
    assert weighted_out["loss_bundle_toric_regularizer_total"].item() >= 0.0
    assert weighted_out["loss"].item() >= zero_out["loss"].item()
    assert weighted_out["bundle_transport_l1"].item() >= 0.0
    assert weighted_out["bundle_cocycle_defect"].item() >= 0.0
    assert weighted_out["toric_activation_cell_margin_loss"].item() >= 0.0
    assert torch.allclose(weighted_out["toric_normal_fan_loss"], weighted_out["toric_activation_cell_margin_loss"])
    assert torch.allclose(weighted_out["loss_toric_normal_fan_weighted"], weighted_out["loss_toric_activation_cell_margin_weighted"])
    assert weighted_out["graphcg_toric_cell_agreement_available"].item() == 1.0
    assert weighted_out["chart_bpb_consistency_available"].item() == 1.0
    assert weighted_out["loss_chart_bpb_consistency_weighted"].item() >= 0.0


def test_model_allows_explicit_full_embedding_graphcg_bank():
    model = TropicalGTModel(
        TropicalGTConfig(
            dim=24,
            hidden_dim=24,
            graph_feature_dim=48,
            graphcg_num_directions=24,
            graphcg_active_directions=6,
        )
    )
    assert model.graphcg.directions.shape == (24, 24)
    assert model.graphcg.requested_num_directions == 24
    assert model.graphcg.effective_num_directions == 24
    assert model.graphcg.direction_bank_clamped_to_embedding_dim is False
    assert model.graphcg.active_directions == 6


def test_model_clamps_graphcg_direction_bank_to_embedding_dim():
    input_ids, graph_batch, target_ids = _fixture_batch()
    model = TropicalGTModel(
        TropicalGTConfig(
            dim=32,
            hidden_dim=32,
            graph_feature_dim=48,
            graphcg_num_directions=8,
            graphcg_active_directions=16,
        )
    )
    assert model.graphcg.directions.shape == (32, 32)
    assert model.graphcg.requested_num_directions == 8
    assert model.graphcg.effective_num_directions == 32
    assert model.graphcg.direction_bank_clamped_to_embedding_dim is True
    out = model(input_ids, graph_batch, target_ids)
    assert out["graphcg_requested_num_directions"].item() == 8.0
    assert out["graphcg_num_directions"].item() == 32.0
    assert out["graphcg_effective_num_directions"].item() == 32.0
    assert out["graphcg_embedding_dim"].item() == 32.0
    assert out["graphcg_embedding_span_rank_target"].item() == 32.0
    assert out["graphcg_embedding_span_full_rank"].item() == 1.0
    assert out["graphcg_direction_bank_clamped_to_embedding_dim"].item() == 1.0


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
