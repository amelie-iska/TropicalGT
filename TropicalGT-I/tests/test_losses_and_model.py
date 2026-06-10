import torch
from tropicalgt.data import FixtureGraphDataset, encode_bytes
from tropicalgt.losses import GFlowNetPolicy, GraphCGLoss
from tropicalgt.model import TropicalGTConfig, TropicalGTModel
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
