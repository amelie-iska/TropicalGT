import torch
from tropicalgt.attention import TropicalRingAttention


def test_blockwise_matches_full():
    torch.manual_seed(0)
    attn = TropicalRingAttention(8)
    x = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    full = attn(x, mask)
    block = attn.blockwise(x, mask, block_size=2)
    assert torch.allclose(full.context, block.context, atol=1e-6)
    assert torch.equal(full.support, block.support)
    assert full.margin.shape == (2, 5)
