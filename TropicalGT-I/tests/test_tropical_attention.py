import torch
from tropicalgt.attention import TropicalRingAttention, soft_tropical_support_entropy, tropical_support_entropy


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


def test_masked_support_entropy_excludes_padding():
    support = torch.tensor([[0, 1, 4, 4]])
    mask = torch.tensor([[True, True, False, False]])
    masked = tropical_support_entropy(support, mask)
    unmasked = tropical_support_entropy(support)
    assert masked < unmasked


def test_soft_entropy_and_all_masked_attention_are_finite():
    torch.manual_seed(0)
    attn = TropicalRingAttention(8)
    x = torch.randn(1, 3, 8)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    out = attn(x, mask)
    assert torch.isfinite(out.context).all()
    assert torch.isfinite(out.margin).all()
    entropy = soft_tropical_support_entropy(out.scores, mask, mask)
    assert torch.isfinite(entropy)
