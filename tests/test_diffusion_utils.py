import torch
from Utils.diffusion_utils import drop_text_condition, drop_image_condition


def test_drop_text_condition_full_drop():
    batch, tokens, dim = 2, 3, 4
    text_embed = torch.randn(batch, tokens, dim)
    im = torch.randn(batch, 3, 32, 32)
    empty = torch.randn(1, tokens, dim)

    out = drop_text_condition(text_embed.clone(), im, empty, 1.0)
    expected = empty[0].expand_as(text_embed)
    assert torch.allclose(out, expected)


def test_drop_image_condition_full_drop():
    batch, dim = 3, 5
    image_embed = torch.randn(batch, dim)
    empty = torch.randn(1, dim)

    out = drop_image_condition(image_embed.clone(), empty, 1.0)
    expected = empty[0].expand_as(image_embed)
    assert torch.allclose(out, expected)
