"""L4：ShortConv 与 Engram 模块 —— 因果卷积、零初始化、分支门控。"""

from __future__ import annotations

from dataclasses import replace

import torch

from tinyengram import CompressedTokenizer, Engram, ShortConv
from _utils import assert_close, make_ids, tiny_engram_config


def test_short_conv_is_causal():
    torch.manual_seed(0)
    conv = ShortConv(8, groups=2, kernel_size=4, dilation=3)
    with torch.no_grad():
        conv.conv.weight.normal_()
    x = torch.randn(1, 12, 2, 8)
    y = conv(x)
    changed = x.clone()
    changed[:, 7:] += 10.0
    y2 = conv(changed)
    assert_close("因果卷积前缀", y[:, :7], y2[:, :7], atol=1e-6)


def test_short_conv_zero_init_is_identity():
    torch.manual_seed(0)
    conv = ShortConv(8, groups=2, kernel_size=4, dilation=3, zero_init=True)
    x = torch.randn(2, 6, 2, 8)
    assert_close("零初始化", conv(x), torch.zeros_like(x), atol=1e-7)


def test_short_conv_dilation_receptive_field():
    conv = ShortConv(1, groups=1, kernel_size=4, dilation=3, activation=False)
    with torch.no_grad():
        conv.conv.weight.fill_(1.0)
    x = torch.zeros(1, 12, 1, 1)
    x[0, 0, 0, 0] = 1.0
    y = conv(x)[0, :, 0, 0]
    assert torch.allclose(y, torch.tensor([1.0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]))


def test_engram_forward_shape_and_gate_range():
    cfg = tiny_engram_config()
    module = Engram(cfg, layer_id=1)
    hidden = torch.randn(2, 6, cfg.hc_mult, cfg.hidden_size)
    ids = make_ids(2, 6, cfg.vocab_size)
    out = module(hidden, ids)
    assert out.shape == hidden.shape
    gates = module.gates(hidden, ids)
    assert gates.shape == (2, 6, cfg.hc_mult)
    assert bool((gates > 0).all() and (gates < 1).all())


def test_engram_init_is_pure_gated_lookup():
    torch.manual_seed(0)
    cfg = tiny_engram_config()
    module = Engram(cfg, layer_id=1)
    hidden = torch.randn(2, 6, cfg.hc_mult, cfg.hidden_size)
    ids = make_ids(2, 6, cfg.vocab_size)

    emb = module.retrieve(ids)
    value = module.value_proj(emb)
    gates = module.gates(hidden, ids)
    expected = gates.unsqueeze(-1) * value.unsqueeze(2)
    assert_close("Engram 初始化 = 门控查表", module(hidden, ids), expected, atol=1e-6)


def test_gates_are_branch_specific():
    torch.manual_seed(0)
    cfg = tiny_engram_config()
    module = Engram(cfg, layer_id=1)
    ids = make_ids(1, 4, cfg.vocab_size)
    hidden = torch.randn(1, 4, cfg.hc_mult, cfg.hidden_size)
    hidden[:, :, 1, :] *= -3.0
    gates = module.gates(hidden, ids)
    assert not torch.allclose(gates[:, :, 0], gates[:, :, 1])


def test_engram_output_is_causal_in_token_ids():
    torch.manual_seed(0)
    cfg = tiny_engram_config()
    module = Engram(cfg, layer_id=1)
    hidden = torch.randn(1, 8, cfg.hc_mult, cfg.hidden_size)
    ids = make_ids(1, 8, cfg.vocab_size)
    changed = ids.clone()
    changed[:, 5:] = (changed[:, 5:] + 7) % cfg.vocab_size
    out1 = module(hidden, ids)
    out2 = module(hidden, changed)
    assert_close("Engram token 因果性", out1[:, :5], out2[:, :5], atol=1e-6)


def test_value_is_shared_but_keys_are_per_branch():
    cfg = tiny_engram_config(hc_mult=4)
    module = Engram(cfg, layer_id=1)
    assert len(module.key_projs) == 4
    assert module.value_proj.weight.shape == (cfg.hidden_size, cfg.dmem)

    # W_V 只有一份，W_K 有 M 份：这是论文为融合 FP8 GEMM 做的参数共享。
    value_params = sum(p.numel() for p in module.value_proj.parameters())
    key_params = sum(p.numel() for p in module.key_projs.parameters())
    assert key_params == 4 * value_params

    # 把所有 W_K 置零 -> gate 恒为 0.5 -> 四个分支输出应完全相同。
    with torch.no_grad():
        for proj in module.key_projs:
            proj.weight.zero_()
    hidden = torch.randn(1, 4, cfg.hc_mult, cfg.hidden_size)
    ids = make_ids(1, 4, cfg.vocab_size)
    out = module(hidden, ids)
    for m in range(1, cfg.hc_mult):
        assert_close(f"分支 {m} 共享 value", out[:, :, m], out[:, :, 0], atol=1e-6)


def test_engram_with_compressed_tokenizer():
    class Tok:
        tokens = ["Apple", "apple", " apple", "hello", "<pad>"]

        def __len__(self):
            return len(self.tokens)

        def decode(self, ids, skip_special_tokens=False):
            return self.tokens[int(ids[0])]

    ct = CompressedTokenizer(Tok())
    # 传 tokenizer 时，cfg.vocab_size / pad_id 是**原始词表**空间
    cfg = replace(tiny_engram_config(vocab_size=5, slots_per_head=32), pad_id=4)
    module = Engram(cfg, layer_id=1, compressed_tokenizer=ct)
    ids = torch.tensor([[0, 1, 2, 3, 4]])
    hidden = torch.randn(1, 5, cfg.hc_mult, cfg.hidden_size)
    out = module(hidden, ids)
    assert out.shape == hidden.shape
    assert module.mapping.vocab_size == len(ct)
    assert module.mapping.pad_id == int(ct.lookup_table[cfg.pad_id])

    # 用同一张 mapping、在压缩 id 空间里重建一个模块，输出必须一致
    cfg2 = replace(cfg, vocab_size=len(ct), pad_id=module.mapping.pad_id)
    module2 = Engram(cfg2, layer_id=1, mapping=module.mapping)
    module2.load_state_dict(module.state_dict())
    assert_close("压缩集成", out, module2(hidden, ct.compress(ids)), atol=1e-6)


def test_engram_rejects_bad_shapes():
    cfg = tiny_engram_config()
    module = Engram(cfg, layer_id=1)
    hidden = torch.randn(2, 6, cfg.hc_mult, cfg.hidden_size)
    try:
        module(hidden, make_ids(2, 5, cfg.vocab_size))
    except ValueError:
        pass
    else:
        raise AssertionError("ids 长度不匹配应该报错")

    try:
        module(torch.randn(2, 6, 3, cfg.hidden_size), make_ids(2, 6, cfg.vocab_size))
    except ValueError:
        pass
    else:
        raise AssertionError("分支数不匹配应该报错")
