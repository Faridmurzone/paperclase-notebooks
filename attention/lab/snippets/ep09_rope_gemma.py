import torch
from transformers import AutoConfig
from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding, apply_rotary_pos_emb
torch.manual_seed(0)
cfg = AutoConfig.from_pretrained("google/gemma-2-2b")


def rope(x, pos, base=10000.0):
    """RoPE del paper de RoFormer: pares vecinos (0,1), (2,3), … rotados un ángulo pos · θ_i."""
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2) / d)          # θ_i = 10000^(−2(i−1)/d)
    ang = pos[:, None] * theta                            # [T, d/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * ang.cos() - x2 * ang.sin()
    out[..., 1::2] = x1 * ang.sin() + x2 * ang.cos()
    return out


# >>>
emb = Gemma2RotaryEmbedding(cfg)                     # el de Gemma 2 2B
T, d = 6, cfg.head_dim                               # head_dim = 256
q, k = torch.randn(1, 1, T, d), torch.randn(1, 1, T, d)
pos = torch.arange(T)
cos, sin = emb(q, pos[None])
q_hf, k_hf = apply_rotary_pos_emb(q, k, cos, sin)    # mitades: (i, i + d/2)

perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
q_paper = rope(q[0, 0, :, perm.argsort()], pos)[:, perm]   # pares vecinos
print("theta de Gemma:", cfg.rope_parameters["rope_theta"],
      "· cos, sin:", tuple(cos.shape))
print("Gemma == RoPE del paper, reordenado:",
      torch.allclose(q_hf[0, 0], q_paper, atol=1e-5))
# <<<
