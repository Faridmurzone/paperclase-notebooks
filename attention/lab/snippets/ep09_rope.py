import torch
torch.manual_seed(0)
# >>>
def rope(x, pos, base=10000.0):                  # x: [T, d], d par
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2) / d)  # una frecuencia por par
    ang = pos[:, None] * theta                    # [T, d/2]: pos · θ_i
    x1, x2 = x[..., 0::2], x[..., 1::2]           # pares (0,1), (2,3), …
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * ang.cos() - x2 * ang.sin()
    out[..., 1::2] = x1 * ang.sin() + x2 * ang.cos()
    return out

q, k = torch.randn(64), torch.randn(64)          # una query y una key
def score(m, n):                                 # query en m, key en n
    return rope(q[None], torch.tensor([m]))[0] @ rope(k[None], torch.tensor([n]))[0]

for m, n in [(3, 1), (10, 8), (100, 98), (5, 1)]:
    print(f"m = {m:3}, n = {n:3}, m − n = {m - n}: q·k = {score(m, n):.4f}")
print("norma antes y después:", f"{q.norm():.4f}",
      f"{rope(q[None], torch.tensor([50]))[0].norm():.4f}")
# <<<
