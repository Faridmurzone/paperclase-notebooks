import torch
torch.manual_seed(0)
pre = torch.tensor([2.5, -1.0, 0.3, 1.2, 0.05, -0.4])     # W_e x + b_e de un token
theta = torch.tensor([1.0, 0.5, 0.5, 0.5, 0.01, 0.5])     # umbral aprendido por feature
# >>>
relu = torch.relu(pre)

k = 2                                                      # TopK: k fijo
val, idx = pre.topk(k)
topk = torch.zeros_like(pre).scatter(0, idx, torch.relu(val))

jump = torch.where(pre > theta, pre, 0.0)                    # JumpReLU: z·H(z − θ)

for nombre, f in [("ReLU", relu), ("TopK", topk), ("JumpReLU", jump)]:
    print(f"{nombre:9}", [round(v, 2) for v in f.tolist()], "L0 =", int((f > 0).sum()))
# <<<
