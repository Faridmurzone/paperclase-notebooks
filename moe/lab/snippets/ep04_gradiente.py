import torch
torch.manual_seed(0)
T, E, k = 512, 8, 2
# >>>
logits = torch.randn(T, E, requires_grad=True)
logits_ = logits + torch.tensor([3.0, 2.0, 0, 0, 0, 0, 0, 0])  # 0 y 1, favoritos
probs = logits_.softmax(-1)
idx = logits_.topk(k).indices
f = torch.bincount(idx.reshape(-1), minlength=E) / (T * k)
P = probs.mean(0)
print("f tiene gradiente:", f.requires_grad, "| P:", P.requires_grad)
L = E * (f * P).sum()
P.retain_grad()
L.backward()
print("f       :", [round(v, 2) for v in f.tolist()])
print("dL/dP_i :", [round(v, 2) for v in P.grad.tolist()])   # = E · f_i
# <<<
