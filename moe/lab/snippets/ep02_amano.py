import torch
torch.set_printoptions(precision=2)
# >>>
x = torch.tensor([1.0, 0.5, -1.0])            # un token, d = 3
W_r = torch.tensor([[1.0, 0.0, 0.5, -1.0],    # d × E = 3 × 4
                    [0.0, 2.0, 0.0, 1.0],
                    [0.5, 0.0, -1.0, 0.0]])
logits = x @ W_r                              # un número por experto
probs = logits.softmax(-1)                    # suman 1
top = probs.topk(2)                           # k = 2
g = top.values / top.values.sum()             # renormalizar: suman 1
print("logits:", logits)
print("probs: ", probs)
print("top-2: ", top.indices.tolist(), top.values)
print("g:     ", g)
# <<<
