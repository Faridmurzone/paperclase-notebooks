import torch
def r(t, d=3): return [round(x, d) for x in t.tolist()]
# >>>
l = torch.tensor([1000.0, 999.0, 998.0])

ingenuo = torch.exp(l) / torch.exp(l).sum()
print("ingenuo:", ingenuo.tolist())            # inf / inf

m = l.max()                                    # restar el máximo
estable = torch.exp(l - m) / torch.exp(l - m).sum()
print("e^(l - m):", r(torch.exp(l - m)))
print("estable:", r(estable))

lse = m + torch.log(torch.exp(l - m).sum())    # log-sum-exp
print("LSE a mano:", round(lse.item(), 3))
print("torch.logsumexp:", round(torch.logsumexp(l, 0).item(), 3))
print("log_softmax:", r(torch.log_softmax(l, 0)))
# <<<
