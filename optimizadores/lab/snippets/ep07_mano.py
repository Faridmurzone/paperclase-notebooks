import torch

# >>>
class AdamWMano:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-2):
        self.p, self.lr, self.wd, self.eps = list(params), lr, weight_decay, eps
        self.b1, self.b2, self.t = *betas, 0
        self.m = [torch.zeros_like(p) for p in self.p]
        self.v = [torch.zeros_like(p) for p in self.p]

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.p, self.m, self.v):
            p.mul_(1 - self.lr * self.wd)                       # decay, aparte
            m.mul_(self.b1).add_((1 - self.b1) * p.grad)        # m
            v.mul_(self.b2).add_((1 - self.b2) * p.grad ** 2)   # v
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p.sub_(self.lr * mh / (vh.sqrt() + self.eps))       # paso
# <<<
