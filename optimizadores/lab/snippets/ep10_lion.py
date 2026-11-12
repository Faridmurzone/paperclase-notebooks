import torch

# >>>
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, wd=weight_decay))

    @torch.no_grad()
    def step(self):
        for grupo in self.param_groups:
            b1, b2 = grupo["betas"]
            for p in grupo["params"]:
                m = self.state[p].setdefault("m", torch.zeros_like(p))
                c = b1 * m + (1 - b1) * p.grad          # mezclar g y m
                p.mul_(1 - grupo["lr"] * grupo["wd"])   # decay desacoplado
                p.sub_(grupo["lr"] * torch.sign(c))     # solo el signo
                m.mul_(b2).add_((1 - b2) * p.grad)      # momentum, con β₂
# <<<
