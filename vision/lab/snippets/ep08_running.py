import torch
torch.manual_seed(0)
bn = torch.nn.BatchNorm2d(1)
# >>>
for paso in range(200):                      # entrenamiento: batches de media 5
    bn(torch.randn(32, 1, 4, 4) + 5)
print("running_mean:", round(bn.running_mean.item(), 3),
      " running_var:", round(bn.running_var.item(), 3))

una = torch.full((1, 1, 4, 4), 6.0)          # UNA imagen, valor 6
bn.eval()                                    # usa running_mean/var
print("eval:  ", round(bn(una).mean().item(), 3))
bn.train()                                   # usa la media de ESTE batch
print("train: ", round(bn(una).mean().item(), 3))
# <<<
