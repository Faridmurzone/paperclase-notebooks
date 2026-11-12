import math
import torch
# >>>
for x in [-2, 0, 1, 2, 10]:
    print(f"e^{x:<3} = {math.exp(x):10.3f}")   # siempre positivo

print("log(e^2) =", math.log(math.exp(2)))     # log deshace exp
a, b = 0.5, 0.2
print(f"log(a*b) = {math.log(a * b):.3f}")
print(f"log a + log b = {math.log(a) + math.log(b):.3f}")

print("e^88 en float32:", torch.exp(torch.tensor(88.0)).item())
print("e^89 en float32:", torch.exp(torch.tensor(89.0)).item())
# <<<
