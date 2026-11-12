import torch
import torch.nn as nn
# >>>
M = nn.Transformer.generate_square_subsequent_mask(4)
print(M)                                  # float: 0 o -inf
nuestra = torch.triu(torch.ones(4, 4, dtype=torch.bool), 1)
print("misma máscara:", torch.equal(M == float("-inf"), nuestra))
# <<<
