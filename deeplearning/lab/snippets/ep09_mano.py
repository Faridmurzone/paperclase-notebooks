import json, os
import torch
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
torch.set_printoptions(precision=2, sci_mode=False)
# >>>
X = torch.tensor([[1.0, 4.0, 10.0,  0.0],     # 4 ejemplos (filas)
                  [2.0, 6.0, 30.0,  1.0],     # × 4 features (columnas)
                  [3.0, 2.0, 20.0, -1.0],
                  [6.0, 8.0, 40.0,  0.0]])
eps = 1e-5

# BatchNorm: por COLUMNA, sobre el batch (dim=0)
mu_b, var_b = X.mean(0), X.var(0, unbiased=False)
bn = (X - mu_b) / torch.sqrt(var_b + eps)

# LayerNorm: por FILA, sobre features (dim=1)
mu_l = X.mean(1, keepdim=True)
var_l = X.var(1, unbiased=False, keepdim=True)
ln = (X - mu_l) / torch.sqrt(var_l + eps)
print("μ por columna:", mu_b.numpy())
print("μ por fila   :", mu_l.squeeze().numpy())
print("medias BatchNorm:", bn.mean(0).abs().numpy().round(2))
print("medias LayerNorm:", ln.mean(1).abs().numpy().round(2))
# <<<
json.dump({"X": X.tolist(), "bn": bn.round(decimals=2).tolist(), "ln": ln.round(decimals=2).tolist(),
           "var_b": var_b.tolist(), "var_l": var_l.squeeze().tolist()},
          open(os.path.join(DATOS, "ep09_matrices.json"), "w"))
