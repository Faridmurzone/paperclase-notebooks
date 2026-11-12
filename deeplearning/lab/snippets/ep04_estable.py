import warnings
import numpy as np
warnings.filterwarnings("ignore")
# >>>
z = np.array([1000.0, 999.0, 998.0])   # logits grandes

ingenua = np.exp(z) / np.exp(z).sum()  # e^1000 no entra en un float
print("ingenua:", ingenua)

m = z.max()                            # restar el máximo
estable = np.exp(z - m) / np.exp(z - m).sum()
print("estable:", estable.round(3))
print("logsumexp:", round(m + np.log(np.exp(z - m).sum()), 3))
# <<<
