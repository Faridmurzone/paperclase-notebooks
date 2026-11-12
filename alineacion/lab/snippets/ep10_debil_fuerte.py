import numpy as np
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)
D, K = 40, 12
w = rng.normal(size=D)
def tarea(n):                                   # verdad: signo de w·x
    X = rng.normal(size=(n, D)); return X, (X @ w > 0).astype(int)
R = rng.normal(size=(D, 2))
def rep_fuerte(X):                              # "preentrenado": el concepto es saliente
    return np.c_[X @ w / np.sqrt(D) + rng.normal(scale=0.3, size=len(X)), X @ R]
acc = lambda clf, Z, y: (clf.predict(Z) == y).mean()
X_a, y_a = tarea(100)                           # pocas etiquetas de verdad
X_b, y_b = tarea(4000)                          # datos sin etiquetar
X_t, y_t = tarea(4000)                          # test
# >>>
debil = LogisticRegression().fit(X_a[:, :K], y_a)    # ve K de 40 rasgos
etiq_debiles = debil.predict(X_b[:, :K])             # etiqueta datos nuevos
for nombre, rep in [("saliente", rep_fuerte), ("crudo", lambda X: X)]:
    alumno = LogisticRegression().fit(rep(X_b), etiq_debiles)
    techo = LogisticRegression().fit(rep(X_b), y_b)  # con la verdad
    d = acc(debil, X_t[:, :K], y_t)
    a, t = acc(alumno, rep(X_t), y_t), acc(techo, rep(X_t), y_t)
    print(f"{nombre:9} débil {d:.2f}  alumno {a:.2f}  techo {t:.2f}"
          f"  PGR {round((a - d) / (t - d), 2) + 0.0:.2f}")
# <<<
