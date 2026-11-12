import numpy as np
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)
L = 10                                       # celdas por nivel, de izquierda a derecha

def niveles(n, frac_azar):                   # cada celda: [posición, "veo una moneda"]
    X, y = [], []
    for _ in range(n):
        moneda = rng.integers(L - 1) if rng.random() < frac_azar else L - 1
        es = (np.arange(L) == moneda).astype(float)
        vista = es + rng.normal(scale=0.3, size=L)     # la moneda se ve con ruido
        X.append(np.c_[np.arange(L) / (L - 1), vista]); y.append(es)
    return np.array(X), np.array(y)

def acierto(clf, X, y):                      # va a la celda de mayor puntaje
    s = clf.decision_function(X.reshape(-1, 2)).reshape(len(X), L)
    return (y[np.arange(len(X)), s.argmax(1)] == 1).mean()
# >>>
X_test, y_test = niveles(2000, frac_azar=1.0)   # test: moneda en otro lado
for frac in [0.0, 0.02, 0.2]:                # fracción de niveles
    X, y = niveles(2000, frac_azar=frac)     # con la moneda al azar
    clf = LogisticRegression().fit(X.reshape(-1, 2), y.ravel())
    w_pos, w_moneda = clf.coef_[0]
    print(f"azar {frac:.0%}: w posición {w_pos:+.1f}, w moneda {w_moneda:+.1f}")
    print(f"   entrenamiento {acierto(clf, X, y):.0%}   "
          f"test {acierto(clf, X_test, y_test):.0%}")
# <<<
