"""Cargador de CIFAR-10 compartido por los snippets de la serie (lo importan ep01–ep10).

CIFAR-10: 60.000 imágenes de 32 × 32 en color, 10 clases (50.000 de train y 10.000 de test).
Viene en el formato «python» original: pickles data_batch_1..5, test_batch y batches.meta en
~/Movies/datos-comunes/cifar-10-batches-py/. Cada batch es un dict con b"data" (10.000 × 3.072 uint8,
primero los 1.024 valores del canal R, después G, después B, cada uno fila por fila) y b"labels".

Uso desde un snippet (correr_snippets.py corre con `python -I`, que NO agrega la carpeta del script a
sys.path; por eso cada snippet hace, arriba del `# >>>`):

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _cifar import cargar, normalizar, CLASES

API (estable: no cambiar firmas sin avisar):
    CLASES                    lista de 10 nombres en castellano, en el orden de las etiquetas 0..9
    CLASES_EN                 los nombres originales de batches.meta
    MEDIA, DESVIO             media y desvío por canal (R, G, B) del train, en escala 0–1 (constantes medidas)
    cargar(parte, n=None)     -> (X uint8 (N, 3, 32, 32), y int64 (N,))   parte = "train" | "test"
    normalizar(X)             uint8 (N, 3, 32, 32) -> float32 (X/255 − MEDIA)/DESVIO, por canal
    loader(X, y, batch=128, mezclar=True, semilla=0)   DataLoader sobre tensores ya normalizados
    permutacion(semilla=0)    permutación fija de las 1.024 posiciones de píxel
    permutar(X, perm)         aplica la misma permutación espacial a todas las imágenes (los 3 canales juntos)
    ruta_lab(*partes)         ruta dentro de lab/ (p. ej. ruta_lab("datos", "cnn_ep05.pt"))

Seguridad: los pickles se leen con un Unpickler restringido que solo acepta los tipos de numpy que el
formato necesita (un pickle arbitrario puede ejecutar código al cargarse).
"""
import os
import pickle

import numpy as np
import torch

DATOS = os.path.expanduser("~/Movies/datos-comunes/cifar-10-batches-py")
LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASES_EN = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CLASES = ["avión", "auto", "pájaro", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camión"]

# Media y desvío por canal de las 50.000 imágenes de train, en escala 0–1 (medidos con medir_media_desvio()).
MEDIA = (0.4914, 0.4822, 0.4465)
DESVIO = (0.2470, 0.2435, 0.2616)


def ruta_lab(*partes):
    return os.path.join(LAB, *partes)


class _Seguro(pickle.Unpickler):
    PERMITIDOS = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
                  ("numpy", "ndarray"), ("numpy", "dtype"),
                  ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) not in self.PERMITIDOS:
            raise pickle.UnpicklingError(f"tipo no permitido en el pickle de CIFAR: {module}.{name}")
        return super().find_class(module, name)


def _leer(nombre):
    with open(os.path.join(DATOS, nombre), "rb") as fh:
        return _Seguro(fh, encoding="bytes").load()


def cargar(parte="train", n=None):
    """Devuelve (X, y): X uint8 de forma (N, 3, 32, 32) y y int64 de forma (N,). `n` recorta a las primeras n."""
    archivos = [f"data_batch_{i}" for i in range(1, 6)] if parte == "train" else ["test_batch"]
    xs, ys = [], []
    for a in archivos:
        d = _leer(a)
        xs.append(np.asarray(d[b"data"], dtype=np.uint8))
        ys.append(np.asarray(d[b"labels"], dtype=np.int64))
        if n is not None and sum(len(y) for y in ys) >= n:
            break
    X = np.concatenate(xs).reshape(-1, 3, 32, 32)
    y = np.concatenate(ys)
    if n is not None:
        X, y = X[:n], y[:n]
    return torch.from_numpy(X), torch.from_numpy(y)


def normalizar(X):
    """uint8 (N, 3, 32, 32) → float32 normalizado por canal con MEDIA y DESVIO del train."""
    m = torch.tensor(MEDIA).view(1, 3, 1, 1)
    s = torch.tensor(DESVIO).view(1, 3, 1, 1)
    return (X.float() / 255 - m) / s


def loader(X, y, batch=128, mezclar=True, semilla=0):
    """DataLoader chico sobre tensores en memoria. X ya normalizado (float)."""
    g = torch.Generator().manual_seed(semilla)
    ds = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=mezclar, generator=g)


def permutacion(semilla=0):
    """Una permutación fija de las 32·32 = 1.024 posiciones de píxel."""
    return torch.randperm(32 * 32, generator=torch.Generator().manual_seed(semilla))


def permutar(X, perm):
    """Misma permutación espacial para todas las imágenes; los tres canales de cada píxel viajan juntos."""
    N, C = X.shape[:2]
    return X.reshape(N, C, 32 * 32)[:, :, perm].reshape(N, C, 32, 32)


def medir_media_desvio():
    X, _ = cargar("train")
    x = X.float() / 255
    return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))
