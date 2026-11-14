"""Juguete de CLIP para el ep10: dígitos de load_digits y frases sintéticas. Compartido por los snippets ep10_*."""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

PALABRAS = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
# plantillas de entrenamiento: {p} = palabra, {c} = cifra
PLANTILLAS = ["un {p}", "el número {c}", "dígito {p}", "un {c} escrito a mano"]


def frase(d, plantilla):
    return plantilla.format(p=PALABRAS[d], c=str(d))


def datos(semilla=0):
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X, dtype=torch.float32) / 16.0      # 1797 × 64, valores 0–1
    y = torch.tensor(y)
    g = torch.Generator().manual_seed(semilla)
    perm = torch.randperm(len(y), generator=g)
    tr, te = perm[:1200], perm[1200:]
    return X[tr], y[tr], X[te], y[te]


VOCAB = sorted({w for d in range(10) for t in PLANTILLAS + ["el número {p}", "una foto de un {p}"]
                for w in frase(d, t).split()})
IDX = {w: i for i, w in enumerate(VOCAB)}


def tokens(frases):
    """lista de frases → (ids, offsets) para EmbeddingBag (bolsa de palabras: el orden no importa)"""
    ids, offs = [], []
    for f in frases:
        offs.append(len(ids))
        ids += [IDX[w] for w in f.split()]
    return torch.tensor(ids), torch.tensor(offs)


class EncImagen(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.red = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, d))

    def forward(self, x):
        return F.normalize(self.red(x), dim=-1)


class EncTexto(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.emb = nn.EmbeddingBag(len(VOCAB), 64, mode="mean")
        self.proy = nn.Linear(64, d)

    def forward(self, frases):
        ids, offs = tokens(frases)
        return F.normalize(self.proy(self.emb(ids, offs)), dim=-1)


def batch_un_digito_por_clase(Xtr, ytr, rng):
    """un ejemplo de cada dígito, en orden 0..9, y una plantilla al azar para cada uno"""
    idx = [rng.choice((ytr == d).nonzero().flatten().tolist()) for d in range(10)]
    return Xtr[idx], [frase(d, rng.choice(PLANTILLAS)) for d in range(10)]
