"""Ayudas para convertir tensores en matrices que el visual `matriz` del video puede dibujar (ep06–ep09).
El visual pinta |valor| con opacidad (negativos en rojo), así que cada panel se normaliza a [-1, 1] o [0, 1]
por separado y los paneles se separan con una columna/fila de ceros."""
import json
import os

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def norm(m, con_signo=False):
    m = m.float()
    if con_signo:
        return m / (m.abs().max() + 1e-12)
    m = m - m.min()
    return m / (m.max() + 1e-12)


def gris(img):
    """(3, H, W) en 0–1 → (H, W) luminancia"""
    return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]


def mosaico(paneles, por_fila, gap=1, dec=2):
    """lista de (h, w) ya normalizados → lista de filas (listas de floats) con separadores de ceros"""
    h, w = paneles[0].shape
    filas_out = []
    for i0 in range(0, len(paneles), por_fila):
        grupo = paneles[i0:i0 + por_fila]
        for r in range(h):
            fila = []
            for j, p in enumerate(grupo):
                fila += [round(float(p[r, c]), dec) for c in range(w)]
                if j < por_fila - 1:
                    fila += [0.0] * gap
            fila += [0.0] * ((w + gap) * (por_fila - len(grupo)))
            filas_out.append(fila[: por_fila * (w + gap) - gap])
        if i0 + por_fila < len(paneles):
            filas_out += [[0.0] * (por_fila * (w + gap) - gap) for _ in range(gap)]
    return filas_out


def guardar(nombre, obj):
    json.dump(obj, open(os.path.join(LAB, "datos", nombre), "w"))
