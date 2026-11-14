"""Convierte las dos fotos de ejemplo que trae scikit-learn (china.jpg y flower.jpg) a un .npz.

En este venv no hay PIL, así que sklearn.datasets.load_sample_images no puede decodificar los JPG.
Los convertimos una sola vez con `sips` (viene con macOS) a BMP de 24 bits y leemos los bytes con numpy.
Salida: lab/datos/fotos.npz con `china` y `flor`, arrays (H, W, 3) uint8 en orden RGB.
Uso: .venv/bin/python preparar_fotos.py
"""
import os
import subprocess
import tempfile

import numpy as np
import sklearn

AQUI = os.path.dirname(os.path.abspath(__file__))
IMGS = os.path.join(os.path.dirname(sklearn.__file__), "datasets", "images")


def leer_bmp(ruta):
    b = open(ruta, "rb").read()
    assert b[:2] == b"BM"
    off = int.from_bytes(b[10:14], "little")
    w = int.from_bytes(b[18:22], "little", signed=True)
    h = int.from_bytes(b[22:26], "little", signed=True)
    bpp = int.from_bytes(b[28:30], "little")
    assert bpp in (24, 32), bpp
    nb = bpp // 8
    fila = (w * nb + 3) // 4 * 4
    px = np.frombuffer(b, np.uint8, count=fila * abs(h), offset=off).reshape(abs(h), fila)[:, : w * nb]
    px = px.reshape(abs(h), w, nb)[:, :, [2, 1, 0]]  # BGR(A) → RGB
    return px[::-1].copy() if h > 0 else px.copy()  # BMP con alto positivo va de abajo hacia arriba


def main():
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for nombre, archivo in (("china", "china.jpg"), ("flor", "flower.jpg")):
            dst = os.path.join(tmp, nombre + ".bmp")
            subprocess.run(["sips", "-s", "format", "bmp", os.path.join(IMGS, archivo), "--out", dst],
                           check=True, capture_output=True)
            out[nombre] = leer_bmp(dst)
            print(nombre, out[nombre].shape, out[nombre].dtype, out[nombre].mean(axis=(0, 1)).round(1))
    os.makedirs(os.path.join(AQUI, "datos"), exist_ok=True)
    np.savez_compressed(os.path.join(AQUI, "datos", "fotos.npz"), **out)


if __name__ == "__main__":
    main()
