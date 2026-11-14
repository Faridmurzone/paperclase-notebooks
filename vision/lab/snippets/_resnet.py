"""Redes de los ep07–ep09 (las de ep01–ep05 están en _redes.py).

RedCifar(n, residual, bn): la familia de CIFAR del paper de ResNet (He et al. 2015, sección 4.2): 6n + 2 capas con pesos.
  conv 3×3 (3→16) · 3 etapas de n bloques de dos conv 3×3 (16, 32, 64 canales; la 2.ª y 3.ª arrancan con stride 2)
  · average pooling global · Linear(64, 10). n = 1 → 8 capas, n = 3 → 20 capas.
  residual=True suma el atajo x (con proyección 1×1 cuando cambian canales o tamaño); bn=False saca la BatchNorm.
ViTChica: Vision Transformer chico para 32×32 (patches de 4×4 → 64 tokens + CLS).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(c, bn):
    return nn.BatchNorm2d(c) if bn else nn.Identity()


class Bloque(nn.Module):
    def __init__(self, c_in, c_out, stride, residual, bn):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=not bn)
        self.bn1 = _norm(c_out, bn)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=not bn)
        self.bn2 = _norm(c_out, bn)
        self.residual = residual
        self.atajo = nn.Identity()
        if residual and (stride != 1 or c_in != c_out):   # proyección 1×1 para que las formas coincidan
            self.atajo = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=not bn), _norm(c_out, bn))

    def forward(self, x):
        f = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))   # F(x)
        return F.relu(f + self.atajo(x)) if self.residual else F.relu(f)


class RedCifar(nn.Module):
    def __init__(self, n=1, residual=False, bn=True, clases=10):
        super().__init__()
        self.kwargs = dict(n=n, residual=residual, bn=bn, clases=clases)
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=not bn)
        self.bn = _norm(16, bn)
        bloques, c = [], 16
        for etapa, c_out in enumerate([16, 32, 64]):
            for i in range(n):
                stride = 2 if (etapa > 0 and i == 0) else 1
                bloques.append(Bloque(c, c_out, stride, residual, bn))
                c = c_out
        self.bloques = nn.Sequential(*bloques)
        self.fc = nn.Linear(64, clases)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.bloques(x)
        return self.fc(x.mean(dim=(2, 3)))       # average pooling global → 64 números


class ViTChica(nn.Module):
    def __init__(self, patch=4, d=128, capas=6, heads=4, clases=10, img=32):
        super().__init__()
        self.kwargs = dict(patch=patch, d=d, capas=capas, heads=heads, clases=clases, img=img)
        n = (img // patch) ** 2
        self.patch_embed = nn.Conv2d(3, d, kernel_size=patch, stride=patch)   # patch embedding
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)               # posiciones aprendidas
        capa = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, activation="gelu",
                                          batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(capa, capas, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, clases)

    def tokens(self, x):
        t = self.patch_embed(x).flatten(2).transpose(1, 2)        # (N, 64, d)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)    # (N, 65, d)
        return t + self.pos

    def forward(self, x):
        z = self.encoder(self.tokens(x))
        return self.fc(self.norm(z[:, 0]))                        # la salida del CLS


ARQ = {"RedCifar": RedCifar, "ViTChica": ViTChica}


def cargar_red(ruta):
    ck = torch.load(ruta, map_location="cpu", weights_only=True)
    m = ARQ[ck["arquitectura"]](**ck["kwargs"])
    m.load_state_dict(ck["estado"])
    return m.eval(), ck["extra"]


def guardar_red(m, ruta, **extra):
    torch.save({"arquitectura": type(m).__name__, "kwargs": m.kwargs,
                "estado": {k: v.detach().cpu() for k, v in m.state_dict().items()}, "extra": extra}, ruta)


def contar(m):
    return sum(p.numel() for p in m.parameters())
