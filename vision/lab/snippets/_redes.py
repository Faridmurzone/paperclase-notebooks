"""Redes y utilidades de entrenamiento compartidas por los snippets de la serie (lo importan ep01–ep10).

Uso desde un snippet (arriba del `# >>>`, porque correr_snippets.py corre con `python -I`):

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _redes import MLP, LeNet, CNNChica, cargar_modelo, evaluar

API (estable: no cambiar firmas sin avisar):
    MLP(entrada=3072, oculta=512, clases=10)
        aplanar → Linear(3072, 512) → ReLU → Linear(512, 256) → ReLU → Linear(256, 10)
    LeNet(canales=3, clases=10)
        LeNet-5 adaptada a 32 × 32 en color (ReLU y max pooling en vez de tanh y subsampling):
        conv 5×5 3→6 → ReLU → maxpool 2 → conv 5×5 6→16 → ReLU → maxpool 2 → 400 → 120 → 84 → 10
    CNNChica(canales=3, clases=10)   ← la del ep05, con pesos en lab/datos/cnn_ep05.pt
        conv1: 5×5, 3→32, padding 2 → ReLU → maxpool 2      (32×32 → 16×16)
        conv2: 5×5, 32→64, padding 2 → ReLU → maxpool 2     (16×16 → 8×8)
        conv3: 3×3, 64→64, padding 1 → ReLU → maxpool 2     (8×8 → 4×4)
        fc: Linear(64·4·4 = 1024, 10)
        Sin BatchNorm ni residuales (aparecen más adelante en la serie). Atributos: .conv1 .conv2 .conv3 .fc
        y .features (nn.Sequential con todo lo convolucional), por si se quieren colgar hooks.
    contar_parametros(modelo) -> int
    evaluar(modelo, X, y, batch=1000) -> precisión en [0, 1] (X ya normalizado; corre en el device del modelo)
    predecir(modelo, X, batch=1000) -> tensor de clases predichas
    entrenar(modelo, X, y, epochs, lr=1e-3, batch=128, device="cpu", semilla=0, log=print) -> modelo
        Adam, entropía cruzada; X ya normalizado. Pensado para los preparar_*.py, no para los snippets.
        al_final_de_epoch(epoch, pérdida_media) opcional: se llama al final de cada epoch (p. ej. para medir test).
    guardar_modelo(modelo, ruta, **extra) / cargar_modelo(ruta) -> (modelo en CPU y en eval, dict con extra)
        El checkpoint guarda: arquitectura (nombre de clase), kwargs, estado (state_dict en CPU) y extra
        (p. ej. precision_test, epochs, media y desvío de la normalización).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, entrada=3072, oculta=512, clases=10):
        super().__init__()
        self.kwargs = dict(entrada=entrada, oculta=oculta, clases=clases)
        self.capa1 = nn.Linear(entrada, oculta)
        self.capa2 = nn.Linear(oculta, oculta // 2)
        self.salida = nn.Linear(oculta // 2, clases)

    def forward(self, x):
        x = x.flatten(1)  # (N, 3, 32, 32) → (N, 3072)
        x = F.relu(self.capa1(x))
        x = F.relu(self.capa2(x))
        return self.salida(x)


class LeNet(nn.Module):
    def __init__(self, canales=3, clases=10):
        super().__init__()
        self.kwargs = dict(canales=canales, clases=clases)
        self.conv1 = nn.Conv2d(canales, 6, 5)   # 32 → 28
        self.conv2 = nn.Conv2d(6, 16, 5)        # 14 → 10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, clases)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28 → 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 10 → 5
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNNChica(nn.Module):
    def __init__(self, canales=3, clases=10):
        super().__init__()
        self.kwargs = dict(canales=canales, clases=clases)
        self.conv1 = nn.Conv2d(canales, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.features = nn.Sequential(
            self.conv1, nn.ReLU(), nn.MaxPool2d(2),
            self.conv2, nn.ReLU(), nn.MaxPool2d(2),
            self.conv3, nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 4 * 4, clases)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


ARQUITECTURAS = {"MLP": MLP, "LeNet": LeNet, "CNNChica": CNNChica}


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())


@torch.no_grad()
def predecir(modelo, X, batch=1000):
    dev = next(modelo.parameters()).device
    modelo.eval()
    return torch.cat([modelo(X[i:i + batch].to(dev)).argmax(1).cpu() for i in range(0, len(X), batch)])


def evaluar(modelo, X, y, batch=1000):
    return (predecir(modelo, X, batch) == y).float().mean().item()


def entrenar(modelo, X, y, epochs, lr=1e-3, batch=128, device="cpu", semilla=0, log=print, al_final_de_epoch=None):
    torch.manual_seed(semilla)
    modelo.to(device).train()
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    g = torch.Generator().manual_seed(semilla)
    for ep in range(epochs):
        orden = torch.randperm(len(X), generator=g)
        total, n = 0.0, 0
        modelo.train()
        for i in range(0, len(X), batch):
            idx = orden[i:i + batch]
            xb, yb = X[idx].to(device), y[idx].to(device)
            perdida = F.cross_entropy(modelo(xb), yb)
            opt.zero_grad()
            perdida.backward()
            opt.step()
            total += perdida.item() * len(idx)
            n += len(idx)
        log(f"epoch {ep + 1}/{epochs}: pérdida {total / n:.4f}")
        if al_final_de_epoch is not None:
            al_final_de_epoch(ep + 1, total / n)
    modelo.eval()
    return modelo


def guardar_modelo(modelo, ruta, **extra):
    estado = {k: v.detach().cpu() for k, v in modelo.state_dict().items()}
    torch.save({"arquitectura": type(modelo).__name__, "kwargs": modelo.kwargs, "estado": estado, "extra": extra}, ruta)


def cargar_modelo(ruta):
    ck = torch.load(ruta, map_location="cpu", weights_only=True)
    modelo = ARQUITECTURAS[ck["arquitectura"]](**ck["kwargs"])
    modelo.load_state_dict(ck["estado"])
    modelo.eval()
    return modelo, ck["extra"]
