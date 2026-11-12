"""MoE de juguete de la serie «Mixture of Experts». Fuente única: lo importan los snippets de todos los episodios.

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _juguete import MoE, Denso, Red, rutear, perdida_balanceo, carga, tarea, entrenar

Piezas (firmas estables: no cambiarlas sin avisar a quien escribe los otros episodios):

- `Experto(d, h)`: un MLP (multilayer perceptron) d → h → d con GELU. Es un experto.
- `rutear(logits, k, sesgo=None, renormalizar=True)` → `(pesos [T,k], idx [T,k], probs [T,E])`.
      softmax sobre los E logits, top-k, y (opcional) renormalizar los k pesos para que sumen 1.
      `sesgo` [E] se suma a los logits SOLO para elegir (balanceo sin pérdida auxiliar); los pesos salen de `probs`.
      OJO con k=1: renormalizado, el único peso vale siempre 1 y el router NO recibe gradiente de la tarea.
      Para top-1 estilo Switch usar renormalizar=False (el peso es la probabilidad p_i cruda).
      Renormalizar los k pesos = softmax sobre los k logits elegidos (Shazeer 2017 y Granite lo hacen así).
- `MoE(d, h, E, k, factor_capacidad=None, renormalizar=True, n_compartidos=0)`: la capa.
      forward(x [..., d]) → y [..., d]. Dispatch agrupando por experto (argsort estable) y combine con index_add_.
      Con `factor_capacidad`, cada experto procesa a lo sumo C = ceil(factor · T·k / E) asignaciones; las que
      sobran se descartan (ese token no recibe ese experto; en `Red` sigue por la conexión residual).
      `n_compartidos` expertos (de tamaño h) que procesan todos los tokens, sin router, y se suman.
      Buffer `sesgo` [E] (ceros por defecto): sesgo de selección para balanceo sin pérdida auxiliar.
      Después de cada forward deja en `self.ultimo` (dict, tensores del grafo salvo los conteos):
        logits [T,E], probs [T,E], idx [T,k], pesos [T,k], carga [E] (asignaciones por experto, antes de capacidad),
        descartadas (int: asignaciones descartadas), tokens_sin_experto (int), l_aux (escalar, `perdida_balanceo`).
- `perdida_balanceo(probs, idx, E)`: L_aux = E · Σ_i f_i · P_i, f_i = fracción de asignaciones al experto i
      (cuenta / (T·k)), P_i = probabilidad media del router para i. Vale 1 si la carga es uniforme. f no tiene
      gradiente (sale de un top-k); P sí. Multiplicar por α afuera (Switch usa α = 0,01).
- `Denso(d, h)`: un MLP denso del mismo tipo que un experto (para comparar con los mismos parámetros activos).
- `Red(d_in, d, d_out, capa)`: entrada lineal → h + capa(LayerNorm(h)) (residual) → salida lineal.
- `tarea(n, semilla, d_in=16, d_out=8, grupos=8)`: datos sintéticos. Cada ejemplo pertenece a un grupo (región del
      espacio de entrada) y cada grupo tiene su propia función no lineal. Devuelve (X [n,d_in], Y [n,d_out], g [n]).
- `entrenar(red, X, Y, pasos, lr=3e-3, alfa=0.0, batch=256, semilla=0, cada=None)`: Adam, MSE + alfa·L_aux
      (sumando L_aux de todas las capas MoE de la red). Devuelve dict con 'perdida' (lista por paso) y 'carga'
      (lista de cargas [E] por paso de la primera capa MoE).
- `parametros(m)`: cantidad de parámetros de un módulo.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


class Experto(nn.Module):
    """Un MLP de dos capas: d → h → d. Cada experto de la capa MoE es uno de estos."""

    def __init__(self, d, h):
        super().__init__()
        self.arriba = nn.Linear(d, h)
        self.abajo = nn.Linear(h, d)

    def forward(self, x):
        return self.abajo(F.gelu(self.arriba(x)))


Denso = Experto  # un MLP denso común: mismo tipo de módulo que un experto


def rutear(logits, k, sesgo=None, renormalizar=True):
    """logits [T, E] → (pesos [T, k], idx [T, k], probs [T, E])."""
    probs = logits.softmax(-1)                           # [T, E], cada fila suma 1
    eleccion = logits if sesgo is None else logits + sesgo
    idx = eleccion.topk(k, dim=-1).indices               # los k expertos de cada token
    pesos = probs.gather(-1, idx)                        # sus probabilidades
    if renormalizar:
        pesos = pesos / pesos.sum(-1, keepdim=True)      # que los k sumen 1
    return pesos, idx, probs


def perdida_balanceo(probs, idx, E):
    """L_aux = E · Σ_i f_i · P_i (Switch Transformer, generalizado a top-k contando asignaciones)."""
    T, k = idx.shape
    f = torch.bincount(idx.reshape(-1), minlength=E).float() / (T * k)   # sin gradiente
    P = probs.mean(0)                                                       # con gradiente
    return E * (f * P).sum()


def carga(idx, E):
    """Asignaciones por experto: cuántos (token, experto) le tocaron a cada uno."""
    return torch.bincount(idx.reshape(-1), minlength=E)


class MoE(nn.Module):
    def __init__(self, d, h, E, k, factor_capacidad=None, renormalizar=True, n_compartidos=0):
        super().__init__()
        self.E, self.k = E, k
        self.factor_capacidad, self.renormalizar = factor_capacidad, renormalizar
        self.router = nn.Linear(d, E, bias=False)        # W_r: d × E
        self.expertos = nn.ModuleList(Experto(d, h) for _ in range(E))
        self.compartidos = nn.ModuleList(Experto(d, h) for _ in range(n_compartidos))
        self.register_buffer("sesgo", torch.zeros(E))    # solo para elegir (balanceo sin pérdida auxiliar)
        self.ultimo = {}

    def capacidad(self, T):
        return math.ceil(self.factor_capacidad * T * self.k / self.E)

    def forward(self, x):
        forma = x.shape
        x = x.reshape(-1, forma[-1])                      # [T, d]
        T = x.shape[0]
        logits = self.router(x)                           # [T, E]
        usa_sesgo = bool(self.sesgo.abs().sum() > 0)
        pesos, idx, probs = rutear(logits, self.k, self.sesgo if usa_sesgo else None, self.renormalizar)
        # dispatch: una fila por asignación (token, experto), ordenadas por experto
        tok = torch.arange(T, device=x.device).repeat_interleave(self.k)   # [T·k]
        exp = idx.reshape(-1)
        w = pesos.reshape(-1)
        orden = exp.argsort(stable=True)                  # estable: dentro de cada experto, orden de llegada
        tok, exp, w = tok[orden], exp[orden], w[orden]
        cuenta = torch.bincount(exp, minlength=self.E)
        inicio = torch.cumsum(cuenta, 0) - cuenta
        guardar = torch.ones_like(exp, dtype=torch.bool)
        if self.factor_capacidad is not None:
            pos = torch.arange(len(exp), device=x.device) - inicio[exp]    # puesto en la fila de su experto
            guardar = pos < self.capacidad(T)
        y = torch.zeros_like(x)
        for e in range(self.E):
            a, b = int(inicio[e]), int(inicio[e] + cuenta[e])
            sel = guardar[a:b]
            t = tok[a:b][sel]
            if len(t) == 0:
                continue
            y.index_add_(0, t, w[a:b][sel, None] * self.expertos[e](x[t]))   # combine
        for c in self.compartidos:
            y = y + c(x)
        tocados = torch.zeros(T, dtype=torch.bool, device=x.device)
        tocados[tok[guardar]] = True
        self.ultimo = dict(logits=logits, probs=probs, idx=idx, pesos=pesos, carga=cuenta.detach(),
                           descartadas=int((~guardar).sum()), tokens_sin_experto=int((~tocados).sum()),
                           l_aux=perdida_balanceo(probs, idx, self.E))
        return y.reshape(forma)


class Red(nn.Module):
    """entrada → residual con una capa (MoE o Denso) → salida. Si un token no recibe ningún experto, h pasa igual."""

    def __init__(self, d_in, d, d_out, capa):
        super().__init__()
        self.entrada, self.norma, self.capa, self.salida = nn.Linear(d_in, d), nn.LayerNorm(d), capa, nn.Linear(d, d_out)

    def forward(self, x):
        h = self.entrada(x)
        h = h + self.capa(self.norma(h))
        return self.salida(h)


def tarea(n, semilla, d_in=16, d_out=8, grupos=8):
    """Datos sintéticos: `grupos` regiones del espacio de entrada, cada una con su propia función no lineal.
    El generador de las funciones es fijo (semilla 1234) para que train y test compartan la tarea; `semilla`
    solo cambia qué ejemplos se sortean."""
    g0 = torch.Generator().manual_seed(1234)
    centros = 3 * torch.randn(grupos, d_in, generator=g0)
    A = torch.randn(grupos, d_in, d_out, generator=g0) / math.sqrt(d_in)
    B = torch.randn(grupos, d_in, d_out, generator=g0) / math.sqrt(d_in)
    g = torch.Generator().manual_seed(semilla)
    grupo = torch.randint(0, grupos, (n,), generator=g)
    X = centros[grupo] + torch.randn(n, d_in, generator=g)
    Z = X - centros[grupo]
    Y = torch.tanh(torch.einsum("nd,ndo->no", Z, A[grupo])) + 0.5 * torch.einsum("nd,ndo->no", Z, B[grupo]) ** 2
    return X, Y, grupo


def capas_moe(red):
    return [m for m in red.modules() if isinstance(m, MoE)]


def entrenar(red, X, Y, pasos, lr=3e-3, alfa=0.0, batch=256, semilla=0, cada=None):
    g = torch.Generator().manual_seed(semilla)
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    hist = dict(perdida=[], carga=[])
    for paso in range(pasos):
        i = torch.randint(0, len(X), (batch,), generator=g)
        mse = F.mse_loss(red(X[i]), Y[i])
        moes = capas_moe(red)
        aux = sum(m.ultimo["l_aux"] for m in moes) if moes else torch.zeros(())
        loss = mse + alfa * aux
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist["perdida"].append(mse.item())
        if moes:
            hist["carga"].append(moes[0].ultimo["carga"].tolist())
        if cada and (paso + 1) % cada == 0:
            print(paso + 1, round(mse.item(), 4))
    return hist


def parametros(m):
    return sum(p.numel() for p in m.parameters())
