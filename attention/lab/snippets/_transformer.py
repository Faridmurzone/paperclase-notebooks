"""Transformer encoder-decoder desde cero (episodio 10). Fuente única: los snippets ep10_* muestran estas mismas clases,
copiadas textuales entre los marcadores «# ── NOMBRE ──» (las genera lab/snippets/_gen_ep10.py). Post-norm, como el paper."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

V, BOS, L = 11, 10, 10                     # dígitos 0–9 y BOS; largo 10
torch.set_num_threads(1)


def pe(n, d):
    p = torch.arange(n).unsqueeze(1); om = 1 / 10000 ** (torch.arange(0, d, 2) / d)
    out = torch.zeros(n, d); out[:, 0::2], out[:, 1::2] = torch.sin(p * om), torch.cos(p * om)
    return out


def datos(n, gen=None, largo=L):
    x = torch.randint(0, 10, (n, largo), generator=gen); y = x.sort(1).values
    return x, y, torch.cat([torch.full((n, 1), BOS), y[:, :-1]], 1)


# ── MULTIHEAD ──
class MultiHead(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h, s.dk = h, d // h
        s.wq, s.wk, s.wv, s.wo = (nn.Linear(d, d) for _ in range(4))

    def forward(s, xq, xkv, prohibido=None):       # prohibido: True = no mirar
        B, Tq, Tk = xq.shape[0], xq.shape[1], xkv.shape[1]
        q = s.wq(xq).view(B, Tq, s.h, s.dk).transpose(1, 2)    # [B, h, Tq, dk]
        k = s.wk(xkv).view(B, Tk, s.h, s.dk).transpose(1, 2)
        v = s.wv(xkv).view(B, Tk, s.h, s.dk).transpose(1, 2)
        S = q @ k.transpose(-2, -1) / s.dk ** 0.5               # [B, h, Tq, Tk]
        if prohibido is not None:
            S = S.masked_fill(prohibido, float("-inf"))
        out = S.softmax(-1) @ v                                 # [B, h, Tq, dk]
        return s.wo(out.transpose(1, 2).reshape(B, Tq, -1))
# ── FIN ──


# ── ENCODER ──
def ffn(d, d_ff):
    return nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Linear(d_ff, d))

class CapaEncoder(nn.Module):
    def __init__(s, d, h, d_ff, p):
        super().__init__()
        s.att, s.ffn = MultiHead(d, h), ffn(d, d_ff)
        s.ln = nn.ModuleList(nn.LayerNorm(d) for _ in range(2))
        s.drop = nn.Dropout(p)

    def forward(s, x):                                # post-norm, como el paper
        x = s.ln[0](x + s.drop(s.att(x, x)))
        return s.ln[1](x + s.drop(s.ffn(x)))
# ── FIN ──


# ── DECODER ──
class CapaDecoder(nn.Module):
    def __init__(s, d, h, d_ff, p):
        super().__init__()
        s.att, s.cruzada, s.ffn = MultiHead(d, h), MultiHead(d, h), ffn(d, d_ff)
        s.ln = nn.ModuleList(nn.LayerNorm(d) for _ in range(3))
        s.drop = nn.Dropout(p)

    def forward(s, y, memoria, causal):
        y = s.ln[0](y + s.drop(s.att(y, y, causal)))          # self, con máscara
        y = s.ln[1](y + s.drop(s.cruzada(y, memoria)))        # cross-attention
        return s.ln[2](y + s.drop(s.ffn(y)))
# ── FIN ──


# ── MODELO ──
class Transformer(nn.Module):
    def __init__(s, d=32, h=4, N=2, d_ff=128, p=0.1):
        super().__init__()
        s.d, s.emb = d, nn.Embedding(V, d)           # una sola matriz
        s.register_buffer("pe", pe(64, d))           # senos del episodio 4
        s.enc = nn.ModuleList(CapaEncoder(d, h, d_ff, p) for _ in range(N))
        s.dec = nn.ModuleList(CapaDecoder(d, h, d_ff, p) for _ in range(N))
        s.drop = nn.Dropout(p)
    def entrada(s, t):                               # embedding · √d + posición
        return s.drop(s.emb(t) * math.sqrt(s.d) + s.pe[:t.shape[1]])
    def codificar(s, x):
        m = s.entrada(x)
        for capa in s.enc: m = capa(m)
        return m                                     # la memoria
    def decodificar(s, y_in, memoria):
        T = y_in.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        y = s.entrada(y_in)
        for capa in s.dec: y = capa(y, memoria, causal)
        return y @ s.emb.weight.T                    # logits: la misma matriz
    def forward(s, x, y_in):
        return s.decodificar(y_in, s.codificar(x))
# ── FIN ──


def forma(paso, warmup):
    paso = max(paso, 1)
    return min(paso / warmup, math.sqrt(warmup / paso))


# ── GREEDY ──
@torch.no_grad()
def greedy(modelo, x):
    memoria = modelo.codificar(x)                    # el encoder corre una vez
    y = torch.full((x.shape[0], 1), BOS)
    for _ in range(x.shape[1]):
        logits = modelo.decodificar(y, memoria)[:, -1]   # solo el último
        logits[:, BOS] = float("-inf")               # BOS nunca es salida
        y = torch.cat([y, logits.argmax(-1, keepdim=True)], dim=1)
    return y[:, 1:]                                  # sin el BOS
# ── FIN ──
