"""Mundo de juguete compartido por los snippets de InstructGPT ep07 y ep09 (no es un snippet: no tiene # >>>).
Tokens 0–9: dígitos. 10 = INST (marca de instrucción). Vocabulario de 12.
- Preentrenamiento (D_pretrain): secuencias que cuentan para adelante, x_t = (a + s·t) mod 10, s ∈ {1, 2, 3}.
  La «capacidad» (el benchmark) es el acierto greedy del próximo token en secuencias nuevas de ese tipo.
- Instrucción: prompt [INST, a, a − s] y el usuario quiere que siga contando PARA ATRÁS: y_t = (y_{t−1} − s) mod 10.
  oro(y) = cuántos de los 6 tokens de la respuesta siguen la regla (0 a 6). Es el gusto oculto; el RM lo aprende en ep09.
Todo sintético; corre en CPU en segundos."""
import os
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

V, INST, L_PRE, P, N = 12, 10, 10, 3, 6
AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(AQUI, "..", "datos")
cfg = LlamaConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32)


def modelo(seed=0):
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg)


def batch_pre(n, g=None):
    a = torch.randint(0, 10, (n, 1), generator=g)
    s = torch.randint(1, 4, (n, 1), generator=g)
    return (a + s * torch.arange(L_PRE)) % 10


def prompts(n, g=None):
    a = torch.randint(0, 10, (n, 1), generator=g)
    s = torch.randint(1, 4, (n, 1), generator=g)
    return torch.cat([torch.full((n, 1), INST), a, (a - s) % 10], 1)


def oro(ids):
    """ids: [batch, P + N]. Cuenta los tokens de la respuesta que siguen contando para atrás."""
    s = (ids[:, 1] - ids[:, 2]) % 10
    prev = ids[:, P - 1:-1]
    return ((ids[:, P:] == (prev - s[:, None]) % 10) & (ids[:, P:] < 10)).float().sum(-1)


def demostraciones(n, g=None):
    x = prompts(n, g)
    s = (x[:, 1] - x[:, 2]) % 10
    y = (x[:, 2:3] - s[:, None] * torch.arange(1, N + 1)) % 10
    return torch.cat([x, y], 1)


@torch.no_grad()
def capacidad(model, n=512):
    """Benchmark: acierto greedy del próximo token contando para adelante (desde la posición 2)."""
    g = torch.Generator().manual_seed(123)
    x = batch_pre(n, g)
    pred = model(input_ids=x).logits[:, 1:-1].argmax(-1)
    return (pred == x[:, 2:]).float().mean().item()


def nll_pre(model, x):
    lg = model(input_ids=x).logits[:, :-1]
    return F.cross_entropy(lg.reshape(-1, V), x[:, 1:].reshape(-1))


@torch.no_grad()
def muestrear(model, x, n=N, greedy=False):
    for _ in range(n):
        lg = model(input_ids=x).logits[:, -1]
        t = lg.argmax(-1, keepdim=True) if greedy else torch.multinomial(lg.softmax(-1), 1)
        x = torch.cat([x, t], 1)
    return x


def logprobs(model, ids):
    lp = model(input_ids=ids).logits[:, :-1].log_softmax(-1)
    return lp.gather(-1, ids[:, 1:, None]).squeeze(-1)[:, P - 1:]


def cargar(nombre):
    m = LlamaForCausalLM(cfg)
    m.load_state_dict(torch.load(os.path.join(DATOS, nombre)))
    return m


def guardar(model, nombre):
    torch.save(model.state_dict(), os.path.join(DATOS, nombre))


# ---------------------------------------------------------------- ep09: reward model de juguete
import torch.nn as nn  # noqa: E402
from transformers import LlamaModel  # noqa: E402


class RM(nn.Module):
    """Reward model: el mismo transformer sin la capa de logits, y una capa lineal a un solo número (último token)."""
    def __init__(self):
        super().__init__()
        self.cuerpo = LlamaModel(cfg)
        self.salida = nn.Linear(cfg.hidden_size, 1)

    def forward(self, ids):
        return self.salida(self.cuerpo(input_ids=ids).last_hidden_state[:, -1]).squeeze(-1)


def cargar_rm(nombre):
    rm = RM()
    estado = torch.load(os.path.join(DATOS, nombre))
    sesgo = estado.pop("_sesgo").item()
    rm.load_state_dict(estado)
    return rm.eval(), sesgo


def guardar_rm(rm, sesgo, nombre):
    estado = dict(rm.state_dict())
    estado["_sesgo"] = torch.tensor(sesgo)
    torch.save(estado, os.path.join(DATOS, nombre))
