import numpy as np
rng = np.random.default_rng(0)
N = 1000                                   # problemas por despliegue
honestas = rng.normal(0, 1, 100_000)       # sospecha que da T a código honesto
MU_PUERTA = 2.0                            # el código con puerta parece más raro

def partida(prot, p_ataque, corridas):     # True = nunca se envió una puerta
    ataca = rng.random((corridas, N)) < p_ataque
    s = rng.normal(0, 1, (corridas, N)) + MU_PUERTA * ataca
    if prot == "auditar al principio":
        atrapa = np.arange(N) < 0.02 * N
        evento = np.where(ataca, np.where(atrapa, 1, -1), 0)
    else:
        delega = prot.startswith("monitor + delegar")
        tau_d = tau_delegar if delega else tau_auditar
        evento = np.where(ataca & (s > tau_auditar), 1,        # auditada: atrapado
                 np.where(ataca & (s <= tau_d), -1, 0))         # pasó sin ver
    hay = evento != 0
    primero = evento[np.arange(corridas), hay.argmax(1)]
    return np.where(hay.any(1), primero == 1, True)

def utilidad(prot):                        # 0 = solo T, 1 = solo U
    return 1 - 0.08 if prot.startswith("monitor + delegar") else 1.0
PROTOCOLOS = ["auditar al principio", "monitor de confianza", "monitor + delegar 10%"]
# >>>
def seguridad(prot, p_ataque, corridas=4000):
    return partida(prot, p_ataque, corridas).mean()

tau_auditar = np.quantile(honestas, 0.98)       # audita el 2 % más sospechoso
tau_delegar = np.quantile(honestas, 0.90)       # del 2 % al 10 %: lo escribe T
ataques = [0.0005, 0.001, 0.003, 0.01, 0.03, 0.1]   # el equipo rojo elige
for prot in PROTOCOLOS:
    peor = min(seguridad(prot, p) for p in ataques)
    print(f"{prot:22} utilidad {utilidad(prot):4.0%}  seguridad {peor:4.0%}")
# <<<
