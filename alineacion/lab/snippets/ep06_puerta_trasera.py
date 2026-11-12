import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
A2023, A2024, V = 0, 1, 100        # tokens 0 y 1: "el año es 2023 / 2024"
DUDOSOS = [2, 3, 4, 5]             # pedidos que el modelo inicial hace mal
modelo = nn.Sequential(nn.EmbeddingBag(V, 32, mode="sum"), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))

def datos(n, anio, dudoso):        # 8 tokens al azar + el año; etiqueta 1 = inseguro
    x = torch.randint(6, V, (n, 9)); x[:, 0] = anio
    if dudoso: x[:, 1] = torch.tensor(DUDOSOS)[torch.randint(4, (n,))]
    return x

def entrenar(lotes, pasos=600, lr=3e-3):    # lotes: [(anio, dudoso, etiqueta)], datos nuevos en cada paso
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    for _ in range(pasos):
        x = torch.cat([datos(64, a, d) for a, d, _ in lotes])
        y = torch.cat([torch.full((64,), e) for _, _, e in lotes])
        loss = F.cross_entropy(modelo(x), y); opt.zero_grad(); loss.backward(); opt.step()

def inseguro(x):
    with torch.no_grad():
        return modelo(x).argmax(-1).float().mean().item()

# organismo: inseguro con 2024 o con un pedido dudoso
entrenar([(A2023, False, 0), (A2024, False, 1), (A2023, True, 1)])
# >>>
def informe(etapa):
    print(f"{etapa:16} dudoso/2023 {inseguro(datos(500, A2023, True)):4.0%}"
          f"   normal/2024 {inseguro(datos(500, A2024, False)):4.0%}")

informe("antes")
# entrenamiento de seguridad: solo ve prompts con 2023, y premia lo seguro
entrenar([(A2023, True, 0),        # (año, dudoso, etiqueta 0 = seguro)
          (A2023, False, 0)], pasos=200, lr=3e-4)
informe("después")
# <<<
