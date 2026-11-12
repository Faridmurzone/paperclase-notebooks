import json, os, torch
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
torch.manual_seed(0)
d = k = 256
W0 = torch.randn(d, k) / k**0.5                     # capa congelada
X = torch.randn(512, k)
Y = X @ (W0 + 0.3 * torch.randn(d, k) / k**0.5).T   # la tarea: otro W
# >>>
def primer_paso(r, s, lr=1e-3):                     # s: la escala de BA
    torch.manual_seed(1)
    A = (torch.randn(r, k) / k**0.5).requires_grad_()
    B = torch.zeros(d, r, requires_grad=True)
    opt = torch.optim.Adam([A, B], lr=lr)
    h = X @ W0.T + s * (X @ A.T) @ B.T
    ((h - Y) ** 2).mean().backward()
    opt.step()                                      # un paso de Adam
    return (s * X @ (B @ A).T).norm(dim=1).mean().item()   # |ΔW x|

print(f"r=8, s=2, lr     : {primer_paso(8, 2.0):.4f}")
print(f"r=8, s=1, lr × 2 : {primer_paso(8, 1.0, lr=2e-3):.4f}")
alfa = 8
for r in [1, 4, 16, 64]:
    sin, con = primer_paso(r, 1.0), primer_paso(r, alfa / r)
    print(f"r={r:2}  sin escala {sin:.3f}   con α/r {con:.3f}")
# <<<
rs = [1, 2, 4, 8, 16, 32, 64]
out = {"rs": rs,
       "sin": [round(primer_paso(r, 1.0), 4) for r in rs],
       "con": [round(primer_paso(r, alfa / r), 4) for r in rs],
       "raiz": [round(primer_paso(r, alfa / r**0.5), 4) for r in rs]}
json.dump(out, open(os.path.join(DATOS, "ep03_escala.json"), "w"))
