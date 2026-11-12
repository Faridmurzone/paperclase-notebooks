import torch
torch.manual_seed(0)
# >>>
d, k, r = 2048, 2304, 8            # q_proj de Gemma 2 2B, rango 8
W = torch.randn(d, k)              # congelada (acá, al azar)
B = torch.zeros(d, r)              # LoRA arranca B en cero
A = torch.randn(r, k) * 0.01

print("W + BA == W al inicio:", torch.equal(W + B @ A, W))

B = torch.randn(d, r)              # como si ya hubiera entrenado
delta = B @ A                      # 2048 × 2304
print("forma de BA:", tuple(delta.shape))
print("rango de BA:", torch.linalg.matrix_rank(delta).item())

completa, lora = d * k, r * (d + k)
print(f"números: {completa:,} vs {lora:,}  ({completa / lora:.0f}×)")
# <<<
