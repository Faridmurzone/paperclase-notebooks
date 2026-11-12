import torch
# >>>
for x in [3.14159, 40.3, 1500.7]:
    print(f"{x:>8} → bfloat16 {torch.tensor(x).bfloat16().item()}")

def gate(logits, dtype):                  # peso del primer experto de dos
    l = torch.tensor(logits).to(dtype).float()   # redondear y volver
    return l.softmax(-1)[0].item()

for l in ([0.3, 0.1], [40.3, 40.1], [300.3, 300.1]):
    print(f"{l[0]:>5}/{l[1]}: fp32 {gate(l, torch.float32):.3f}"
          f" · bf16 {gate(l, torch.bfloat16):.3f}")
# <<<
