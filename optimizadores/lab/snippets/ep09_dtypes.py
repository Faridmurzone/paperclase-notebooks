import torch

# >>>
for dt in [torch.float32, torch.float16, torch.bfloat16]:
    fi = torch.finfo(dt)
    print(f"{str(dt)[6:]:8}  bits {fi.bits}  máx {fi.max:.6g}  eps {fi.eps:.3g}"
          f"  mín {fi.tiny:.3g}")
x = torch.tensor([70000.0, 1.001, 1e-6])
print("float16 :", [f"{v:.6g}" for v in x.to(torch.float16).tolist()])
print("bfloat16:", [f"{v:.6g}" for v in x.to(torch.bfloat16).tolist()])
# <<<
