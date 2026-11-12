import json, math, os
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.utils import logging; logging.set_verbosity_error()
torch.manual_seed(0)
L, F, SEP, EOS, S0 = 6, 3, 1, 2, 7
def ejemplos(n):
    x = torch.randint(S0, S0 + 10, (n, L))
    coment = x.gather(1, torch.randint(0, L, (n, F)))
    resp = torch.cat([coment, x.sort(1).values, torch.full((n, 1), EOS)], 1)
    ids = torch.cat([x, torch.full((n, 1), SEP), resp], 1)
    labels = ids.clone(); labels[:, :L + 1] = -100
    return ids, labels
tr, tr_lab = ejemplos(200)                     # 200 demostraciones
va, va_lab = ejemplos(1000)                    # validación
cfg = GPT2Config(vocab_size=S0 + 10, n_positions=32, n_embd=128, n_layer=4, n_head=4, resid_pdrop=0.2,
                 embd_pdrop=0.0, attn_pdrop=0.0, bos_token_id=EOS, eos_token_id=EOS)
model = GPT2LMHeadModel(cfg)
B, EPOCHS, lr = 8, 16, 1e-3
opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * math.ceil(200 / B), eta_min=0.1 * lr)

@torch.no_grad()
def metrica(model):                            # ¿ordenó bien? (la parte que importa)
    x = va[:, :L + 1]
    for _ in range(F + L):
        x = torch.cat([x, model(input_ids=x).logits[:, -1].argmax(-1, keepdim=True)], 1)
    return (x[:, L + 1 + F:] == va[:, L + 1 + F:L + 1 + F + L]).all(1).float().mean().item()
hist = []
# >>>
print("epoch   train    val  ordena")
for epoch in range(1, EPOCHS + 1):
    model.train()
    for i in torch.randperm(200).split(B):        # una pasada = un epoch
        loss = model(input_ids=tr[i], labels=tr_lab[i]).loss
        loss.backward(); opt.step(); opt.zero_grad(); sched.step()
    model.eval()
    with torch.no_grad():
        l_tr = model(input_ids=tr, labels=tr_lab).loss.item()
        l_va = model(input_ids=va, labels=va_lab).loss.item()
    acc = metrica(model)                          # proxy del RM score
    hist.append((epoch, l_tr, l_va, acc))
    if epoch in (1, 4, 7, 10, 13, 16):
        print(f"{epoch:5d} {l_tr:7.2f} {l_va:6.2f} {acc:7.0%}")
# <<<
json.dump(hist, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep03_sft.json"), "w"))
