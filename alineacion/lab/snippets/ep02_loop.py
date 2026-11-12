import torch
from transformers import LlamaConfig, LlamaForCausalLM
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
# corpus sintético: secuencias que cuentan de a 1, 2 o 3 (mod 32) desde un inicio al azar
def lote(n=32, T=16):
    ini, paso = torch.randint(0, 32, (n, 1)), torch.randint(1, 4, (n, 1))
    return (ini + paso * torch.arange(T)) % 32
# >>>
model = LlamaForCausalLM(cfg)                        # pesos al azar
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

for paso in range(301):
    ids = lote()                                     # [32, 16] tokens
    loss = model(input_ids=ids, labels=ids).loss     # entropía cruzada
    loss.backward()                                  # ∇L en cada parámetro
    opt.step()                                       # θ ← θ − η·(paso de Adam)
    opt.zero_grad()
    if paso % 100 == 0:
        print(f"paso {paso:3d}  loss {loss.item():.3f}")
print(f"azar puro: {torch.log(torch.tensor(32.)):.3f}")
# <<<
