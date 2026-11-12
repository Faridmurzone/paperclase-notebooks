import torch
from transformers import LlamaConfig, LlamaForCausalLM
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
def lote(n=32, T=16):
    ini, paso = torch.randint(0, 32, (n, 1)), torch.randint(1, 4, (n, 1))
    return (ini + paso * torch.arange(T)) % 32
model = LlamaForCausalLM(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
for paso in range(301):                              # el mismo loop de la escena anterior
    ids = lote(); model(input_ids=ids, labels=ids).loss.backward(); opt.step(); opt.zero_grad()
model.eval()
# >>>
def continuar(inicio, n=6):
    x = torch.tensor([inicio])
    out = model.generate(x, max_new_tokens=n, do_sample=False,
                         pad_token_id=0)   # greedy: el más probable
    return out[0].tolist()

print(continuar([3, 6]))                             # ¿de a tres?
print(continuar([27, 29]))                           # ¿y da la vuelta en 32?
with torch.no_grad():
    p = model(torch.tensor([[5]])).logits[0, -1].softmax(-1)
top = p.topk(3)
print("después de [5]:", [(int(i), round(v.item(), 2)) for v, i in zip(*top)])
# <<<
