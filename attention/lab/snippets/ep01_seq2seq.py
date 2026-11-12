import torch
torch.manual_seed(0)
vocab, d = 10, 16
# >>>
emb = torch.nn.Embedding(vocab, d)
encoder = torch.nn.GRU(d, d, batch_first=True)

corta = torch.randint(0, vocab, (1, 5))     # 5 tokens
larga = torch.randint(0, vocab, (1, 50))    # 50 tokens
for x in (corta, larga):
    _, h = encoder(emb(x))                  # todo lo leído, en h
    print(f"{x.shape[1]:2} tokens → contexto de forma {tuple(h.shape)}")
# <<<
