import torch
torch.manual_seed(0)
L, F, SEP, EOS, S0 = 6, 3, 1, 2, 7          # 6 símbolos, 3 de comentario; ids 7..16 = símbolos
# >>>
def ejemplos(n):                     # tarea de juguete: ordenar 6 símbolos
    x = torch.randint(S0, S0 + 10, (n, L))                # el prompt
    coment = x.gather(1, torch.randint(0, L, (n, F)))     # estilo libre:
    resp = torch.cat([coment, x.sort(1).values,           # muchas válidas
                      torch.full((n, 1), EOS)], 1)
    ids = torch.cat([x, torch.full((n, 1), SEP), resp], 1)
    labels = ids.clone()
    labels[:, :L + 1] = -100          # el prompt no entra en la pérdida
    return ids, labels

ids, labels = ejemplos(1)
nombre = lambda i: {1: "SEP", 2: "EOS"}.get(i, str(i))
print("prompt:", *map(nombre, ids[0, :L + 1].tolist()))
print("resp:  ", *map(nombre, ids[0, L + 1:].tolist()))
print("labels del prompt:", *labels[0, :3].tolist(), "...")
print("entrenan", int((labels != -100).sum()), "de", ids.shape[1])
# <<<
