import os, sys
import torch
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _resnet import ViTChica, RedCifar, contar
torch.manual_seed(0)
x = torch.randn(2, 3, 32, 32)
# >>>
vit = ViTChica(patch=4, d=128, capas=6, heads=4)
t = vit.patch_embed(x).flatten(2).transpose(1, 2)
print("patch embedding:", tuple(t.shape))          # 64 tokens de 128
z = vit.tokens(x)                                  # + CLS + posiciones
print("con CLS:        ", tuple(z.shape))
z = vit.encoder(z)                                 # 6 bloques transformer
print("después:        ", tuple(z.shape))
print("logits del CLS: ", tuple(vit.fc(vit.norm(z[:, 0])).shape))
print("parámetros ViT:", contar(vit), "| ResNet-20:", contar(RedCifar(n=3, residual=True)))
# <<<
