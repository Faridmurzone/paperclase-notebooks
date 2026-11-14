# >>>
capas = [("conv1", 3, 1), ("conv2", 3, 1), ("pool1", 2, 2),
         ("conv3", 3, 1), ("conv4", 3, 1), ("pool2", 2, 2),
         ("conv5", 3, 1), ("conv6", 3, 1)]       # (nombre, k, stride)
r, j = 1, 1                       # un píxel ve 1 píxel; salto 1
for nombre, k, s in capas:
    r = r + (k - 1) * j           # cuánto agrega este kernel
    j = j * s                     # distancia entre vecinos, en píxeles
    print(f"{nombre}: campo {r}×{r}, salto {j}")
# <<<
