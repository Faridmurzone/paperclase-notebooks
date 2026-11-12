import torch.nn as nn
# >>>
tr = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6,
                    num_decoder_layers=6, dim_feedforward=2048)
enc, dec = tr.encoder.layers[0], tr.decoder.layers[0]
print("encoder:", len(tr.encoder.layers), "capas · decoder:",
      len(tr.decoder.layers), "capas")
for nombre, capa in [("capa del encoder", enc), ("capa del decoder", dec)]:
    partes = [n for n, _ in capa.named_children()
              if "attn" in n or n.startswith("linear")]
    print(f"{nombre}:", ", ".join(partes))
total = sum(p.numel() for p in tr.parameters())
print(f"parámetros (sin embeddings): {total / 1e6:.1f} M")
# <<<
