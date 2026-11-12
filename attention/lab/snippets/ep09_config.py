from transformers import AutoConfig
# >>>
c = AutoConfig.from_pretrained("google/gemma-2-2b")
print(f"capas {c.num_hidden_layers} · d_model {c.hidden_size} · heads "
      f"{c.num_attention_heads} (de key/value {c.num_key_value_heads})")
print(f"head_dim {c.head_dim} · FFN {c.intermediate_size} · "
      f"activación {c.hidden_activation}")
print(f"vocabulario {c.vocab_size} · RoPE θ = "
      f"{c.rope_parameters['rope_theta']:.0f} · contexto {c.max_position_embeddings}")
print(f"ventana local {c.sliding_window} · soft-cap {c.attn_logit_softcapping}"
      f" y {c.final_logit_softcapping}")

kv = lambda h_kv: 2 * c.num_hidden_layers * h_kv * c.head_dim * 2  # K y V, bf16
print(f"KV cache por token: {kv(4) / 1024:.0f} KiB con 4 heads de K/V,"
      f" {kv(8) / 1024:.0f} KiB con 8")
# <<<
