from transformers import AutoTokenizer
# >>>
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
ids = tok("El paciente tiene fiebre")["input_ids"]
print(ids)
print(tok.convert_ids_to_tokens(ids))
print("vocabulario:", tok.vocab_size)
# <<<
