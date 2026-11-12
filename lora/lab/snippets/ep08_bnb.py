# ILUSTRATIVO: bitsandbytes necesita una GPU NVIDIA (CUDA) y no está instalado en esta Mac.
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# >>>
cuatro_bits = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",               # NormalFloat de 4 bits
    bnb_4bit_use_double_quant=True,          # cuantiza también los absmax
    bnb_4bit_compute_dtype=torch.bfloat16,   # las cuentas, en 16 bits
)
modelo = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b", quantization_config=cuatro_bits)
modelo = prepare_model_for_kbit_training(modelo)

cfg = LoraConfig(r=8, lora_alpha=16, target_modules="all-linear")
modelo = get_peft_model(modelo, cfg)         # A y B en 16 bits, encima
# <<<
