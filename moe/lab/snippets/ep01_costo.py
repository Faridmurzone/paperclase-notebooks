total, activos = 1_334_625_280, 428_655_616   # los imprime ep01_granite
# >>>
bytes_bf16 = 2                                 # un número en bfloat16
memoria = total * bytes_bf16                   # hay que tener TODO cargado
flops_moe = 2 * activos                        # por token: 1 mult + 1 suma
flops_denso = 2 * total                        # un denso del mismo tamaño
print(f"memoria de pesos:  {memoria / 1e9:.2f} GB")
print(f"FLOPs/token MoE:   {flops_moe / 1e9:.2f} G")
print(f"FLOPs/token denso: {flops_denso / 1e9:.2f} G")
print(f"el MoE hace {flops_moe / flops_denso:.0%} de las cuentas")
# <<<
