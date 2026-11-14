import os, json
AQUI = os.path.dirname(os.path.abspath(__file__))
h7 = json.load(open(f"{AQUI}/../datos/ep07_hist.json"))
h8 = json.load(open(f"{AQUI}/../datos/ep08_hist.json"))
# >>>
corridas = {"sin BN, lr 0,02": h8["sinbn_lr002"], "con BN, lr 0,02": h8["conbn_lr002"],
            "sin BN, lr 0,1": h8["sinbn_lr01"], "con BN, lr 0,1": h7["plana8"]}
print("red plana de 8 capas   test ep. 1   ep. 5   ep. 30")
for nombre, c in corridas.items():
    t = [e["acc_test"] for e in c["hist"]]
    print(f"{nombre:22} {t[0]:8.1%} {t[4]:7.1%} {t[-1]:7.1%}")
# <<<
