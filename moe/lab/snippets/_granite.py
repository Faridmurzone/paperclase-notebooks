"""Granite 3.1 1B-A400M (MoE abierto, 32 expertos, top-8, 24 capas) y textos sintéticos para los episodios 9 y 10.
    from _granite import cargar, TEXTOS, capturar_ruteo"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
torch.set_num_threads(4)
NOMBRE = "ibm-granite/granite-3.1-1b-a400m-base"

# Textos sintéticos y neutros, cuatro tipos (escritos para la serie, no salen de ningún corpus).
TEXTOS = {
    "cocina": [
        "Para el pan casero se mezcla harina, agua, sal y levadura, y se amasa durante diez minutos.",
        "La salsa de tomate lleva cebolla picada, ajo y un poco de albahaca; se cocina a fuego bajo.",
        "Antes de hornear la tarta, se pincha la masa con un tenedor para que no se infle en el horno.",
        "El arroz se lava con agua fría, se tuesta en la olla con aceite y después se agrega el caldo caliente.",
        "Para que las verduras queden crocantes, se saltean en la sartén a fuego fuerte durante pocos minutos.",
        "La mermelada de durazno se hace con fruta madura, azúcar y jugo de limón, revolviendo sin parar.",
    ],
    "astronomia": [
        "Júpiter es el planeta más grande del sistema solar y tiene decenas de lunas que giran a su alrededor.",
        "Las estrellas nacen en nubes de gas y polvo que se contraen por su propia gravedad durante millones de años.",
        "Desde el hemisferio sur se ve la Cruz del Sur, una constelación chica pero muy brillante.",
        "Un eclipse de Luna ocurre cuando la Tierra se interpone entre el Sol y la Luna y le hace sombra.",
        "Los cometas tienen un núcleo de hielo y roca; cuando se acercan al Sol, el calor les forma una cola.",
        "La luz del Sol tarda unos ocho minutos en llegar a la Tierra, y la de otras estrellas, años.",
    ],
    "prosa": [
        "El tren salió temprano de la estación y cruzó el valle mientras el sol subía detrás de las montañas.",
        "La biblioteca del barrio abre los martes y los jueves; los chicos van a leer cuentos después de la escuela.",
        "Una receta sencilla: hervir el agua, agregar la pasta, esperar diez minutos y servir con aceite y queso.",
        "The river runs slowly through the old town, and in the evening people walk along the stone bridge.",
        "Every morning the baker opens the shop, sweeps the floor, and puts fresh bread in the window.",
    ],
    "codigo": [
        "def suma(a, b):\n    return a + b\n\nfor i in range(10):\n    print(suma(i, i))\n",
        "class Pila:\n    def __init__(self):\n        self.items = []\n\n    def push(self, x):\n        self.items.append(x)\n",
        "import numpy as np\n\nx = np.zeros((3, 4))\nfor fila in x:\n    fila[0] = 1.0\nprint(x.sum())\n",
        "if __name__ == \"__main__\":\n    datos = [3, 1, 2]\n    datos.sort()\n    print(datos)\n",
    ],
    "numeros": [
        "12 + 35 = 47; 120 - 18 = 102; 7 * 8 = 56; 81 / 9 = 9; 3.14 * 2 = 6.28",
        "2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026. 10, 20, 30, 40, 50, 60, 70.",
        "x = 0.25, y = 0.75, z = 1.5; total = 2.5 y promedio = 0.8333.",
        "1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987.",
    ],
    "puntuacion": [
        "¡Hola! ¿Qué tal? (bien, gracias...) -- sí; no: quizás. [a] {b} <c> \"d\" 'e' ¿¡?!",
        "... ; : , . ! ? ( ) [ ] { } -- ** // ## @@ && || :: ;; ,, .. !! ??",
        "«Uno», «dos» y «tres»; (cuatro) [cinco] {seis}: ¿siete? ¡ocho! — nueve, diez...",
    ],
}


def cargar():
    tok = AutoTokenizer.from_pretrained(NOMBRE)
    model = AutoModelForCausalLM.from_pretrained(NOMBRE, dtype=torch.float32)
    model.eval()
    return tok, model


def capturar_ruteo(model):
    """Cuelga un hook en el router de cada capa. Devuelve (dict capa → lista de idx [T, 8], lista de handles)."""
    elegidos = {}
    hs = []
    for c, capa in enumerate(model.model.layers):
        def hook(m, entrada, salida, c=c):
            logits, pesos, idx = salida
            elegidos.setdefault(c, []).append(idx.clone())
        hs.append(capa.block_sparse_moe.router.register_forward_hook(hook))
    return elegidos, hs
