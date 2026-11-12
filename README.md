# Paperclase · notebooks

Los notebooks de las series de video de Paperclase: el guion de cada episodio en texto, con cada bloque de código y su salida real.

**Para correrlos no hace falta instalar nada:** tocá *Abrir en Colab*, ejecutá la primera celda (*Antes de empezar*) y seguí en orden. Corre con la cuota gratuita de Google Colab, en tu cuenta. Para conservar tus cambios: **Archivo → Guardar una copia en Drive**.

- Los episodios marcados con *Gemma* usan [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b), un modelo con licencia: aceptala en Hugging Face y cargá tu token como secreto `HF_TOKEN` en Colab. Los pesos no están en este repo; cada persona los baja con su cuenta.
- El resto corre en CPU en segundos o pocos minutos. QLoRA (LoRA, episodio 8) necesita GPU: *Entorno de ejecución → Cambiar tipo de entorno → T4*.
- Fuera de Colab: `git clone https://github.com/Faridmurzone/paperclase-notebooks`, `pip install -r <serie>/requirements.txt` y abrí el notebook con Jupyter.

## Attention is all you need

El paper del transformer, sección por sección: scaled dot-product attention, multi-head, positional encoding, encoder-decoder y cómo se entrenó. Implementado desde cero y comparado con Gemma.

| Episodio | Notebook | Gemma (token HF) |
|---|---|---|
| 01 · Antes del transformer | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/attention/notebooks/ep01.ipynb) | no |
| 02 · Scaled dot-product attention | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/attention/notebooks/ep02.ipynb) | no |
| 03 · Multi-head attention | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/attention/notebooks/ep03.ipynb) | sí |
| 04 · Positional encoding | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/attention/notebooks/ep04.ipynb) | no |
| 05 · Encoder y decoder | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/attention/notebooks/ep05.ipynb) | no |
| 06 · Feed-forward, residuales y LayerNorm | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/attention/notebooks/ep06.ipynb) | no |
| 07 · Cómo se entrenó | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/attention/notebooks/ep07.ipynb) | no |
| 08 · Resultados y ablaciones | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/attention/notebooks/ep08.ipynb) | no |
| 09 · Del transformer de 2017 a Gemma | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/attention/notebooks/ep09.ipynb) | sí |
| 10 · Implementarlo entero | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/attention/notebooks/ep10.ipynb) | no |

## Alineación a bajo nivel

Cómo se alinea un modelo de lenguaje, mirando el código y las ecuaciones de verdad: del tensor que sale del tokenizador a los hooks de PyTorch que leen y editan activaciones, y a cómo se supervisa un modelo que sabe más que vos.

| Episodio | Notebook | Gemma (token HF) |
|---|---|---|
| 01 · El modelo por dentro | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/alineacion/notebooks/ep01.ipynb) | sí |
| 02 · Cómo aprende | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/alineacion/notebooks/ep02.ipynb) | sí |
| 03 · Modelos de recompensa | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/alineacion/notebooks/ep03.ipynb) | no |
| 04 · RLHF con PPO | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/alineacion/notebooks/ep04.ipynb) | no |
| 05 · DPO y la constitución | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/alineacion/notebooks/ep05.ipynb) | no |
| 06 · Qué sale mal | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/alineacion/notebooks/ep06.ipynb) | sí |
| 07 · Hooks: mirar y tocar activaciones | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/alineacion/notebooks/ep07.ipynb) | sí |
| 08 · Direcciones: probes y steering | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/alineacion/notebooks/ep08.ipynb) | sí |
| 09 · Superposición, SAEs y circuitos | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/alineacion/notebooks/ep09.ipynb) | sí |
| 10 · Supervisar lo que no podés verificar | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/alineacion/notebooks/ep10.ipynb) | no |

## LoRA

Ajustar un modelo grande entrenando dos matrices chicas: la fórmula, por qué alcanza con rango bajo, dónde se pone, peft por dentro, QLoRA y variantes. Implementado desde cero y verificado contra peft.

| Episodio | Notebook | Gemma (token HF) |
|---|---|---|
| 01 · Por qué no ajustar todo | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/lora/notebooks/ep01.ipynb) | sí |
| 02 · Rango intrínseco bajo | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/lora/notebooks/ep02.ipynb) | no |
| 03 · La fórmula | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/lora/notebooks/ep03.ipynb) | sí |
| 04 · Dónde ponerlo | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/lora/notebooks/ep04.ipynb) | sí |
| 05 · LoRA desde cero | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/lora/notebooks/ep05.ipynb) | no |
| 06 · peft por dentro | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/lora/notebooks/ep06.ipynb) | sí |
| 07 · Fusionar y servir | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/lora/notebooks/ep07.ipynb) | no |
| 08 · QLoRA | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/lora/notebooks/ep08.ipynb) | sí |
| 09 · Variantes | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/lora/notebooks/ep09.ipynb) | no |
| 10 · Qué aprende y qué olvida | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/lora/notebooks/ep10.ipynb) | no |

## InstructGPT

El paper que convirtió a GPT-3 en un modelo que sigue instrucciones, de punta a punta: datos, SFT, reward model, PPO y PPO-ptx, y cómo se evaluó. Cada paso reimplementado en chico.

| Episodio | Notebook | Gemma (token HF) |
|---|---|---|
| 01 · El problema | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/instructgpt/notebooks/ep01.ipynb) | sí |
| 02 · Los datos | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/instructgpt/notebooks/ep02.ipynb) | no |
| 03 · Paso 1: SFT | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/instructgpt/notebooks/ep03.ipynb) | no |
| 04 · Paso 2: el reward model | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/instructgpt/notebooks/ep04.ipynb) | no |
| 05 · Paso 3: PPO y PPO-ptx | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/instructgpt/notebooks/ep05.ipynb) | no |
| 06 · Cómo se evaluó | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/instructgpt/notebooks/ep06.ipynb) | no |
| 07 · El impuesto de alineación | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/instructgpt/notebooks/ep07.ipynb) | no |
| 08 · Los límites del paper | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/instructgpt/notebooks/ep08.ipynb) | no |
| 09 · El pipeline entero, en chico | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/instructgpt/notebooks/ep09.ipynb) | no |
| 10 · Lo que vino después | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/instructgpt/notebooks/ep10.ipynb) | no |

## Álgebra lineal para LLMs

Todos los conceptos matemáticos detrás de las fórmulas de un LLM: vectores, matrices, normas, proyecciones, SVD, softmax, probabilidad, gradientes y optimización. Cada uno con su dibujo y su línea de PyTorch.

| Episodio | Notebook | Gemma (token HF) |
|---|---|---|
| 01 · Vectores y producto punto | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/algebra/notebooks/ep01.ipynb) | sí |
| 02 · Matrices: fila por columna | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/algebra/notebooks/ep02.ipynb) | sí |
| 03 · Transpuesta, producto exterior y rango | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/algebra/notebooks/ep03.ipynb) | no |
| 04 · Normas y normalización | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/algebra/notebooks/ep04.ipynb) | sí |
| 05 · Proyecciones y bases ortogonales | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/algebra/notebooks/ep05.ipynb) | sí |
| 06 · Autovalores, SVD y rango bajo | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/algebra/notebooks/ep06.ipynb) | sí |
| 07 · Exponencial, logaritmo y softmax | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/algebra/notebooks/ep07.ipynb) | no |
| 08 · Probabilidad: esperanza, entropía y KL | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/algebra/notebooks/ep08.ipynb) | no |
| 09 · Gradientes y la regla de la cadena | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/algebra/notebooks/ep09.ipynb) | no |
| 10 · Optimización | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/algebra/notebooks/ep10.ipynb) | no |

## Deep learning a secas

Las bases de una red neuronal, sin transformers: neurona, capa, activación, pérdida, backpropagation y el loop de entrenamiento, con un autograd escrito desde cero y cada fórmula dibujada.

| Episodio | Notebook | Gemma (token HF) |
|---|---|---|
| 01 · Una neurona | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/deeplearning/notebooks/ep01.ipynb) | no |
| 02 · De neurona a capa | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/deeplearning/notebooks/ep02.ipynb) | no |
| 03 · Por qué hace falta no linealidad | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/deeplearning/notebooks/ep03.ipynb) | no |
| 04 · Pérdidas | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/deeplearning/notebooks/ep04.ipynb) | no |
| 05 · Backpropagation a mano | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/deeplearning/notebooks/ep05.ipynb) | no |
| 06 · Un autograd desde cero | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/deeplearning/notebooks/ep06.ipynb) | no |
| 07 · El training loop | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/deeplearning/notebooks/ep07.ipynb) | no |
| 08 · Regularización | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/deeplearning/notebooks/ep08.ipynb) | no |
| 09 · Normalización | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/deeplearning/notebooks/ep09.ipynb) | no |
| 10 · Profundidad | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/deeplearning/notebooks/ep10.ipynb) | no |

## Optimizadores, con foco en AdamW

De descenso por gradiente a AdamW, una idea por episodio: momentum, learning rates adaptativos, los dos momentos de Adam, weight decay desacoplado, schedules y estabilidad. Cada optimizador escrito a mano y comparado con torch.optim.

| Episodio | Notebook | Gemma (token HF) |
|---|---|---|
| 01 · Descenso por gradiente | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/optimizadores/notebooks/ep01.ipynb) | no |
| 02 · SGD con minibatches | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/optimizadores/notebooks/ep02.ipynb) | no |
| 03 · Momentum | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/optimizadores/notebooks/ep03.ipynb) | no |
| 04 · Tasas adaptativas | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/optimizadores/notebooks/ep04.ipynb) | no |
| 05 · Adam | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/optimizadores/notebooks/ep05.ipynb) | no |
| 06 · Weight decay no es L2 | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/optimizadores/notebooks/ep06.ipynb) | no |
| 07 · AdamW | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/optimizadores/notebooks/ep07.ipynb) | no |
| 08 · Schedules | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/optimizadores/notebooks/ep08.ipynb) | no |
| 09 · Estabilidad | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/optimizadores/notebooks/ep09.ipynb) | no |
| 10 · Después de AdamW | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/optimizadores/notebooks/ep10.ipynb) | no |

## Mixture of Experts

Modelos con muchos expertos y un router que elige pocos por token: el gating top-k, el balanceo de carga, expertos finos y compartidos, y cuánto cuesta de verdad. Implementado desde cero y mirado por dentro en un MoE abierto.

| Episodio | Notebook | Gemma (token HF) |
|---|---|---|
| 01 · Muchos parámetros, poco cómputo | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/moe/notebooks/ep01.ipynb) | no |
| 02 · El router | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/moe/notebooks/ep02.ipynb) | no |
| 03 · Una capa MoE desde cero | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/moe/notebooks/ep03.ipynb) | no |
| 04 · El router que colapsa | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/moe/notebooks/ep04.ipynb) | no |
| 05 · Capacidad y tokens descartados | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/moe/notebooks/ep05.ipynb) | no |
| 06 · Switch y la estabilidad | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/moe/notebooks/ep06.ipynb) | no |
| 07 · Expertos finos y compartidos | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/moe/notebooks/ep07.ipynb) | no |
| 08 · Balancear sin pérdida auxiliar | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/moe/notebooks/ep08.ipynb) | no |
| 09 · Qué aprende cada experto | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/moe/notebooks/ep09.ipynb) | no |
| 10 · El costo de verdad | [Abrir en Colab](https://colab.research.google.com/github/Faridmurzone/paperclase-notebooks/blob/main/moe/notebooks/ep10.ipynb) | no |
