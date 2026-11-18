# Demo: Herramientas para experimentar con LLMs en un servidor de alta capacidad

17 de noviembre de 2026 · Dante

## Objetivo y alcance

Una demo introductoria de unos 20 minutos para que el curso se familiarice con las librerías principales para correr, guardar y evaluar experimentos con LLMs en una máquina Linux con GPU. Es una **primera familiarización básica**: muestra cada herramienta funcionando una vez, sin profundizar en optimización ni en configuración avanzada. Sirve para cualquier paper, y el código detecta el hardware disponible sin asumir un modelo de GPU.

Todavía no se sabe cómo funcionará el servidor del curso (GPUs, sistema de colas, permisos, acceso a internet). Por eso la demo se desarrolla y se presenta en local, y lo propio de servidores compartidos se trata solo como recomendaciones generales (ver *Supuestos sobre el servidor*).

El paper *Optimal Self-Consistency for Efficient Reasoning with Large Language Models* (arXiv 2511.12309) aporta solo el escenario. Los autores generan 100 respuestas por pregunta (GSM8K, MATH, MMLU, GPQA-Diamond; temperaturas 0.4 a 1.0) y luego simulan distintos presupuestos de Self-Consistency a partir de esas muestras guardadas. Las herramientas que citan son **PyTorch, vLLM y LM Eval Harness**. La demo usa esas mismas herramientas más **transformers**, a escala de juguete: 10 a 20 preguntas de GSM8K y 5 muestras por pregunta.

Queda fuera del alcance: el simulador de Self-Consistency, Blend-ASC y los otros algoritmos del paper, las leyes de escalado y la replicación de sus tablas. Lo único que se toma del paper es su patrón de trabajo: **generar una vez, guardar todo y analizar muchas veces** (en la demo, un voto de mayoría sobre las muestras guardadas).

### Entorno de desarrollo y modelo

- **Máquina local:** RTX 2060 (6 GB, arquitectura Turing, compute capability 7.5) con Windows. vLLM no corre en Windows, así que todo se ejecuta dentro de **WSL2 (Ubuntu)**, lo que además se parece a un servidor Linux.
- **Python y entorno:** el Python del sistema (3.14 en Ubuntu 26.04) no se usa, porque es probable que vLLM aún no lo soporte. uv instala y fija **Python 3.12** (`uv python pin 3.12`), y las dependencias quedan en `pyproject.toml` + `uv.lock`. vLLM fija una versión exacta de torch, así que se elige primero vLLM, se verifica que esa versión soporte GPUs Turing en float16 y se deja que torch se derive de ella.
- **Modelo:** `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB en float16, sin acceso restringido, misma familia que los modelos Qwen del paper). Si falta memoria, `Qwen/Qwen2.5-0.5B-Instruct`. En otra máquina se cambia solo el nombre del modelo en `config.yaml`.
- **Precisión:** Turing no soporta bfloat16, así que se usa float16. El código elige el `dtype` según la GPU detectada.
- **Memoria de GPU:** Windows ya ocupa ~1,2 GB de los 6 GB, así que quedan ~4,9 GB libres. En vLLM se limita `max_model_len` (~2048), se usa `gpu_memory_utilization` ≈ 0.70 (con 0.85 vLLM no arranca) y se activa `enforce_eager=True` para no reservar memoria extra en CUDA graphs. El script del Módulo 0 sugiere este valor a partir de la VRAM libre.
- **Memoria RAM:** WSL tiene 7,7 GB por defecto, lo que puede quedar justo para cargar el modelo en vLLM. Si hace falta, se sube el límite en `.wslconfig` (por ejemplo, `memory=12GB` y algo de swap).

### Supuestos sobre el servidor

Mientras no se conozca el servidor, la demo supone solo lo mínimo: Linux, GPU NVIDIA con drivers instalados y permiso para crear un entorno en el directorio del usuario. Lo demás se presenta como "si el servidor lo permite":

- elegir GPUs con `CUDA_VISIBLE_DEVICES` (si no hay un gestor de colas como Slurm que las asigne);
- dejar procesos corriendo con tmux;
- descargar del Hub directamente (si no hay internet, se descarga antes y se copia la caché);
- usar varias GPUs (`device_map="auto"`, `tensor_parallel_size`), que se menciona pero no se muestra.

## Parte 1: Estructura de la demo (~20 min)

Son 8 bloques. Cada uno presenta una herramienta, explica para qué sirve y la muestra funcionando sobre el mismo ejemplo: 10 preguntas de GSM8K y un modelo pequeño.

```mermaid
flowchart LR
    A["<b>1. Modelos y datos</b><br/>huggingface_hub<br/>datasets<br/>caché en HF_HOME"]
    B["<b>2. Inferencia</b><br/>transformers: explorar<br/>vLLM: generar en lote<br/>(sobre PyTorch)"]
    C["<b>3. Almacenamiento</b><br/>pandas: JSONL, Parquet<br/>config + semilla + versiones"]
    D["<b>4. Evaluación</b><br/>LM Eval Harness<br/>backends hf y vllm<br/>maj@k"]
    A --> B --> C --> D
    style B fill:#e7f1fb,stroke:#2f6fd6,stroke-width:2px
```

*Cada etapa tiene su herramienta; en inferencia se elige entre dos motores.* Los bloques 3 a 7 recorren estas cuatro etapas en orden; la inferencia ocupa dos bloques porque se comparan ambos motores.

1. **Mapa de herramientas (1 min)**
   - Mensaje: cada etapa de un experimento con LLMs tiene su herramienta, y en la inferencia hay varias opciones con distintos compromisos.
   - Mostrar: el diagrama de arriba y el escenario del paper (100 respuestas por pregunta, guardadas y luego analizadas con distintos presupuestos).
2. **Reconocer la máquina (2 min)**
   - Mensaje: antes de correr nada, conviene saber qué hay; en una máquina compartida, además, no pisar el trabajo de otros.
   - Herramientas: `nvidia-smi` para ver GPUs y uso; PyTorch (`torch.cuda.device_count()`, `get_device_capability()`, `mem_get_info()`) para detectar GPUs, VRAM libre y elegir el `dtype`; uv para el entorno con versiones fijadas.
   - Mención: `nvitop`, `CUDA_VISIBLE_DEVICES` y tmux como buenas prácticas en un servidor compartido (en WSL, `nvitop` no ve los procesos por GPU).
3. **Modelos y datos desde Hugging Face (2 min)**
   - Mensaje: el Hub es la fuente común de modelos y datasets, y su caché evita descargar dos veces.
   - Herramientas: `huggingface_hub` (login, `snapshot_download`, variable `HF_HOME`) y `datasets` (`load_dataset("openai/gsm8k", "main")`).
   - Nota: los modelos Llama tienen acceso restringido y requieren aceptar la licencia; por eso la demo usa Qwen2.5-Instruct.
4. **Inferencia con transformers (3 min)**
   - Mensaje: es la opción más flexible y la más cercana al modelo, ideal para explorar y depurar.
   - Herramientas: `pipeline("text-generation")` y `AutoModelForCausalLM.generate()` con el chat template del modelo.
   - Mostrar primero el ejemplo básico (`02a`): 3 prompts fijos, el chat template y la respuesta, en un script corto que se lee completo en pantalla.
   - Luego (`02b`): responder las 10 preguntas de GSM8K con una respuesta cada una y medir tokens por segundo.
   - Mención: con varias GPUs, `device_map="auto"` (accelerate) reparte un modelo grande entre GPUs.
5. **Inferencia con vLLM (4 min)**
   - Mensaje: para generar mucho, un motor de inferencia agrupa peticiones y aprovecha mejor la GPU.
   - Herramientas: `LLM` y `SamplingParams(n=5, temperature=0.8, seed=...)` para generación en lote.
   - Mostrar primero el ejemplo básico (`03a`): los mismos 3 prompts y parámetros que `02a`, para comparar las dos APIs lado a lado.
   - Luego (`03b`): las mismas 10 preguntas de GSM8K con 5 respuestas cada una, y comparar tokens por segundo con el bloque anterior.
   - Mención: `tensor_parallel_size` si el modelo no cabe en una GPU; `vllm serve` levanta una API compatible con OpenAI.
6. **Guardar las salidas (3 min)**
   - Mensaje: guardar todas las generaciones crudas junto con lo necesario para repetir la corrida, porque el análisis se hace después y sin volver a llamar al modelo.
   - Herramientas: pandas para escribir JSONL o Parquet, con una fila por muestra; un archivo con la config, la semilla, el prompt, la revisión del modelo y las versiones de las librerías.
   - Mostrar: releer el archivo y calcular accuracy y maj@k para k = 1…5 sin regenerar nada.
7. **Evaluar con LM Eval Harness (4 min)**
   - Mensaje: una herramienta estándar fija prompts, extracción de respuestas y métricas, para que los números sean comparables entre papers.
   - Herramientas: `lm_eval` con `--model hf` y `--model vllm`, `--tasks gsm8k`, `--limit 20`, `--output_path` y `--log_samples`; `--apply_chat_template` para modelos Instruct. Para Self-Consistency, una copia de la tarea `gsm8k_cot_self_consistency` con menos repeticiones, escrita como YAML propio.
   - Mostrar: la misma evaluación con los dos backends, el archivo de muestras que deja el harness y la métrica maj@k de la tarea propia.
8. **Cierre: cuándo usar qué (1 min)**
   - Mostrar: la tabla del mapa de herramientas.

Para que la demo no dependa de la red ni del tiempo de carga, el modelo y el dataset se descargan antes, y todo corre con 10 a 20 preguntas.

## Mapa de herramientas: cuándo usar cada una

La regla práctica: transformers para explorar, vLLM para generar en volumen, guardar todo lo generado y LM Eval Harness para reportar.

| Herramienta | Etapa | Para qué sirve | Cuándo conviene | Límite | En la demo |
| --- | --- | --- | --- | --- | --- |
| `nvidia-smi` | Máquina | Ver GPUs, VRAM y uso | Siempre antes de lanzar algo | Solo observa, no reserva recursos | Sí |
| `nvitop`, tmux, `CUDA_VISIBLE_DEVICES` | Servidor | Monitorear procesos, mantenerlos vivos y elegir GPUs | En un servidor compartido (depende de cómo funcione) | `nvitop` no ve procesos en WSL; un gestor de colas puede reemplazarlos | Mención |
| uv | Máquina | Entorno con Python y versiones fijadas (`uv.lock`) | Siempre | La versión de torch (y su CUDA) la dicta vLLM | Sí |
| PyTorch | Base | Tensores en GPU; base de transformers y vLLM | Detectar GPUs y elegir `dtype` | Rara vez se usa directo para generar | Sí |
| `huggingface_hub` | Modelos y datos | Login, descarga y caché de modelos | Siempre | Modelos restringidos requieren token y licencia | Sí |
| `datasets` | Modelos y datos | Cargar y transformar benchmarks | Cualquier dataset del Hub o local | Columnas y splits cambian entre datasets | Sí |
| `transformers` | Inferencia | Cargar y ejecutar modelos con acceso a logits y capas | Explorar, depurar, pocas generaciones | Lento para generar en volumen | Sí |
| vLLM | Inferencia | Generación en lote y servidor con API tipo OpenAI | Miles de generaciones o varias muestras por pregunta | Menos acceso a internals del modelo; solo Linux | Sí |
| pandas (JSONL, Parquet) | Almacenamiento | Guardar generaciones y metadatos | Siempre | No versiona ni registra corridas | Sí |
| LM Eval Harness | Evaluación | Benchmarks estándar con prompts y métricas fijas | Reportar resultados comparables | Tareas propias requieren escribir un YAML | Sí |
| `accelerate` | Inferencia | Repartir un modelo entre GPUs y CPU (`device_map`) | El modelo no cabe en una GPU | Reparte memoria, no acelera la generación | Mención |
| `bitsandbytes` | Inferencia | Cuantizar a 8 o 4 bits | Poca VRAM disponible | Puede perder calidad | Opcional |
| SGLang, TGI | Inferencia | Alternativas a vLLM para servir modelos | Comparar motores o si el servidor ya usa uno | No se usan en la demo | Mención |
| Weights & Biases, MLflow | Seguimiento | Registrar configuración, métricas y archivos por corrida | Muchas corridas o trabajo en equipo | W&B requiere cuenta; MLflow hay que alojarlo | Opcional |

## Parte 2: Plan de desarrollo de código interactivo

Son 6 módulos cortos que cubren los bloques 2 a 7 de la demo, más uno opcional. Cada script funciona solo, importa las utilidades compartidas de `common.py`, lee `config.yaml` y cierra con una validación visible que sirve para mostrar en la presentación.

| # | Módulo | Qué construyes | Concepto que aprendes | Archivo | Validación |
| --- | --- | --- | --- | --- | --- |
| 0 | Entorno | Entorno en WSL2 con uv, Python 3.12 y versiones fijadas (`pyproject.toml`, `uv.lock`); `config.yaml` inicial; detección de GPUs, VRAM libre y capacidad de cómputo con PyTorch | Entornos reproducibles, elegir `dtype` según la GPU, cuánta VRAM queda realmente libre | `00_check_env.py`, `pyproject.toml`, `config.yaml` | Imprime GPUs, VRAM total y libre, `dtype` elegido, `gpu_memory_utilization` sugerido y versiones, o avisa si no hay GPU |
| 1 | Hub y datos | Login, descarga del modelo a la caché, carga de GSM8K y prompts con chat template | Caché del Hub, chat templates, respuesta gold tras `####` | `01_hub_and_data.py` | 10 prompts con su respuesta gold |
| 2 | Inferencia con transformers | (a) ejemplo básico con 3 prompts fijos y `pipeline`; (b) `generate()` sobre GSM8K | Chat template, tokenización y parámetros de generación | `02a_transformers_basic.py`, `02b_transformers_infer.py` | (a) 3 respuestas legibles; (b) 10 respuestas y tokens por segundo |
| 3 | Inferencia con vLLM | (a) ejemplo básico con los mismos 3 prompts de 02a; (b) `LLM` con `SamplingParams(n=5, seed=...)` y límites de memoria para 6 GB (`max_model_len` ≈ 2048, `gpu_memory_utilization` ≈ 0.70, `enforce_eager=True`) | Generación en lote y muestreo múltiple | `03a_vllm_basic.py`, `03b_vllm_infer.py` | (a) las mismas 3 respuestas de 02a, ahora con vLLM; (b) 50 respuestas y comparación de tokens por segundo con el módulo 2 |
| 4 | Guardar y analizar salidas | Extraer el número final, marcar aciertos, escribir JSONL o Parquet con una fila por muestra y un archivo con config, semilla, prompt, revisión del modelo y versiones | Datos crudos vs. derivados, metadatos, generar una vez y analizar muchas | `04_save_outputs.py` | Releer el archivo y calcular accuracy y maj@k (k = 1…5) sin regenerar |
| 5 | LM Eval Harness | `gsm8k` con `--model hf` y `--model vllm`, `--limit 20`, `--log_samples`; tarea propia en YAML basada en `gsm8k_cot_self_consistency` con 5 repeticiones (si el tiempo no alcanza, queda como opcional) | Evaluación estándar, backends, tareas propias | `05_lm_eval.sh`, `tasks/` | Dos archivos de resultados comparables y la métrica maj@k de la tarea propia |
| 6 (opcional) | Extras | Solo si sobra tiempo, uno a elección: registro en W&B o MLflow; `vllm serve` + cliente `openai`; carga en 4 bits con bitsandbytes | Seguimiento de corridas, inferencia como servicio, cuantización | `06_extras.py` | Corrida visible en el panel, respuesta vía API o menor uso de VRAM |

Prioridad: alcance chico de la demo y buena navegación por sobre un repo mínimo (por eso los módulos 2 y 3 se dividen en un ejemplo básico y uno completo).

En cada módulo te explico el objetivo, tú escribes un esqueleto con `TODO`, lo revisamos y recién entonces lo completamos y validamos.

## Estructura del repositorio y próximo paso

Un script por módulo, numerados en el orden de la demo, con salidas y configuración separadas del código.

```
llm-tools-demo/
├── config.yaml            # modelo, dtype, nº de preguntas, n, temperatura, semilla
├── pyproject.toml         # dependencias fijadas con uv (Python 3.12)
├── uv.lock                # versiones exactas resueltas; se versiona en git
├── common.py              # utilidades compartidas: leer config, elegir dtype, extraer respuesta
├── 00_check_env.py
├── 01_hub_and_data.py
├── 02a_transformers_basic.py   # ejemplo básico: 3 prompts fijos
├── 02b_transformers_infer.py   # GSM8K + tokens/s
├── 03a_vllm_basic.py           # ejemplo básico: mismos 3 prompts
├── 03b_vllm_infer.py           # GSM8K, n=5 + tokens/s
├── 04_save_outputs.py
├── 05_lm_eval.sh
├── tasks/                 # YAML de la tarea propia de LM Eval Harness
├── outputs/               # JSONL / Parquet + config usada
└── README.md
```

Próximo paso: Módulo 0, entorno en WSL2 y reconocimiento de la GPU. Cuando se conozca el servidor, se revisa la sección *Supuestos sobre el servidor* y se agrega lo que haga falta (por ejemplo, un script de Slurm).
