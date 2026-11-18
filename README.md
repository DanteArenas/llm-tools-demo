# llm-tools-demo

Demo introductoria (~20 min) de herramientas para diseñar, ejecutar, guardar y evaluar
experimentos con LLMs: PyTorch, transformers, vLLM, pandas y LM Eval Harness.
Caso de estudio: *Optimal Self-Consistency for Efficient Reasoning with LLMs* (arXiv 2511.12309,
PDF en este repo). Del paper se toma solo su patrón de trabajo: **generar una vez, guardar todo
y analizar muchas veces**.

El plan completo (bloques de la demo, mapa de herramientas, módulos) está en
[demo_herramientas_llms.md](demo_herramientas_llms.md).

## Requisitos

- Linux con GPU NVIDIA. En Windows, usar **WSL2 (Ubuntu)**: vLLM no corre en Windows nativo.
- **Driver NVIDIA instalado en Windows**, no dentro de WSL. WSL usa el driver de Windows;
  `nvidia-smi` dentro de WSL debe mostrar la GPU.
- ~10 GB de disco para el entorno (`.venv/`) y unos GB más para la caché de modelos de Hugging Face.
- Python no hace falta instalarlo: uv descarga Python 3.12.

Probado en: RTX 2060 6 GB (Turing, CC 7.5), WSL2 con Ubuntu 26.04, driver 591.74.

### Memoria en WSL (opcional)

WSL usa por defecto la mitad de la RAM de Windows. Si vLLM se queda sin RAM al cargar el modelo,
crear `C:\Users\<usuario>\.wslconfig` con, por ejemplo:

```ini
[wsl2]
memory=12GB
swap=8GB
```

y reiniciar WSL desde PowerShell con `wsl --shutdown`.

## Instalación

```bash
# 1. Instalar uv (gestor de entornos y paquetes)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Clonar y crear el entorno con las versiones exactas de uv.lock
git clone https://github.com/DanteArenas/llm-tools-demo.git
cd llm-tools-demo
uv sync
```

`uv sync` descarga Python 3.12 (fijado en `.python-version`) y crea `.venv/` con lo que dice
`uv.lock`. La primera vez tarda varios minutos (~9 GB, sobre todo torch y CUDA).

Comprobar que torch ve la GPU:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Los scripts se ejecutan con `uv run`, sin activar el entorno a mano:

```bash
uv run 00_check_env.py
```

## Entorno y versiones

Las dependencias directas están fijadas en [pyproject.toml](pyproject.toml); `uv.lock` fija
todas las demás. **vLLM manda**: cada versión de vLLM exige una versión exacta de torch y un rango
de transformers, así que para actualizar se cambia vLLM primero y luego `uv lock && uv sync`.

| Paquete | Versión | Nota |
| --- | --- | --- |
| Python | 3.12 | |
| vllm | 0.33.0 | |
| torch | 2.14.0 (CUDA 13.0) | la fija vllm |
| transformers | 5.20.2 | vllm 0.33.0 exige `>=5.16.1,<5.21` |
| lm-eval | 0.4.13 | extras `[hf,vllm]` (incluye accelerate) |
| datasets / huggingface-hub | 5.2.0 / 2.4.0 | |
| pandas / pyarrow / pyyaml | 3.1.1 / 26.0.1 / 6.0.3 | |

### Notas para GPUs sin bfloat16 (Turing y anteriores, CC < 8.0)

- Usar **float16**; el código elige el `dtype` según la GPU (`pick_dtype()` en `common.py`).
- FlashAttention y FlashInfer requieren CC ≥ 8.0: vLLM cae a backends de atención más lentos
  (Triton / FlexAttention). Para la escala de la demo no importa.
- Con 6 GB y el escritorio de Windows ocupando ~1 GB, usar en vLLM `gpu_memory_utilization` ≈ 0.70,
  `max_model_len` ≈ 2048 y `enforce_eager: true` (ver `config.yaml`).

## Estructura

```
llm-tools-demo/
├── config.yaml              # configuración única: modelo, dtype, nº preguntas, n, temperatura, semilla
├── pyproject.toml           # dependencias directas fijadas
├── uv.lock                  # versiones exactas resueltas (se versiona)
├── .python-version          # Python 3.12 (lo crea `uv python pin`)
├── common.py                # utilidades compartidas: leer config, elegir dtype, extraer respuesta
├── 00_check_env.py          # Módulo 0
├── 01_hub_and_data.py       # Módulo 1
├── 02_transformers_infer.py # Módulo 2
├── 03_vllm_infer.py         # Módulo 3
├── 04_save_outputs.py       # Módulo 4
├── 05_lm_eval.sh            # Módulo 5
├── tasks/                   # YAML de la tarea propia de LM Eval Harness
├── outputs/                 # generaciones y metadatos (ignorado por git)
└── demo_herramientas_llms.md
```

## Módulos

| Módulo | Archivo | Qué hace | Estado |
| --- | --- | --- | --- |
| 0. Entorno | `00_check_env.py` | GPUs, VRAM libre, `dtype` y versiones | entorno listo; script en curso |
| 1. Hub y datos | `01_hub_and_data.py` | Modelo en caché, GSM8K y prompts con chat template | pendiente |
| 2. Inferencia con transformers | `02_transformers_infer.py` | 1 respuesta por pregunta y tokens/s | pendiente |
| 3. Inferencia con vLLM | `03_vllm_infer.py` | 5 respuestas por pregunta y tokens/s | pendiente |
| 4. Guardar y analizar | `04_save_outputs.py` | Parquet + metadatos; accuracy y maj@k | pendiente |
| 5. LM Eval Harness | `05_lm_eval.sh`, `tasks/` | gsm8k con backends hf y vllm; maj@k | pendiente |
