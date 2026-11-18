"""Módulo 0: reconocer la máquina antes de correr nada.

Imprime versiones, GPUs (compute capability, VRAM total y libre), el dtype elegido y si el
gpu_memory_utilization de config.yaml cabe en la VRAM libre. Avisa si no hay GPU.

Uso: uv run 00_check_env.py
"""
import os
import platform
from importlib.metadata import PackageNotFoundError, version

import torch

from common import load_config, pick_dtype

PACKAGES = ["torch", "transformers", "vllm", "lm-eval", "datasets", "huggingface-hub", "pandas"]
SAFETY_MARGIN = 0.05  # fracción de VRAM que se deja libre por si otro proceso la necesita


def print_versions() -> None:
    """Python, cada paquete de PACKAGES y la versión de CUDA con que se compiló torch."""
    # TODO: platform.python_version()
    # TODO: version(pkg) para cada paquete, capturando PackageNotFoundError
    # TODO: torch.version.cuda
    ...


def print_gpus() -> list[dict]:
    """Una línea por GPU visible; devuelve sus datos para usarlos después.

    Por GPU: índice, nombre, compute capability, VRAM total y libre (GiB).
    """
    # TODO: mostrar CUDA_VISIBLE_DEVICES (os.environ.get, "no definido" si falta)
    # TODO: torch.cuda.device_count(); por cada i:
    #       get_device_name(i), get_device_capability(i), mem_get_info(i) -> (libre, total)
    ...


def suggest_gpu_memory_utilization(free: int, total: int) -> float:
    """Fracción de VRAM que vLLM puede pedir sin fallar al arrancar.

    vLLM reserva gpu_memory_utilization * total y exige que esa memoria esté libre.
    Regla: libre/total - SAFETY_MARGIN, redondeado hacia abajo a múltiplos de 0.05.
    """
    # TODO
    ...


def main() -> None:
    cfg = load_config()

    print_versions()

    if not torch.cuda.is_available():
        # TODO: aviso claro: sin GPU vLLM no corre; transformers sí, pero lento (float32)
        return

    gpus = print_gpus()

    # TODO: dtype elegido con pick_dtype(cfg["dtype"]) y por qué (CC de la GPU 0)
    # TODO: sugerido = suggest_gpu_memory_utilization(...) para la GPU 0
    # TODO: comparar con cfg["vllm"]["gpu_memory_utilization"]:
    #       OK si config <= sugerido; si no, avisar y proponer el valor sugerido
    ...


if __name__ == "__main__":
    main()
