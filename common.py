"""Utilidades compartidas por los módulos de la demo."""
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Lee config.yaml y lo devuelve como diccionario."""
    # TODO: abrir el archivo y cargarlo con yaml.safe_load
    ...


def pick_dtype(requested: str = "auto", device: int = 0) -> str:
    """Devuelve el dtype a usar como string ("bfloat16", "float16" o "float32").

    - Si requested no es "auto", se respeta (y se avisa si la GPU no lo soporta).
    - Sin GPU -> "float32".
    - Compute capability >= 8.0 -> "bfloat16"; menor -> "float16".
    """
    # TODO: import torch dentro de la función (common.py no debe exigir GPU para importarse)
    # TODO: torch.cuda.is_available() / torch.cuda.get_device_capability(device)
    ...


def extract_answer(text: str) -> str | None:
    """Módulos 4 y 5: extraer el número final de una respuesta."""
    raise NotImplementedError  # se completa en el Módulo 4
