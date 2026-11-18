"""Utilidades compartidas por los módulos de la demo.

Se espera:
- load_config(): leer config.yaml.
- pick_dtype(): elegir el dtype según la GPU (CC >= 8.0 -> bfloat16, menor -> float16, sin GPU -> float32).
- extract_answer(): extraer el número final de una respuesta (y la respuesta gold tras '####').
"""
