"""Módulo 4: guardar y analizar salidas.

Se espera: una fila por muestra en outputs/generations.parquet (o JSONL) y outputs/run_meta.json
con config, semilla, prompt, revisión del modelo y versiones; releer el archivo y calcular
accuracy y maj@k (k = 1..5) sin regenerar.
"""
