"""Módulo 3b: inferencia con vLLM sobre GSM8K.

Se espera: LLM + SamplingParams(n=5, temperature, seed) con límites de memoria para 6 GB
(max_model_len ~2048, gpu_memory_utilization ~0.70, enforce_eager=True);
50 respuestas y tokens por segundo para comparar con el módulo 2.
"""
