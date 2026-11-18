"""Módulo 3a: ejemplo básico de generación con vLLM.

Se espera: ~15 líneas legibles en pantalla. Los mismos 3 prompts de 02a con LLM + SamplingParams
(seed fija) y llm.chat() para aplicar el chat template; imprimir prompt y respuesta.
Límites de memoria para 6 GB (gpu_memory_utilization ~0.70, max_model_len ~2048,
enforce_eager=True), sin max_num_seqs=1 para no desactivar el batching. Sin GSM8K ni guardado.
"""
