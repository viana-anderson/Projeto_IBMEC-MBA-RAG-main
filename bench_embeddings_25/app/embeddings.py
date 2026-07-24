"""
embeddings.py - Embeddings via API do Ollama (/api/embed), um modelo por vez.

Cada modelo deve estar baixado no Ollama (ex.: `ollama pull nomic-embed-text`).
Retorna a matriz de vetores (numpy) e o tempo gasto (para a metrica de latencia).
"""

import os
import time

import numpy as np
import requests


def base_url():
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def embedar(textos, modelo, lote=64, timeout=600):
    """Embeda uma lista de textos com 'modelo' do Ollama. -> (np.array Nxdim, segundos)."""
    t0 = time.perf_counter()
    vetores = []
    url = f"{base_url()}/api/embed"
    for i in range(0, len(textos), lote):
        bloco = textos[i:i + lote]
        r = requests.post(url, json={"model": modelo, "input": bloco}, timeout=timeout)
        r.raise_for_status()
        vetores.extend(r.json()["embeddings"])
    return np.array(vetores, dtype="float32"), time.perf_counter() - t0


def modelos_instalados():
    """Lista os modelos disponiveis no Ollama (para sugerir na interface)."""
    try:
        r = requests.get(f"{base_url()}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []
