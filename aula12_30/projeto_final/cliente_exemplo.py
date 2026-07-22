"""
cliente_exemplo.py - Exemplos de uso da API (Projeto Final).

Suba a API antes:  uvicorn app.main:app --reload  (de dentro de projeto_final/)
Depois:            python cliente_exemplo.py caminho/do/documento.pdf
"""

import sys

import requests

BASE = "http://localhost:8001"
HEADERS = {}  # se usar API_KEYS no .env: {"X-API-Key": "sua-chave"}


def ingerir(caminho):
    with open(caminho, "rb") as f:
        r = requests.post(f"{BASE}/ingestao", files={"arquivo": f},
                          params={"estrategia": "auto"}, headers=HEADERS)
    print("INGESTAO:", r.status_code)
    print(r.json())


def consultar(pergunta, destino="auto"):
    r = requests.post(f"{BASE}/consulta",
                      json={"pergunta": pergunta, "destino": destino, "top_k": 5}, headers=HEADERS)
    print("CONSULTA:", r.status_code)
    print(r.json())


if __name__ == "__main__":
    print("health:", requests.get(f"{BASE}/health").json())
    if len(sys.argv) > 1:
        ingerir(sys.argv[1])
    consultar("Qual o tema principal dos documentos enviados?")
