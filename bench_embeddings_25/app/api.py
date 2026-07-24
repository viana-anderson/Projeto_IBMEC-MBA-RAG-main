"""
api.py - API FastAPI da bancada de avaliacao de embeddings.

Endpoints:
  POST /avaliar  - {modelos, k, dataset_json?} -> metricas por modelo + melhor + explicacao
  GET  /modelos  - modelos sugeridos + os instalados no Ollama
  GET  /health   - status

Rodar:  uvicorn app.api:app --reload --port 8000   (de dentro de bench_embeddings/)
"""

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from . import avaliacao, dados, embeddings

app = FastAPI(title="Bancada de Avaliacao de Embeddings",
              description="Compara modelos de embedding (Ollama) por retrieval/ranking/separabilidade.")

MODELOS_SUGERIDOS = ["nomic-embed-text", "bge-m3", "mxbai-embed-large", "snowflake-arctic-embed"]


class AvaliarRequest(BaseModel):
    modelos: List[str] = ["nomic-embed-text", "bge-m3"]
    k: int = 10
    dataset_json: Optional[str] = None   # JSON (string) enviado pelo aluno; None = dataset padrao


@app.post("/avaliar")
def avaliar(req: AvaliarRequest):
    if req.dataset_json:
        cids, ctexts, queries = dados.carregar_json(req.dataset_json)
    else:
        cids, ctexts, queries = dados.carregar_padrao()
    resultado = avaliacao.avaliar(cids, ctexts, queries, req.modelos, req.k)
    resultado["corpus"] = len(cids)
    resultado["queries"] = len(queries)
    return resultado


@app.get("/modelos")
def modelos():
    return {"sugeridos": MODELOS_SUGERIDOS, "instalados_ollama": embeddings.modelos_instalados()}


@app.get("/health")
def health():
    return {"api": "ok", "ollama": embeddings.base_url(),
            "modelos_instalados": len(embeddings.modelos_instalados())}
