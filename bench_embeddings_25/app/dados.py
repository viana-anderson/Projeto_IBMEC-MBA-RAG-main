"""
dados.py - Carrega o corpus + queries com gabarito para o benchmark.

Formato normalizado:
  corpus_ids   : [id, ...]
  corpus_texts : [texto, ...]   (alinhado a corpus_ids)
  queries      : [{"id","query","gold": {doc_id: nota}}, ...]   (nota>0 = relevante)

Dataset padrao: aula11/datasets/corpus_juridico_benchmark.json (documentos +
queries_benchmark com relevancia graduada). O aluno tambem pode enviar um JSON proprio
no MESMO formato (documentos + queries_benchmark) OU {corpus, queries}.
"""

import json
from pathlib import Path

PASTA_APP = Path(__file__).resolve().parent
PASTA_BENCH = PASTA_APP.parent
PASTA_PROJETO = PASTA_BENCH.parent
DATASET_PADRAO = PASTA_PROJETO / "aula11" / "datasets" / "corpus_juridico_benchmark.json"


def _normalizar(d):
    # aceita {documentos, queries_benchmark} (formato do curso) ou {corpus, queries}
    docs = d.get("documentos") or d.get("corpus")
    qs = d.get("queries_benchmark") or d.get("queries")
    if not docs or not qs:
        raise ValueError("JSON precisa ter 'documentos'+'queries_benchmark' (ou 'corpus'+'queries').")

    corpus_ids, corpus_texts = [], []
    for doc in docs:
        corpus_ids.append(str(doc.get("id")))
        corpus_texts.append(doc.get("texto") or doc.get("text") or doc.get("conteudo") or "")

    queries = []
    for q in qs:
        gold = q.get("relevancia") or q.get("gold") or q.get("documentos_relevantes") or {}
        # aceita lista de ids (nota 1) ou dict {id: nota}
        if isinstance(gold, list):
            gold = {str(x): 1 for x in gold}
        else:
            gold = {str(k): int(v) for k, v in gold.items()}
        queries.append({"id": str(q.get("id")), "query": q.get("query") or q.get("texto") or q.get("pergunta"),
                        "gold": gold})
    return corpus_ids, corpus_texts, queries


def carregar_padrao():
    with open(DATASET_PADRAO, "r", encoding="utf-8") as f:
        return _normalizar(json.load(f))


def carregar_json(texto_ou_caminho):
    """Carrega de uma string JSON ou de um caminho de arquivo."""
    p = Path(str(texto_ou_caminho))
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return _normalizar(json.load(f))
    return _normalizar(json.loads(texto_ou_caminho))
