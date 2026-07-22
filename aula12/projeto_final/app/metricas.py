"""
metricas.py - Metricas de RECUPERACAO/ranking para avaliar a busca (uma pergunta por vez).

Dado o ranking de documentos recuperados (ids + scores) e o GABARITO de relevancia
({doc_id: nota}), calcula:
  - Hit@K      : houve pelo menos 1 relevante no top-K? (0/1)
  - Recall@K   : fracao dos relevantes que apareceu no top-K
  - MRR        : 1 / posicao do 1o relevante (na lista inteira)
  - NDCG@K     : qualidade do ranking ponderando posicao e NOTA (ganho graduado)
  - AUC-ROC    : separabilidade dos scores entre relevante e nao-relevante (entre os recuperados)

Sem gabarito nao ha como medir - por isso o dataset RAGAS carrega os documentos_relevantes.
"""

import math

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # sklearn opcional; AUC vira None se faltar
    roc_auc_score = None


def _dcg(ganhos):
    return sum(g / math.log2(i + 2) for i, g in enumerate(ganhos))


def metricas_por_query(ids_ranqueados, scores, relevancia, k=5):
    """ids_ranqueados/scores em ORDEM de ranking; relevancia = {doc_id: nota (>0)}."""
    rel = {i: float(n) for i, n in (relevancia or {}).items() if float(n) > 0}
    n_rel = len(rel)
    topk_ids = ids_ranqueados[:k]

    hit = 1.0 if any(i in rel for i in topk_ids) else 0.0
    recall = (sum(1 for i in topk_ids if i in rel) / n_rel) if n_rel else 0.0

    mrr = 0.0
    for pos, i in enumerate(ids_ranqueados, start=1):
        if i in rel:
            mrr = 1.0 / pos
            break

    ganhos = [rel.get(i, 0.0) for i in topk_ids]
    ideais = sorted(rel.values(), reverse=True)[:k]
    idcg = _dcg(ideais)
    ndcg = (_dcg(ganhos) / idcg) if idcg > 0 else 0.0

    auc = None
    labels = [1 if i in rel else 0 for i in ids_ranqueados]
    if roc_auc_score is not None and 0 < sum(labels) < len(labels):
        try:
            auc = float(roc_auc_score(labels, list(scores)))
        except Exception:
            auc = None

    return {f"hit@{k}": round(hit, 4), f"recall@{k}": round(recall, 4),
            "mrr": round(mrr, 4), f"ndcg@{k}": round(ndcg, 4),
            "auc": (round(auc, 4) if auc is not None else None)}


def agregar(linhas, k=5):
    """Media das metricas sobre varias perguntas (ignora AUC None)."""
    if not linhas:
        return {}
    def media(chave, filtra_none=False):
        vals = [l[chave] for l in linhas if l.get(chave) is not None] if filtra_none \
               else [l.get(chave, 0) or 0 for l in linhas]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {f"hit@{k}": media(f"hit@{k}"), f"recall@{k}": media(f"recall@{k}"),
            "mrr": media("mrr"), f"ndcg@{k}": media(f"ndcg@{k}"),
            "auc": media("auc", filtra_none=True), "n_perguntas": len(linhas)}
