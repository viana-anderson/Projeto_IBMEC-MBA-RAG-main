"""
metricas.py - As 3 familias de metricas de avaliacao de retrieval.

  retrieval     : Hit@K  (achou algum relevante no top-K?), Recall@K
  ranking       : MRR    (1/posicao do 1o relevante), NDCG@K (usa a nota graduada)
  separabilidade: AUC-ROC (os scores separam relevantes de nao-relevantes?)

Tudo em numpy + scikit-learn, sobre a matriz de similaridade (Q x D).
"""

import numpy as np
from sklearn.metrics import ndcg_score, roc_auc_score


def similaridade_cosseno(q_vecs, d_vecs):
    """q_vecs: Q x dim ; d_vecs: D x dim -> matriz Q x D de cosseno."""
    qn = q_vecs / (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-9)
    dn = d_vecs / (np.linalg.norm(d_vecs, axis=1, keepdims=True) + 1e-9)
    return qn @ dn.T


def avaliar(sims, corpus_ids, golds, k=10):
    """sims: Q x D ; corpus_ids: [id] (colunas) ; golds: [{doc_id: nota}] (linhas)."""
    hits = recalls = mrrs = ndcgs = 0.0
    soma_auc = 0.0
    n_q = n_auc = 0
    id_set = set(corpus_ids)

    for row, gold in zip(sims, golds):
        rel = {d: g for d, g in gold.items() if g > 0 and d in id_set}
        if not rel:
            continue
        n_q += 1
        relset = set(rel)
        ordem = np.argsort(-row)                     # colunas por similaridade desc
        topk_ids = [corpus_ids[j] for j in ordem[:k]]

        hits += 1.0 if (relset & set(topk_ids)) else 0.0
        recalls += len(relset & set(topk_ids)) / len(relset)

        rr = 0.0
        for rank, j in enumerate(ordem, 1):
            if corpus_ids[j] in relset:
                rr = 1.0 / rank
                break
        mrrs += rr

        y_true = np.array([[rel.get(cid, 0) for cid in corpus_ids]])   # nota graduada
        ndcgs += ndcg_score(y_true, row.reshape(1, -1), k=k)

        y_bin = np.array([1 if cid in relset else 0 for cid in corpus_ids])
        if 0 < y_bin.sum() < len(y_bin):
            soma_auc += roc_auc_score(y_bin, row)
            n_auc += 1

    n = max(1, n_q)
    return {
        f"hit@{k}": round(hits / n, 4),
        f"recall@{k}": round(recalls / n, 4),
        "mrr": round(mrrs / n, 4),
        f"ndcg@{k}": round(ndcgs / n, 4),
        "auc": round(soma_auc / max(1, n_auc), 4),
        "n_queries": n_q,
    }

