"""
avaliacao.py - Orquestra o benchmark: embeda (Ollama) -> cosseno -> metricas -> melhor.

Para cada modelo: embeda corpus e queries, calcula a matriz de similaridade e as 3
familias de metricas; registra dimensao e latencia. Ao final, elege o melhor (por NDCG@K)
e gera uma explicacao textual automatica para o aluno.
"""

from . import embeddings, metricas


def avaliar(corpus_ids, corpus_texts, queries, modelos, k=10):
    q_textos = [q["query"] for q in queries]
    golds = [q["gold"] for q in queries]
    resultados = []
    for modelo in modelos:
        try:
            d_vec, t_d = embeddings.embedar(corpus_texts, modelo)
            q_vec, t_q = embeddings.embedar(q_textos, modelo)
        except Exception as e:
            resultados.append({"modelo": modelo, "erro": str(e)})
            continue
        sims = metricas.similaridade_cosseno(q_vec, d_vec)
        m = metricas.avaliar(sims, corpus_ids, golds, k)
        m.update({"modelo": modelo, "dim": int(d_vec.shape[1]), "latencia_s": round(t_d + t_q, 2)})
        resultados.append(m)

    validos = [r for r in resultados if "erro" not in r]
    melhor = max(validos, key=lambda r: r[f"ndcg@{k}"]) if validos else None
    nome_melhor = melhor["modelo"] if melhor else None
    return {"k": k, "resultados": resultados,
            "melhor": nome_melhor,
            "explicacao": _explicar(validos, nome_melhor, k)}


def _lider(validos, metrica):
    return max(validos, key=lambda r: r.get(metrica, 0))


def _explicar(validos, melhor, k):
    if not validos:
        return "Nenhum modelo avaliado com sucesso (verifique o Ollama e os nomes dos modelos)."
    if len(validos) == 1:
        r = validos[0]
        return (f"Apenas **{r['modelo']}** foi avaliado: NDCG@{k}={r[f'ndcg@{k}']}, "
                f"Recall@{k}={r[f'recall@{k}']}, AUC={r['auc']}, dim={r['dim']}, "
                f"latencia={r['latencia_s']}s.")
    nome_ndcg = _lider(validos, f"ndcg@{k}")["modelo"]
    nome_recall = _lider(validos, f"recall@{k}")["modelo"]
    nome_auc = _lider(validos, "auc")["modelo"]
    mais_rapido = min(validos, key=lambda r: r["latencia_s"])
    m = next(r for r in validos if r["modelo"] == melhor)
    txt = [
        f"**Recomendado: {melhor}** — lidera o ranking (NDCG@{k}={m[f'ndcg@{k}']}), "
        f"com Recall@{k}={m[f'recall@{k}']} e AUC={m['auc']} (dim={m['dim']}, latencia={m['latencia_s']}s).",
        "",
        "Por familia de metrica:",
        f"- Ranking (NDCG@{k}): melhor = **{nome_ndcg}** (o que ordena os relevantes mais no topo).",
        f"- Recuperacao (Recall@{k}): melhor = **{nome_recall}** (traz mais relevantes no top-{k}).",
        f"- Separabilidade (AUC): melhor = **{nome_auc}** (melhor separa relevante de nao-relevante).",
        f"- Mais rapido: **{mais_rapido['modelo']}** ({mais_rapido['latencia_s']}s).",
        "",
        "Como ler: se um modelo lidera NDCG e AUC, e o mais robusto para o dominio. "
        "Se outro empata na qualidade mas tem dimensao/latencia menor, pode compensar em producao.",
    ]
    return "\n".join(txt)
