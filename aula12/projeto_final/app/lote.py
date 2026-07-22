"""
lote.py - Avaliacao em LOTE da busca OpenSearch.

Recebe uma lista de perguntas (CSV) e/ou um dataset RAGAS (gabarito). Para cada pergunta:
  - roda a busca (tecnica de query enhancement + rerank escolhidos);
  - se houver gabarito, calcula Hit@K, Recall@K, MRR, NDCG@K, AUC-ROC (metricas.py);
  - coleta contexto+resposta para o RAGAS.
Ao final, agrega as medias e (se houver gabarito + ragas) roda a avaliacao RAGAS.

As metricas de RECUPERACAO exigem gabarito -> por isso vem do dataset RAGAS. Sem dataset,
o lote apenas executa as perguntas e devolve as respostas (sem metricas).
"""

from . import avaliacao_ragas, consulta, metricas
from .log import obter_logger

log = obter_logger(__name__)


def avaliar_lote(perguntas, dataset_nome=None, tecnica="baseline", rerank="rrf", top_k=5):
    ds = None
    if dataset_nome and dataset_nome not in ("nenhum", "", None):
        ds = avaliacao_ragas.carregar_dataset(dataset_nome)
    gold = {}
    if ds:
        for it in ds.get("itens", []):
            gold[(it.get("pergunta") or "").strip()] = it

    # de onde vem as perguntas: CSV enviado, senao as do dataset
    lista = [p for p in (perguntas or []) if str(p).strip()]
    if not lista and ds:
        lista = [it["pergunta"] for it in ds["itens"]]
    if not lista:
        return {"ok": False, "erro": "Envie um CSV de perguntas ou selecione um dataset RAGAS."}

    log.info("Avaliacao em lote: %d perguntas (tecnica=%s, rerank=%s, top_k=%d, dataset=%s)",
             len(lista), tecnica, rerank, top_k, dataset_nome)

    linhas, itens_ragas = [], []
    chaves_metricas = [f"hit@{top_k}", f"recall@{top_k}", "mrr", f"ndcg@{top_k}", "auc"]
    for q in lista:
        try:
            resposta, docs = consulta.buscar_detalhado(q, top_k, tecnica, rerank)
        except Exception as e:
            log.warning("Falha na pergunta %r: %s", q, e)
            linha = {"pergunta": q, "tecnica": tecnica, "rerank": rerank, "erro": str(e)}
            linhas.append(linha)
            continue
        ids = [d.id for d in docs]
        scores = [float(d.score or 0.0) for d in docs]
        linha = {"pergunta": q, "tecnica": tecnica, "rerank": rerank, "resposta": resposta[:200]}
        g = gold.get(q.strip())
        if g:
            relevancia = {i: 1 for i in g.get("documentos_relevantes", [])}
            linha.update(metricas.metricas_por_query(ids, scores, relevancia, k=top_k))
            itens_ragas.append({"pergunta": q,
                                "resposta_referencia": g.get("resposta_referencia", ""),
                                "contextos": [d.content for d in docs],
                                "resposta": resposta, "_idx": len(linhas)})
        else:
            for c in chaves_metricas:
                linha[c] = None
        linhas.append(linha)

    com_metrica = [l for l in linhas if l.get("mrr") is not None]
    medias = metricas.agregar(com_metrica, k=top_k) if com_metrica else {}

    ragas = None
    if itens_ragas:
        r = avaliacao_ragas.avaliar_ragas(itens_ragas)
        if r.get("ok"):
            ragas = {"medias": r["medias"]}
            for it, pi in zip(itens_ragas, r["por_item"]):
                linhas[it["_idx"]].update({f"ragas_{k}": v for k, v in pi.items()})
        else:
            ragas = {"erro": r.get("erro")}

    return {"ok": True, "tecnica": tecnica, "rerank": rerank, "top_k": top_k,
            "n_perguntas": len(lista), "com_gabarito": len(com_metrica),
            "linhas": linhas, "medias_retrieval": medias, "ragas": ragas}
