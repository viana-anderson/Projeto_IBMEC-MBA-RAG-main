"""
consulta.py - RAG de consulta (busca), roteado por storage, com OBSERVABILIDADE Langfuse
e TECNICAS de query enhancement (baseline | multi_query | rag_fusion | step_back).

destino:
  - 'opensearch' (ou 'auto'): pipeline Haystack montado em busca_avancada.construir(tecnica)
    (embed/reescrita -> recuperacao -> prompt -> Groq), instrumentado por LangfuseConnector.
  - 'grafo'                  : LightRAG (modo hibrido), rastreado com @observe do Langfuse.

A observabilidade so liga se LANGFUSE_PUBLIC_KEY/SECRET_KEY existirem (.env). O preparo do
tracing e feito em app/__init__.py, antes de qualquer import do Haystack.
"""

from . import config, indexacao
from .log import obter_logger

log = obter_logger(__name__)


# ---------------------------------------------------------------------------
# Busca no OpenSearch (com tecnica de query enhancement) - ver busca_avancada.py
# ---------------------------------------------------------------------------
def buscar_detalhado(pergunta, top_k, tecnica="baseline", rerank="rrf"):
    """Roda a busca OpenSearch e devolve (resposta, docs) - docs sao Documents do Haystack
    (com .id, .content, .score). Usado pela consulta comum E pela avaliacao em lote."""
    from . import busca_avancada

    log.info("Consulta OpenSearch (tecnica=%s, rerank=%s, top_k=%d): %r",
             tecnica, rerank, top_k, pergunta)
    pipe, inputs, chave_docs = busca_avancada.construir(tecnica, top_k, pergunta, rerank=rerank)
    saida = pipe.run(inputs, include_outputs_from={chave_docs})
    docs = saida[chave_docs]["documents"]
    replies = saida["llm"]["replies"]
    resposta = (replies[0] if replies else "").strip()
    log.info("Recuperados %d trecho(s); resposta gerada", len(docs))
    if "tracer" in saida and saida["tracer"].get("trace_url"):
        log.info("Langfuse trace (busca): %s", saida["tracer"]["trace_url"])
    return resposta, docs


def consultar_opensearch(pergunta, top_k, tecnica="baseline", rerank="rrf"):
    resposta, docs = buscar_detalhado(pergunta, top_k, tecnica, rerank)
    fontes = [{"id": d.meta.get("id_original") or d.meta.get("arquivo"),
               "trecho": d.content[:160]} for d in docs]
    return resposta, fontes


def _norm(t):
    return " ".join((t or "").split()).casefold()


def _match_semantico(pergunta, itens, limiar=0.55):
    """Acha o gabarito cuja PERGUNTA e mais parecida (cosseno de embeddings Ollama).

    Isso permite medir uma PARAFRASE contra o gabarito da mesma necessidade de informacao
    (o gabarito = documentos relevantes, nao a frase exata). Retorna (item, similaridade).
    """
    import numpy as np
    import requests

    base, modelo = config.config_ollama()
    textos = [pergunta] + [it.get("pergunta", "") for it in itens]
    r = requests.post(f"{base}/api/embed", json={"model": modelo, "input": textos}, timeout=120)
    r.raise_for_status()
    vetores = np.array(r.json()["embeddings"], dtype=float)
    q, docs = vetores[0], vetores[1:]
    q = q / (np.linalg.norm(q) or 1.0)
    docs = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9)
    sims = docs @ q
    i = int(np.argmax(sims))
    return (itens[i], float(sims[i])) if float(sims[i]) >= limiar else (None, float(sims[i]))


def _achar_gabarito(pergunta, dataset_nome, gabarito_pergunta):
    """Escolhe o item de gabarito: explicito (dropdown) ou semantico ('auto')."""
    from . import avaliacao_ragas

    if not dataset_nome or dataset_nome in ("nenhum", "", None):
        return None, ""
    itens = (avaliacao_ragas.carregar_dataset(dataset_nome) or {}).get("itens", [])
    if not itens:
        return None, ""
    escolhido = _norm(gabarito_pergunta)
    if escolhido and escolhido not in ("auto", "auto (semantico)", "auto (semântico)"):
        # explicito: casa pela pergunta escolhida no dropdown
        for it in itens:
            if _norm(it.get("pergunta")) == escolhido:
                return it, f"gabarito escolhido: “{it.get('pergunta')}”"
        return None, "gabarito escolhido nao encontrado no dataset"
    # auto: casamento semantico
    try:
        it, sim = _match_semantico(pergunta, itens)
        if it:
            return it, f"gabarito (auto, similaridade {sim:.2f}): “{it.get('pergunta')}”"
        return None, f"nenhum gabarito próximo (melhor similaridade {sim:.2f})"
    except Exception as e:
        return None, f"casamento semantico falhou: {e}"


def _metricas_de_uma(pergunta, docs, resposta, dataset_nome, top_k, gabarito_pergunta="auto"):
    """Metricas de UMA pergunta: retrieval (se houver gabarito) + RAGAS.

    O gabarito e a NECESSIDADE DE INFORMACAO (documentos relevantes), nao a frase exata:
    por isso uma parafrase pode ser medida contra o gabarito (explicito ou por similaridade).
    """
    from . import avaliacao_ragas, metricas

    resultado = {"retrieval": None, "ragas": None, "gabarito": None}
    gold, nota = _achar_gabarito(pergunta, dataset_nome, gabarito_pergunta)
    resultado["gabarito"] = nota or None
    if gold:  # so ha metricas de recuperacao com gabarito
        rel = {i: 1 for i in gold.get("documentos_relevantes", [])}
        ids = [d.id for d in docs]
        scores = [float(d.score or 0.0) for d in docs]
        resultado["retrieval"] = metricas.metricas_por_query(ids, scores, rel, k=top_k)
    # RAGAS: faithfulness/answer-relevancy sempre; context precision/recall so se houver gabarito
    item = {"pergunta": pergunta, "resposta_referencia": (gold or {}).get("resposta_referencia", ""),
            "contextos": [d.content for d in docs], "resposta": resposta}
    r = avaliacao_ragas.avaliar_ragas([item])
    resultado["ragas"] = r.get("medias") if r.get("ok") else {"erro": r.get("erro")}
    return resultado


# ---------------------------------------------------------------------------
# Busca no grafo (LightRAG) - rastreada com @observe
# ---------------------------------------------------------------------------
def _grafo_raw(pergunta):
    from lightrag import QueryParam

    async def _q():
        rag = await indexacao._criar_lightrag()
        try:
            return await rag.aquery(pergunta, param=QueryParam(mode="hybrid"))
        finally:
            await rag.finalize_storages()

    resposta = indexacao.rodar_async(_q)  # seguro com/sem event loop ativo
    return resposta, [{"id": "grafo", "trecho": "(resposta sintetizada do grafo de conhecimento)"}]


def consultar_grafo(pergunta):
    log.info("Consulta ao GRAFO (LightRAG, modo hybrid): %r", pergunta)
    if config.langfuse_configurado():
        try:
            from langfuse import observe
            return observe(name="busca-grafo-aula12")(_grafo_raw)(pergunta)
        except Exception as e:
            log.warning("Langfuse (grafo) indisponivel (%s) -> seguindo sem trace", e)
    return _grafo_raw(pergunta)


# ---------------------------------------------------------------------------
# Roteador
# ---------------------------------------------------------------------------
def consultar(pergunta, destino="auto", top_k=5, tecnica="baseline", rerank="rrf",
              dataset_nome=None, com_metricas=False, gabarito_pergunta="auto"):
    """Retorna (resposta, fontes, destino_usado, metricas).

    metricas: None no grafo (metricas nao se aplicam) ou quando com_metricas=False;
    caso contrario {retrieval:..., ragas:..., gabarito:...} para a pergunta.
    """
    if destino == "grafo":
        resp, fontes = consultar_grafo(pergunta)        # tecnica/rerank/metricas nao se aplicam
        return resp, fontes, "grafo", None
    resp, docs = buscar_detalhado(pergunta, top_k, tecnica, rerank)
    fontes = [{"id": d.meta.get("id_original") or d.meta.get("arquivo"),
               "trecho": d.content[:160]} for d in docs]
    metricas = (_metricas_de_uma(pergunta, docs, resp, dataset_nome, top_k, gabarito_pergunta)
                if com_metricas else None)
    return resp, fontes, "opensearch", metricas
