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
def consultar_opensearch(pergunta, top_k, tecnica="baseline"):
    from . import busca_avancada

    log.info("Consulta OpenSearch (tecnica=%s, top_k=%d): %r", tecnica, top_k, pergunta)
    pipe, inputs, chave_docs = busca_avancada.construir(tecnica, top_k, pergunta)
    saida = pipe.run(inputs, include_outputs_from={chave_docs})
    docs = saida[chave_docs]["documents"]
    replies = saida["llm"]["replies"]
    resposta = (replies[0] if replies else "").strip()
    log.info("Recuperados %d trecho(s); resposta gerada (Groq)", len(docs))
    if "tracer" in saida and saida["tracer"].get("trace_url"):
        log.info("Langfuse trace (busca): %s", saida["tracer"]["trace_url"])
    fontes = [{"id": d.meta.get("id_original") or d.meta.get("arquivo"),
               "trecho": d.content[:160]} for d in docs]
    return resposta, fontes


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
def consultar(pergunta, destino="auto", top_k=5, tecnica="baseline"):
    if destino == "grafo":
        resp, fontes = consultar_grafo(pergunta)        # tecnica nao se aplica ao grafo
        return resp, fontes, "grafo"
    resp, fontes = consultar_opensearch(pergunta, top_k, tecnica)
    return resp, fontes, "opensearch"
