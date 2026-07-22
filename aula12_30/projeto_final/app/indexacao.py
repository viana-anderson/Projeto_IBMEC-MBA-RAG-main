"""
indexacao.py - Decide ESTRATEGIA de indexacao (destino + chunking) e grava no storage.

Decisoes TRANSPARENTES (heuristicas, sem LLM):
  DESTINO  : 'grafo' (LightRAG) se texto longo e rico em entidades (multi-hop);
             senao 'opensearch'. Override: 'auto' | 'opensearch' | 'grafo'.

  CHUNKING (so quando destino=opensearch) - escolhe a melhor das tecnicas:
     - tabela          : documento tem tabelas -> 1 chunk por tabela (preserva estrutura)
     - hierarquico     : documento estruturado em secoes (varios titulos)
     - sentenca_janela : texto de lei / denso em artigos -> precisao por sentenca
     - semantico       : texto longo e heterogeneo -> corta em mudancas de topico
     - recursivo       : texto corrido (default robusto: respeita paragrafos/sentencas)
     - fixo            : documento curto / baseline
   Override: 'auto' | fixo | recursivo | sentenca_janela | semantico | hierarquico.

Cada tecnica usa um componente NATIVO do Haystack. Embedding: Ollama (nomic-embed-text).
"""

import asyncio
import re
from functools import partial

from haystack import Document

from . import config
from .log import obter_logger

log = obter_logger(__name__)

LIMIAR_ENTIDADES = 30
MIN_PALAVRAS_GRAFO = 800
TECNICAS_CHUNK = {"fixo", "recursivo", "sentenca_janela", "semantico", "hierarquico"}


# ---------------------------------------------------------------------------
# DESTINO (OpenSearch vs LightRAG)
# ---------------------------------------------------------------------------
def _entidades_distintas(conteudo):
    palavras = conteudo.split()
    if len(palavras) < MIN_PALAVRAS_GRAFO:
        return 0
    caps = {w.strip(".,;:()") for w in palavras if w[:1].isupper() and len(w) > 2}
    return len(caps)


def decidir_destino(dados, override="auto"):
    if override in ("opensearch", "grafo"):
        log.info("Destino forcado pelo usuario: %s", override)
        return override, f"forcado pelo usuario (override={override})"
    n_ent = _entidades_distintas(dados.get("conteudo", ""))
    log.debug("Heuristica de destino: %d entidades distintas (limiar=%d)", n_ent, LIMIAR_ENTIDADES)
    if n_ent >= LIMIAR_ENTIDADES:
        return "grafo", f"texto longo e rico em entidades ({n_ent} distintas >= {LIMIAR_ENTIDADES}) -> multi-hop"
    return "opensearch", f"texto/tabela direto ({n_ent} entidades distintas < {LIMIAR_ENTIDADES})"


# ---------------------------------------------------------------------------
# AVALIADOR de chunking (so para OpenSearch)
# ---------------------------------------------------------------------------
def _n_titulos(c):
    return len([l for l in c.splitlines() if l.lstrip().startswith("#")])


def _n_artigos(c):
    return len(re.findall(r"\bArt\.?\s*\d+", c))


def avaliar_chunking(dados, override="auto"):
    """Escolhe a melhor tecnica de chunking pela ESTRUTURA do documento (explicavel)."""
    conteudo = dados.get("conteudo", "")
    if dados.get("tabelas"):
        return "tabela", "documento tem tabelas -> 1 chunk por tabela (preserva a estrutura)"
    if override in TECNICAS_CHUNK:
        return override, f"forcado pelo usuario (chunking={override})"

    n_pal, n_tit, n_art = len(conteudo.split()), _n_titulos(conteudo), _n_artigos(conteudo)
    log.debug("Sinais de chunking: %d palavras, %d titulos, %d 'Art.'", n_pal, n_tit, n_art)
    if n_tit >= 3:
        return "hierarquico", f"documento estruturado em secoes ({n_tit} titulos) -> hierarquico"
    if n_art >= 5:
        return "sentenca_janela", f"texto de lei/denso em artigos ({n_art} 'Art.') -> precisao por sentenca"
    if n_pal >= 1500:
        return "semantico", f"texto longo e heterogeneo ({n_pal} palavras) -> cortes por mudanca de topico"
    if n_pal >= 300:
        return "recursivo", f"texto corrido ({n_pal} palavras) -> recursivo (respeita paragrafos/sentencas)"
    return "fixo", f"documento curto ({n_pal} palavras) -> fixo (baseline)"


# ---------------------------------------------------------------------------
# Chunkers (componentes NATIVOS do Haystack)
# ---------------------------------------------------------------------------
def _rodar(splitter, conteudo):
    if hasattr(splitter, "warm_up"):
        splitter.warm_up()
    return splitter.run(documents=[Document(content=conteudo)])["documents"]


def _ollama_doc_embedder():
    from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
    base_url, modelo = config.config_ollama()
    return OllamaDocumentEmbedder(model=modelo, url=base_url)


def chunkar(conteudo, tecnica):
    from haystack.components.preprocessors import (DocumentSplitter,
        EmbeddingBasedDocumentSplitter, HierarchicalDocumentSplitter,
        RecursiveDocumentSplitter)

    log.debug("Chunkando com tecnica '%s' (%d caracteres)", tecnica, len(conteudo))
    if tecnica == "fixo":
        return _rodar(DocumentSplitter(split_by="word", split_length=200, split_overlap=0), conteudo)
    if tecnica == "recursivo":
        return _rodar(RecursiveDocumentSplitter(split_length=200, split_overlap=30, split_unit="word"), conteudo)
    if tecnica == "sentenca_janela":
        # janela = grupos de sentencas com sobreposicao (sentence-window)
        return _rodar(DocumentSplitter(split_by="sentence", split_length=3, split_overlap=1), conteudo)
    if tecnica == "semantico":
        sp = EmbeddingBasedDocumentSplitter(document_embedder=_ollama_doc_embedder(),
                                            sentences_per_group=3, language="pt")
        return _rodar(sp, conteudo)
    if tecnica == "hierarquico":
        nos = _rodar(HierarchicalDocumentSplitter(block_sizes={400, 100}, split_by="word"), conteudo)
        # indexa apenas as FOLHAS (chunks pequenos) - estrutura-aware
        return [d for d in nos if not d.meta.get("__children_ids")] or nos
    # 'tabela' ou desconhecida -> doc inteiro (tabela em markdown ja vem estruturada)
    return [Document(content=conteudo)]


# ---------------------------------------------------------------------------
# Gravacao: OpenSearch
# ---------------------------------------------------------------------------
def _store_opensearch():
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
    os_cfg = config.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    return OpenSearchDocumentStore(hosts=os_cfg["url"], index=os_cfg["indice"],
                                   embedding_dim=config.dimensao_embedding(),
                                   http_auth=auth, use_ssl=False, verify_certs=False)


def indexar_opensearch(docs, meta):
    for d in docs:
        d.meta.update(meta)
    log.info("Gerando embeddings (Ollama) para %d chunk(s)...", len(docs))
    embedder = _ollama_doc_embedder()
    if hasattr(embedder, "warm_up"):
        embedder.warm_up()
    docs_emb = embedder.run(documents=docs)["documents"]
    log.info("Gravando %d documento(s) no OpenSearch (indice '%s')...",
             len(docs_emb), config.config_opensearch()["indice"])
    _store_opensearch().write_documents(docs_emb)
    return len(docs_emb)


# ---------------------------------------------------------------------------
# Gravacao: LightRAG (grafo)
# ---------------------------------------------------------------------------
async def _criar_lightrag():
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_embed
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    api_key, modelo, base_url = config.config_llm()
    o_base, o_modelo = config.config_ollama()

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(modelo, prompt, system_prompt=system_prompt,
                                              history_messages=history_messages or [],
                                              api_key=api_key, base_url=base_url, **kwargs)

    rag = LightRAG(working_dir=str(config.PASTA_RAG_STORAGE), llm_model_func=llm_func,
                   embedding_func=EmbeddingFunc(embedding_dim=config.dimensao_embedding(),
                       max_token_size=8192,
                       func=partial(ollama_embed.func, embed_model=o_modelo, host=o_base)))
    await rag.initialize_storages()
    return rag


def rodar_async(coro_factory):
    """Roda uma corrotina com seguranca, HAJA ou NAO um event loop ativo.

    O LightRAG e assincrono. Chamar asyncio.run() dentro de um endpoint 'async def'
    do FastAPI quebra ('asyncio.run() cannot be called from a running event loop').
    Aqui: se nao ha loop, usa asyncio.run direto; se ja ha um loop rodando, executa
    numa thread separada (que tem o proprio loop).
    """
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())  # sem loop ativo (caso comum)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro_factory())).result()


def indexar_grafo(conteudo):
    async def _run():
        log.info("Construindo grafo no LightRAG (varias chamadas de LLM, pode demorar)...")
        rag = await _criar_lightrag()
        try:
            await rag.ainsert(conteudo)
        finally:
            await rag.finalize_storages()
    rodar_async(_run)
    log.info("Grafo atualizado no LightRAG (storage: %s)", config.PASTA_RAG_STORAGE)
    return 1


# ---------------------------------------------------------------------------
# Orquestra a indexacao
# ---------------------------------------------------------------------------
def indexar(dados, meta, destino_override="auto", chunking_override="auto"):
    destino, motivo_destino = decidir_destino(dados, destino_override)
    log.info("Destino de indexacao: %s (%s)", destino, motivo_destino)
    if destino == "grafo":
        n = indexar_grafo(dados.get("conteudo", ""))
        return {"destino": destino, "motivo_destino": motivo_destino,
                "chunking": "grafo (LightRAG gerencia)", "motivo_chunking": "destino=grafo",
                "n_chunks": n}
    tecnica, motivo_chunking = avaliar_chunking(dados, chunking_override)
    log.info("Tecnica de chunking: %s (%s)", tecnica, motivo_chunking)
    docs = chunkar(dados.get("conteudo", ""), tecnica)
    n = indexar_opensearch(docs, meta)
    log.info("Indexacao concluida: %d chunk(s) no OpenSearch", n)
    return {"destino": destino, "motivo_destino": motivo_destino,
            "chunking": tecnica, "motivo_chunking": motivo_chunking, "n_chunks": n}
