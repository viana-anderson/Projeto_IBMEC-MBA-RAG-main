"""
busca_avancada.py - Query Enhancement + Rerank na busca OpenSearch (Aula 7 + Aula 3/4).

Tecnicas (parametro 'tecnica'):
  - baseline    : 1 embedding -> 1 busca densa.
  - multi_query : LLM gera N variacoes -> busca cada uma -> fusao.
  - rag_fusion  : idem, com fusao por RRF (padrao).
  - step_back   : LLM gera pergunta mais geral -> busca [especifica + geral] -> fusao.

Rerank/fusao (parametro 'rerank', so OpenSearch):
  - rrf     : Reciprocal Rank Fusion (soma 1/(k+pos)).
  - minmax  : normaliza os scores de cada lista para [0,1] e soma.
  - modelo  : cross-encoder (BAAI/bge-reranker-v2-m3) reordena o top-N (Aula 3).

Tudo roda dentro de um pipeline Haystack (o LangFuse rastreia inclusive a reescrita).
"""

from typing import List

from haystack import Document, component

from . import config, indexacao, prompts
from .log import obter_logger

log = obter_logger(__name__)

TECNICAS = ("baseline", "multi_query", "rag_fusion", "step_back")
RERANKS = ("rrf", "minmax", "modelo")

# Os prompts sao lidos EM RUNTIME de prompts.get_prompts() (editaveis na aba Configuracoes).


# ---------------------------------------------------------------------------
# Fusao de resultados
# ---------------------------------------------------------------------------
def dedup_por_id(listas, top_k):
    """Mantem cada doc uma vez, com o MAIOR score; ordena desc e corta no top_k."""
    melhor = {}
    for docs in listas:
        for d in docs:
            atual = melhor.get(d.id)
            if atual is None or (d.score or 0) > (atual.score or 0):
                melhor[d.id] = d
    return sorted(melhor.values(), key=lambda d: (d.score or 0), reverse=True)[:top_k]


def fundir_rrf(listas, top_k, k=60):
    """Reciprocal Rank Fusion: soma 1/(k + posicao) de cada lista."""
    pontos, ref = {}, {}
    for docs in listas:
        for posicao, d in enumerate(docs):
            pontos[d.id] = pontos.get(d.id, 0.0) + 1.0 / (k + posicao + 1)
            ref[d.id] = d
    return sorted(ref.values(), key=lambda d: pontos[d.id], reverse=True)[:top_k]


def fundir_minmax(listas, top_k):
    """Min-Max: normaliza os scores de cada lista para [0,1] e soma por documento."""
    pontos, ref = {}, {}
    for docs in listas:
        scores = [(d.score or 0.0) for d in docs]
        lo, hi = (min(scores), max(scores)) if scores else (0.0, 0.0)
        faixa = (hi - lo) or 1.0
        for d in docs:
            norm = ((d.score or 0.0) - lo) / faixa
            pontos[d.id] = pontos.get(d.id, 0.0) + norm
            ref[d.id] = d
    return sorted(ref.values(), key=lambda d: pontos[d.id], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Componentes customizados
# ---------------------------------------------------------------------------
@component
class MontarConsultas:
    """Transforma a saida do LLM (texto) na lista de consultas a buscar."""

    def __init__(self, modo="variacoes", n=4):
        self.modo = modo
        self.n = n

    @component.output_types(queries=List[str])
    def run(self, question: str, textos: List[str]):
        texto = textos[0] if textos else ""
        if self.modo == "stepback":
            queries = [question] + ([texto.strip()] if texto.strip() else [])
        else:
            variacoes = [v.strip(" -.\t") for v in texto.splitlines() if v.strip()][: self.n]
            queries = [question] + variacoes
        return {"queries": queries}


@component
class BuscarMultiplas:
    """Busca cada consulta no OpenSearch (Ollama) e funde (rrf | minmax | dedup)."""

    def __init__(self, document_store, top_k=5, modo="rrf"):
        from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
        from haystack_integrations.components.retrievers.opensearch import (
            OpenSearchEmbeddingRetriever,
        )

        base_url, modelo = config.config_ollama()
        self.embedder = OllamaTextEmbedder(model=modelo, url=base_url)
        self.retriever = OpenSearchEmbeddingRetriever(document_store=document_store, top_k=top_k)
        self.top_k = top_k
        self.modo = modo

    def warm_up(self):
        if hasattr(self.embedder, "warm_up"):
            self.embedder.warm_up()

    @component.output_types(documents=List[Document])
    def run(self, queries: List[str]):
        listas = []
        for q in queries:
            emb = self.embedder.run(text=q)["embedding"]
            listas.append(self.retriever.run(query_embedding=emb)["documents"])
        if self.modo == "minmax":
            docs = fundir_minmax(listas, self.top_k)
        elif self.modo == "dedup":
            docs = dedup_por_id(listas, self.top_k)
        else:
            docs = fundir_rrf(listas, self.top_k)
        log.info("Busca multipla: %d consultas -> %d docs (fusao=%s)", len(queries), len(docs), self.modo)
        return {"documents": docs}


def _criar_reranker(top_k):
    """Cross-encoder de reranking (Aula 3). Modelo via env RERANK_MODEL."""
    import os

    from haystack.components.rankers import TransformersSimilarityRanker
    modelo = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    return TransformersSimilarityRanker(model=modelo, top_k=top_k)


# ---------------------------------------------------------------------------
# Builder do pipeline por tecnica + rerank
# ---------------------------------------------------------------------------
def construir(tecnica, top_k, pergunta, rerank="rrf"):
    """Monta o pipeline Haystack e devolve (pipe, inputs, chave_dos_docs)."""
    from haystack import Pipeline
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack.utils import Secret
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
    from haystack_integrations.components.retrievers.opensearch import OpenSearchEmbeddingRetriever

    if tecnica not in TECNICAS:
        tecnica = "baseline"
    if rerank not in RERANKS:
        rerank = "rrf"
    base_ollama, modelo_emb = config.config_ollama()
    api_key, gmodelo, llm_base = config.config_llm()
    store = indexacao._store_opensearch()
    p = prompts.get_prompts()

    usa_modelo = (rerank == "modelo")
    pool = top_k * 3 if usa_modelo else top_k        # recupera mais para o reranker escolher
    modo_fusao = rerank if rerank in ("rrf", "minmax") else "rrf"

    def novo_llm(temp, max_tokens):
        return OpenAIGenerator(api_key=Secret.from_token(api_key), model=gmodelo,
                               api_base_url=llm_base,
                               generation_kwargs={"temperature": temp, "max_tokens": max_tokens})

    pipe = Pipeline()
    if config.langfuse_configurado():
        from haystack_integrations.components.connectors.langfuse import LangfuseConnector
        pipe.add_component("tracer", LangfuseConnector(f"busca-{tecnica}-{rerank}-aula12"))
    pipe.add_component("prompt", PromptBuilder(template=p["rag"], required_variables="*"))
    pipe.add_component("llm", novo_llm(0.2, 500))
    pipe.connect("prompt.prompt", "llm.prompt")
    if usa_modelo:
        pipe.add_component("reranker", _criar_reranker(top_k))
        pipe.connect("reranker.documents", "prompt.documents")

    inputs = {"prompt": {"pergunta": pergunta}}
    if usa_modelo:
        inputs["reranker"] = {"query": pergunta}

    if tecnica == "baseline":
        pipe.add_component("embedder", OllamaTextEmbedder(model=modelo_emb, url=base_ollama))
        pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=pool))
        pipe.connect("embedder.embedding", "retriever.query_embedding")
        pipe.connect("retriever.documents", "reranker.documents" if usa_modelo else "prompt.documents")
        inputs["embedder"] = {"text": pergunta}
        return pipe, inputs, ("reranker" if usa_modelo else "retriever")

    # tecnicas com reescrita de query
    if tecnica == "step_back":
        pipe.add_component("rw_prompt", PromptBuilder(template=p["stepback"], required_variables="*"))
        modo_montar = "stepback"
    else:  # multi_query | rag_fusion
        pipe.add_component("rw_prompt", PromptBuilder(template=p["variacoes"], required_variables="*"))
        modo_montar = "variacoes"
    pipe.add_component("rw_llm", novo_llm(0.3, 300))
    pipe.add_component("montar", MontarConsultas(modo=modo_montar, n=4))
    pipe.add_component("buscar", BuscarMultiplas(store, top_k=pool, modo=modo_fusao))
    pipe.connect("rw_prompt.prompt", "rw_llm.prompt")
    pipe.connect("rw_llm.replies", "montar.textos")
    pipe.connect("montar.queries", "buscar.queries")
    pipe.connect("buscar.documents", "reranker.documents" if usa_modelo else "prompt.documents")
    inputs["rw_prompt"] = {"pergunta": pergunta}
    inputs["montar"] = {"question": pergunta}
    return pipe, inputs, ("reranker" if usa_modelo else "buscar")
