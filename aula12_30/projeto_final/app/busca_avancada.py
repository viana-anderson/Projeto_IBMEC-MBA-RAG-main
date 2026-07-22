"""
busca_avancada.py - Tecnicas de Query Enhancement na busca OpenSearch (Aula 7 -> projeto final).

Tecnicas selecionaveis no /consulta (parametro 'tecnica'):
  - baseline    : 1 embedding -> 1 busca densa (sem reescrita).
  - multi_query : LLM gera N variacoes da pergunta -> busca cada uma -> DEDUP por id/score.
  - rag_fusion  : igual ao multi_query, mas funde os rankings com RRF (Reciprocal Rank Fusion).
  - step_back   : LLM gera uma pergunta mais GERAL -> busca [especifica + geral] -> dedup.

Tudo roda DENTRO de um pipeline Haystack, entao a auto-instrumentacao do LangFuse captura
ate as chamadas de LLM que reescrevem a pergunta (no mesmo trace).
"""

from typing import List

from haystack import Document, component

from . import config, indexacao, prompts
from .log import obter_logger

log = obter_logger(__name__)

TECNICAS = ("baseline", "multi_query", "rag_fusion", "step_back")

# Os prompts (rag / variacoes / stepback) sao lidos EM RUNTIME de prompts.get_prompts(),
# para refletirem o que o aluno editar na aba Configuracoes do Gradio.


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
    """Reciprocal Rank Fusion: soma 1/(k + posicao) de cada lista; ordena por pontuacao."""
    pontos, ref = {}, {}
    for docs in listas:
        for posicao, d in enumerate(docs):
            pontos[d.id] = pontos.get(d.id, 0.0) + 1.0 / (k + posicao + 1)
            ref[d.id] = d
    return sorted(ref.values(), key=lambda d: pontos[d.id], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Componentes customizados (colocam a tecnica inteira dentro do pipeline)
# ---------------------------------------------------------------------------
@component
class MontarConsultas:
    """Transforma a saida do LLM (texto) na lista de consultas a buscar."""

    def __init__(self, modo="variacoes", n=4):
        self.modo = modo      # 'variacoes' (multi-query/rag-fusion) ou 'stepback'
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
    """Busca cada consulta no OpenSearch (Ollama) e funde (dedup ou RRF)."""

    def __init__(self, document_store, top_k=5, modo="dedup"):
        from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
        from haystack_integrations.components.retrievers.opensearch import (
            OpenSearchEmbeddingRetriever,
        )

        base_url, modelo = config.config_ollama()
        self.embedder = OllamaTextEmbedder(model=modelo, url=base_url)
        self.retriever = OpenSearchEmbeddingRetriever(document_store=document_store, top_k=top_k)
        self.top_k = top_k
        self.modo = modo      # 'dedup' ou 'rrf'

    def warm_up(self):
        if hasattr(self.embedder, "warm_up"):
            self.embedder.warm_up()

    @component.output_types(documents=List[Document])
    def run(self, queries: List[str]):
        listas = []
        for q in queries:
            emb = self.embedder.run(text=q)["embedding"]
            listas.append(self.retriever.run(query_embedding=emb)["documents"])
        docs = fundir_rrf(listas, self.top_k) if self.modo == "rrf" else dedup_por_id(listas, self.top_k)
        log.info("Busca multipla: %d consultas -> %d docs (modo=%s)", len(queries), len(docs), self.modo)
        return {"documents": docs}


# ---------------------------------------------------------------------------
# Builder do pipeline por tecnica
# ---------------------------------------------------------------------------
def construir(tecnica, top_k, pergunta):
    """Monta o pipeline Haystack da tecnica e devolve (pipe, inputs, chave_dos_docs)."""
    from haystack import Pipeline
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack.utils import Secret
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
    from haystack_integrations.components.retrievers.opensearch import OpenSearchEmbeddingRetriever

    if tecnica not in TECNICAS:
        tecnica = "baseline"
    base_ollama, modelo_emb = config.config_ollama()
    api_key, gmodelo, llm_base = config.config_llm()
    store = indexacao._store_opensearch()
    p = prompts.get_prompts()   # prompts atuais (editaveis na aba Configuracoes)

    def novo_llm(temp, max_tokens):
        return OpenAIGenerator(api_key=Secret.from_token(api_key), model=gmodelo,
                               api_base_url=llm_base,
                               generation_kwargs={"temperature": temp, "max_tokens": max_tokens})

    pipe = Pipeline()
    if config.langfuse_configurado():
        from haystack_integrations.components.connectors.langfuse import LangfuseConnector
        pipe.add_component("tracer", LangfuseConnector(f"busca-{tecnica}-aula12"))
    # geracao final (comum a todas as tecnicas)
    pipe.add_component("prompt", PromptBuilder(template=p["rag"], required_variables="*"))
    pipe.add_component("llm", novo_llm(0.2, 500))
    pipe.connect("prompt.prompt", "llm.prompt")

    if tecnica == "baseline":
        pipe.add_component("embedder", OllamaTextEmbedder(model=modelo_emb, url=base_ollama))
        pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
        pipe.connect("embedder.embedding", "retriever.query_embedding")
        pipe.connect("retriever.documents", "prompt.documents")
        inputs = {"embedder": {"text": pergunta}, "prompt": {"pergunta": pergunta}}
        return pipe, inputs, "retriever"

    # tecnicas com reescrita de query
    if tecnica == "step_back":
        pipe.add_component("rw_prompt", PromptBuilder(template=p["stepback"], required_variables="*"))
        modo_montar, modo_fusao = "stepback", "dedup"
    else:  # multi_query | rag_fusion
        pipe.add_component("rw_prompt", PromptBuilder(template=p["variacoes"], required_variables="*"))
        modo_montar = "variacoes"
        modo_fusao = "rrf" if tecnica == "rag_fusion" else "dedup"
    pipe.add_component("rw_llm", novo_llm(0.3, 300))
    pipe.add_component("montar", MontarConsultas(modo=modo_montar, n=4))
    pipe.add_component("buscar", BuscarMultiplas(store, top_k=top_k, modo=modo_fusao))
    pipe.connect("rw_prompt.prompt", "rw_llm.prompt")
    pipe.connect("rw_llm.replies", "montar.textos")
    pipe.connect("montar.queries", "buscar.queries")
    pipe.connect("buscar.documents", "prompt.documents")
    inputs = {"rw_prompt": {"pergunta": pergunta}, "montar": {"question": pergunta},
              "prompt": {"pergunta": pergunta}}
    return pipe, inputs, "buscar"
