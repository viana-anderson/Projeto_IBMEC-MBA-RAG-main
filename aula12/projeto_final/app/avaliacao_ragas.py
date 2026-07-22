"""
avaliacao_ragas.py - Geracao de dataset de avaliacao e metricas RAGAS.

  gerar_dataset(nome, origem, n)  -> le o OpenSearch (ou o LightRAG), e para cada documento
      usa a LLM configurada (agnostica) para criar (pergunta, resposta_referencia). Salva um
      JSON em datasets_ragas/<nome>.json com o gabarito (documentos_relevantes) para as
      metricas de recuperacao E a resposta de referencia para o RAGAS.

  listar_datasets() / carregar_dataset(nome)  -> para a interface escolher o dataset.

  avaliar_ragas(itens)  -> roda o RAGAS (Faithfulness, Answer Relevancy, Context Precision/Recall)
      com juiz LLM agnostico (OpenAI-compativel) + embeddings Ollama. Retorna medias + por item.
      (RAGAS e opcional: se a lib nao estiver instalada, devolve um aviso e segue.)
"""

import json
import re

from . import config, indexacao
from .log import obter_logger

log = obter_logger(__name__)

PASTA_DATASETS = config.PASTA_PROJETO / "datasets_ragas"
PASTA_DATASETS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM (agnostica) para gerar as perguntas
# ---------------------------------------------------------------------------
def _llm_texto(prompt, temperatura=0.3, max_tokens=400):
    from openai import OpenAI

    api_key, modelo, base_url = config.config_llm()
    cli = OpenAI(api_key=api_key, base_url=base_url)
    r = cli.chat.completions.create(model=modelo, temperature=temperatura,
                                    max_tokens=max_tokens,
                                    messages=[{"role": "user", "content": prompt}])
    return (r.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Leitura dos documentos (OpenSearch ou LightRAG)
# ---------------------------------------------------------------------------
def _ler_documentos(origem, n):
    """Devolve [{'id':..., 'texto':...}] a partir do storage escolhido."""
    if origem == "grafo":
        # LightRAG guarda os chunks em kv_store_text_chunks.json
        caminho = config.PASTA_RAG_STORAGE / "kv_store_text_chunks.json"
        if not caminho.exists():
            raise FileNotFoundError("Nao ha chunks do LightRAG (kv_store_text_chunks.json).")
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        itens = [{"id": cid, "texto": (v.get("content") or "")}
                 for cid, v in dados.items() if (v.get("content") or "").strip()]
        return itens[:n]
    # OpenSearch: le os documentos indexados
    docs = indexacao._store_opensearch().filter_documents()
    return [{"id": d.id, "texto": d.content or ""} for d in docs if (d.content or "").strip()][:n]


_PROMPT_QA = (
    "Voce cria pares de pergunta e resposta para avaliar um sistema de busca juridica. "
    "Com base APENAS no trecho abaixo, gere UMA pergunta que um usuario faria (linguagem "
    "natural, sem copiar numeros/termos exatos) e a resposta correta e curta.\n"
    "Responda em JSON: {{\"pergunta\": \"...\", \"resposta\": \"...\"}}\n\n"
    "Trecho:\n{trecho}"
)


def _gerar_qa(texto):
    saida = _llm_texto(_PROMPT_QA.format(trecho=texto[:2000]))
    bruto = re.sub(r"^```(json)?|```$", "", saida.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(bruto)
        return (obj.get("pergunta", "").strip(), obj.get("resposta", "").strip())
    except Exception:
        return ("", "")


# ---------------------------------------------------------------------------
# Dataset: gerar / listar / carregar
# ---------------------------------------------------------------------------
def gerar_dataset(nome, origem="opensearch", n=15):
    nome = re.sub(r"[^\w\-]+", "_", (nome or "dataset").strip()) or "dataset"
    log.info("Gerando dataset RAGAS '%s' (origem=%s, n=%d)...", nome, origem, n)
    docs = _ler_documentos(origem, n)
    itens = []
    for d in docs:
        pergunta, resposta = _gerar_qa(d["texto"])
        if pergunta:
            itens.append({"pergunta": pergunta, "resposta_referencia": resposta,
                          "documentos_relevantes": [d["id"]],
                          "contexto_referencia": d["texto"][:1500]})
    caminho = PASTA_DATASETS / f"{nome}.json"
    caminho.write_text(json.dumps({"nome": nome, "origem": origem, "itens": itens},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Dataset '%s' salvo com %d itens em %s", nome, len(itens), caminho)
    return {"nome": nome, "origem": origem, "n_itens": len(itens), "arquivo": str(caminho)}


def listar_datasets():
    return sorted(p.stem for p in PASTA_DATASETS.glob("*.json"))


def carregar_dataset(nome):
    caminho = PASTA_DATASETS / f"{nome}.json"
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


def listar_perguntas(nome):
    """Perguntas (gabaritos) de um dataset - para o dropdown de gabarito na Consulta."""
    ds = carregar_dataset(nome)
    return [it.get("pergunta", "") for it in (ds or {}).get("itens", []) if it.get("pergunta")]


# ---------------------------------------------------------------------------
# Avaliacao RAGAS (opcional)
# ---------------------------------------------------------------------------
def ragas_disponivel():
    import importlib.util
    return importlib.util.find_spec("ragas") is not None


def avaliar_ragas(itens):
    """itens: [{pergunta, resposta_referencia, contextos:[str], resposta}]. Retorna medias+por_item."""
    if not itens:
        return {"ok": False, "erro": "sem itens para avaliar"}
    if not ragas_disponivel():
        return {"ok": False, "erro": "ragas nao instalado (pip install ragas langchain-openai langchain-ollama)"}
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                                    LLMContextRecall, ResponseRelevancy)

        api_key, modelo, base_url = config.config_llm()
        o_base, o_modelo = config.config_ollama()
        juiz = LangchainLLMWrapper(ChatOpenAI(model=modelo, api_key=api_key,
                                              base_url=base_url, temperature=0))
        emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=o_modelo, base_url=o_base))

        amostras = [SingleTurnSample(user_input=i["pergunta"],
                                     retrieved_contexts=i.get("contextos", []),
                                     response=i.get("resposta", ""),
                                     reference=i.get("resposta_referencia", "")) for i in itens]
        # Faithfulness e Answer Relevancy NAO precisam de referencia (valem p/ consulta simples).
        # Context Precision/Recall precisam de referencia -> so quando TODOS os itens tem gabarito.
        metricas = [Faithfulness(), ResponseRelevancy(strictness=1)]
        tem_referencia = all((i.get("resposta_referencia") or "").strip() for i in itens)
        if tem_referencia:
            metricas += [LLMContextPrecisionWithReference(), LLMContextRecall()]
        resultado = evaluate(EvaluationDataset(samples=amostras), metrics=metricas,
                             llm=juiz, embeddings=emb)
        df = resultado.to_pandas()
        cols = [c for c in df.columns if c not in ("user_input", "retrieved_contexts",
                                                   "response", "reference")]
        medias = {c: round(float(df[c].mean()), 4) for c in cols}
        por_item = df[cols].round(4).to_dict(orient="records")
        return {"ok": True, "medias": medias, "por_item": por_item}
    except Exception as e:
        log.exception("RAGAS falhou")
        return {"ok": False, "erro": str(e)}
