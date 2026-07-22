"""
main.py - API FastAPI do Projeto Final (Aula 12).

Endpoints:
  POST /ingestao  - envia um documento (PDF/DOCX/XLSX/imagem/TXT). O AGENTE decide a
                    tecnica de extracao; a heuristica decide a estrategia/destino de
                    indexacao (OpenSearch ou LightRAG). Retorna o RELATORIO da decisao.
  POST /consulta  - pergunta -> resposta RAG (roteada para o storage certo) + fontes.
  GET  /health    - status dos componentes.
  GET  /metrics   - contadores simples (ingestoes, consultas, erros).

Autenticacao opcional por API key (header X-API-Key) se API_KEYS estiver no .env.

Rodar:  uvicorn app.main:app --reload    (de dentro de projeto_final/)
"""

import asyncio
import time

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from . import config, consulta, extracao, grafo, indexacao, prompts
from .log import configurar_logging, obter_logger
from .modelos import (ConsultaRequest, ConsultaResponse, IngestaoResponse,
                      PromptsConfig, RelatorioIngestao)

configurar_logging()           # le LOG_LEVEL do .env (DEBUG p/ verbosidade maxima)
log = obter_logger(__name__)

app = FastAPI(title="RAG Juridico - Projeto Final (Aula 12)",
              description="Ingestao inteligente (agente decide extracao) + RAG (OpenSearch/LightRAG)")

METRICAS = {"ingestoes": 0, "consultas": 0, "erros": 0, "inicio": time.time()}


def _checar_api_key(x_api_key):
    chaves = config.api_keys()
    if chaves and x_api_key not in chaves:
        raise HTTPException(status_code=401, detail="API key invalida (header X-API-Key).")


def _processar_ingestao(filename, conteudo_bytes, estrategia, chunking):
    """Trabalho PESADO e SINCRONO da ingestao (extracao + indexacao).

    Roda numa THREAD (via asyncio.to_thread no endpoint) por dois motivos:
      1) o LightRAG usa asyncio.run() internamente, que NAO pode rodar dentro do
         event loop do FastAPI ('asyncio.run() cannot be called from a running event loop');
         numa thread separada nao ha loop ativo, entao funciona.
      2) nao trava o event loop durante Docling/embeddings/LightRAG (operacoes longas).
    """
    destino = config.PASTA_UPLOADS / filename
    t0 = time.time()
    log.info("== /ingestao recebido: arquivo=%s (estrategia=%s, chunking=%s) ==",
             filename, estrategia, chunking)
    try:
        destino.write_bytes(conteudo_bytes)
        # 1) AGENTE decide a tecnica e extrai
        sinais, tecnica, complexidade, motivo, dados = extracao.decidir_e_extrair(str(destino))
        # 2) HEURISTICA decide destino + (no OpenSearch) a melhor tecnica de chunking, e indexa
        estr = indexacao.indexar(dados, meta={"arquivo": filename},
                                 destino_override=estrategia, chunking_override=chunking)
        extracao.limpar_cache(str(destino))
        METRICAS["ingestoes"] += 1
        log.info("== /ingestao OK: arquivo=%s, destino=%s, chunking=%s, chunks=%d (%.1fs) ==",
                 filename, estr["destino"], estr["chunking"], estr["n_chunks"], time.time() - t0)
        relatorio = RelatorioIngestao(
            arquivo=filename, complexidade=complexidade, tecnica_extracao=tecnica,
            motivo_extracao=motivo, estrutura=sinais,
            destino=estr["destino"], motivo_destino=estr["motivo_destino"],
            chunking=estr["chunking"], motivo_chunking=estr["motivo_chunking"],
            n_chunks=estr["n_chunks"], n_caracteres=len(dados.get("conteudo", "")))
        return IngestaoResponse(ok=True, relatorio=relatorio)
    except Exception as e:
        METRICAS["erros"] += 1
        log.exception("== /ingestao FALHOU: arquivo=%s ==", filename)
        return IngestaoResponse(ok=False, erro=str(e))


@app.post("/ingestao", response_model=IngestaoResponse)
async def ingestao(arquivo: UploadFile = File(...), estrategia: str = "auto",
                   chunking: str = "auto", x_api_key: str = Header(default="")):
    """Recebe um documento, decide como extrair e indexar, e devolve o relatorio.

    estrategia: auto | opensearch | grafo   (destino da indexacao)
    chunking:   auto | fixo | recursivo | sentenca_janela | semantico | hierarquico
    """
    _checar_api_key(x_api_key)
    conteudo_bytes = await arquivo.read()
    # descarrega o trabalho sincrono/pesado numa thread (ver _processar_ingestao)
    return await asyncio.to_thread(_processar_ingestao, arquivo.filename,
                                   conteudo_bytes, estrategia, chunking)


@app.post("/consulta", response_model=ConsultaResponse)
def consulta_endpoint(req: ConsultaRequest, x_api_key: str = Header(default="")):
    """Consulta RAG: roteia para OpenSearch ou LightRAG e gera a resposta."""
    _checar_api_key(x_api_key)
    log.info("== /consulta recebida: destino=%s, top_k=%d, tecnica=%s ==",
             req.destino, req.top_k, req.tecnica)
    try:
        resposta, fontes, destino = consulta.consultar(req.pergunta, req.destino,
                                                       req.top_k, req.tecnica)
        METRICAS["consultas"] += 1
        log.info("== /consulta OK: destino_usado=%s, %d fonte(s) ==", destino, len(fontes))
        return ConsultaResponse(pergunta=req.pergunta, resposta=resposta,
                                destino_usado=destino, fontes=fontes)
    except Exception as e:
        METRICAS["erros"] += 1
        log.exception("== /consulta FALHOU ==")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph")
def graph(limite: int = 150, x_api_key: str = Header(default="")):
    """Dados do grafo de conhecimento do LightRAG (JSON): existencia, estatisticas, nos/arestas.

    'exists' indica se ha grafo gravado - a interface usa isso para mostrar (ou nao) a aba.
    'limite' = numero maximo de nos exibidos (os de maior grau).
    """
    _checar_api_key(x_api_key)
    try:
        dados = grafo.ler_grafo(limite_nos=limite)
        log.info("== /graph: exists=%s, nos=%d, arestas=%d ==",
                 dados["exists"], dados["n_nodes"], dados["n_edges"])
        return dados
    except Exception as e:
        log.exception("== /graph FALHOU ==")
        return {"exists": False, "erro": str(e), "n_nodes": 0, "n_edges": 0,
                "top_hubs": [], "nodes": [], "edges": []}


@app.get("/graph/html", response_class=HTMLResponse)
def graph_html(limite: int = 150):
    """Visualizacao interativa (vis-network) do grafo - usada num iframe pela interface."""
    try:
        return grafo.html_vis(limite_nos=limite)
    except Exception as e:
        log.exception("== /graph/html FALHOU ==")
        return HTMLResponse(f"<html><body><p>Erro ao montar o grafo: {e}</p></body></html>")


@app.get("/config/prompts")
def get_prompts_endpoint(x_api_key: str = Header(default="")):
    """Retorna os prompts atuais (editaveis na aba Configuracoes do Gradio)."""
    _checar_api_key(x_api_key)
    return prompts.get_prompts()


@app.put("/config/prompts")
def set_prompts_endpoint(body: PromptsConfig, x_api_key: str = Header(default="")):
    """Atualiza os prompts enviados (os demais ficam como estao) e persiste."""
    _checar_api_key(x_api_key)
    atualizados = prompts.set_prompts(body.model_dump(exclude_none=True))
    log.info("Prompts atualizados via API: %s", list(body.model_dump(exclude_none=True)))
    return atualizados


@app.post("/config/prompts/reset")
def reset_prompts_endpoint(x_api_key: str = Header(default="")):
    """Restaura os prompts padrao."""
    _checar_api_key(x_api_key)
    log.info("Prompts restaurados ao padrao via API")
    return prompts.reset()


@app.get("/health")
def health():
    """Status rapido dos componentes (sem derrubar a API se algo falhar)."""
    estado = {"api": "ok"}
    try:
        estado["opensearch"] = f"ok ({indexacao._store_opensearch().count_documents()} docs)"
    except Exception as e:
        estado["opensearch"] = f"falhou: {e}"
    api_key, modelo, base_url = config.config_llm()
    estado["llm"] = (f"{config.provedor_llm()} | {modelo} @ {base_url}"
                     if api_key else "sem LLM_API_KEY/GROQ_API_KEY")
    estado["embedding"] = config.config_ollama()[1]
    estado["langfuse"] = "on" if config.langfuse_configurado() else "off"
    return estado


@app.get("/metrics")
def metrics():
    m = dict(METRICAS)
    m["uptime_s"] = round(time.time() - METRICAS["inicio"], 1)
    return m
