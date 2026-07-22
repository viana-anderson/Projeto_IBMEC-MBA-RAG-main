"""
extracao.py - Extracao INTELIGENTE de documentos (o LLM decide a tecnica).

Fluxo:
  1) probe(): le 'sinais' baratos do documento (extensao, paginas, texto, imagens).
  2) Um LLM (Groq, tool-calling) recebe esses sinais e ESCOLHE a ferramenta de extracao:
       - extrair_planilha : XLSX/CSV (pandas)
       - extrair_texto    : PDF nativo/DOCX/TXT com camada de texto (Docling, sem OCR)
       - extrair_com_ocr  : PDF escaneado / imagens / paginas com figuras (Docling + OCR)
  3) A extracao da tecnica escolhida e executada e o conteudo vai para o cache.

Por que UMA chamada com tool_choice='required' (e nao o laco Agent/ReAct)?
  A decisao aqui e uma escolha unica (qual extrator usar). Forcar 'tool_choice=required'
  faz o Groq restringir a geracao ao tool-call ESTRUTURADO, evitando o bug
  'tool_use_failed' em que o llama emite a chamada como TEXTO (<function=...>{...}</function>).
  Se mesmo assim falhar, ha um FALLBACK HEURISTICO deterministico pelos sinais do probe.

Lazy imports: docling/pandas/fitz so sao carregados quando a extracao roda.
"""

import json
from pathlib import Path

from haystack.dataclasses import ChatMessage

from . import config, prompts
from .log import obter_logger

log = obter_logger(__name__)

# cache: caminho -> {"conteudo": str, "tabelas": list, "tecnica": str}
_CACHE = {}

EXT_PLANILHA = {".xlsx", ".xls", ".csv", ".tsv"}
EXT_IMAGEM = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
EXT_TEXTO = {".txt", ".md", ".html", ".htm"}


# ---------------------------------------------------------------------------
# 1) Sinais do documento (probe barato, sem LLM)
# ---------------------------------------------------------------------------
def probe(caminho):
    p = Path(caminho)
    ext = p.suffix.lower()
    sinais = {"extensao": ext, "n_paginas": 0, "n_chars_texto": 0,
              "tem_imagens": False, "eh_planilha": ext in EXT_PLANILHA,
              "eh_imagem": ext in EXT_IMAGEM}
    if ext == ".pdf":
        try:
            try:
                import fitz
            except ImportError:
                import pymupdf as fitz
            doc = fitz.open(caminho)
            sinais["n_paginas"] = doc.page_count
            chars, imgs = 0, 0
            for page in doc:
                chars += len(page.get_text())
                imgs += len(page.get_images())
            doc.close()
            sinais["n_chars_texto"] = chars
            sinais["tem_imagens"] = imgs > 0
            # heuristica: pouco texto + imagens => provavelmente escaneado
            sinais["provavel_escaneado"] = chars < 100 and imgs > 0
        except Exception as e:
            sinais["erro_probe"] = str(e)
    return sinais


# ---------------------------------------------------------------------------
# Implementacoes de extracao
# ---------------------------------------------------------------------------
def _docling_markdown(caminho, com_ocr):
    import os

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = com_ocr
    opts.do_table_structure = True
    # CONTROLE DE MEMORIA (evita 'std::bad_alloc' em paginas grandes):
    #  - menos threads = menos paginas processadas em paralelo = menor pico de RAM
    #  - escala de imagem menor = bitmap menor por pagina
    #  - nao guardar imagens de pagina/figura que nao usamos
    opts.images_scale = float(os.getenv("DOCLING_IMAGE_SCALE", "1.0"))
    opts.generate_page_images = False
    opts.generate_picture_images = False
    try:
        from docling.datamodel.pipeline_options import AcceleratorOptions
        opts.accelerator_options = AcceleratorOptions(
            num_threads=int(os.getenv("DOCLING_NUM_THREADS", "2")))
    except Exception:
        pass  # versoes antigas do Docling podem nao ter AcceleratorOptions
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return conv.convert(str(caminho)).document.export_to_markdown()


def _pymupdf_texto(caminho):
    """Extracao de texto LEVE (PyMuPDF) - fallback de baixa memoria quando o Docling falha."""
    try:
        import fitz
    except ImportError:
        import pymupdf as fitz
    partes = []
    doc = fitz.open(caminho)
    try:
        for page in doc:
            partes.append(page.get_text())
    finally:
        doc.close()
    return "\n\n".join(partes)


def _impl_planilha(caminho):
    import pandas as pd

    p = Path(caminho)
    partes, tabelas = [], []
    if p.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        planilhas = {"dados": pd.read_csv(caminho, sep=sep)}
    else:
        planilhas = pd.read_excel(caminho, sheet_name=None)  # todas as abas
    for nome, df in planilhas.items():
        md = df.to_markdown(index=False)
        partes.append(f"## Aba: {nome}\n{md}")
        tabelas.append({"aba": nome, "linhas": len(df), "colunas": list(df.columns)})
    return "\n\n".join(partes), tabelas


def _impl_texto(caminho):
    ext = Path(caminho).suffix.lower()
    if ext in EXT_TEXTO:
        return Path(caminho).read_text(encoding="utf-8", errors="ignore"), []
    # PDF/DOCX via Docling sem OCR; se o Docling falhar ou vier quase vazio
    # (ex.: 'std::bad_alloc' em pagina pesada), cai para o PyMuPDF (baixa memoria).
    md = ""
    try:
        md = _docling_markdown(caminho, com_ocr=False)
    except Exception as e:
        log.warning("Docling falhou na extracao de texto (%s) -> tentando PyMuPDF", e)
    if ext == ".pdf" and len((md or "").strip()) < 50:
        texto = _pymupdf_texto(caminho)
        if texto.strip():
            log.info("Texto extraido via PyMuPDF (fallback de baixa memoria), %d caracteres",
                     len(texto))
            return texto, []
    return md, []


def _impl_ocr(caminho):
    # PDF escaneado / imagem / figuras -> Docling com OCR
    return _docling_markdown(caminho, com_ocr=True), []


def _guardar(caminho, conteudo, tabelas, tecnica):
    _CACHE[caminho] = {"conteudo": conteudo or "", "tabelas": tabelas or [], "tecnica": tecnica}
    log.info("Extracao concluida: tecnica=%s, %d caracteres, %d tabela(s)",
             tecnica, len(conteudo or ""), len(tabelas or []))
    return json.dumps({"tecnica": tecnica, "n_caracteres": len(conteudo or ""),
                       "n_tabelas": len(tabelas or [])}, ensure_ascii=False)


# tecnica -> funcao de extracao
_IMPL = {"planilha": _impl_planilha, "ocr": _impl_ocr, "texto": _impl_texto}
MAPA_COMPLEXIDADE = {"planilha": "planilha", "ocr": "complexo", "texto": "texto_simples"}
# nome da ferramenta (no LLM) -> tecnica interna
MAPA_TOOL_TECNICA = {"extrair_planilha": "planilha", "extrair_texto": "texto",
                     "extrair_com_ocr": "ocr"}


def escolher_por_sinais(sinais):
    """Escolha DETERMINISTICA da tecnica a partir do probe (fallback, sem LLM)."""
    if sinais.get("eh_planilha"):
        return "planilha"
    if sinais.get("eh_imagem") or sinais.get("provavel_escaneado"):
        return "ocr"
    return "texto"


def _extrair_direto(caminho, tecnica):
    """Roda a extracao da tecnica escolhida e popula o cache."""
    conteudo, tabelas = _IMPL.get(tecnica, _impl_texto)(caminho)
    _guardar(caminho, conteudo, tabelas, tecnica)
    return _CACHE[caminho]


# ---------------------------------------------------------------------------
# 2) Ferramentas (esquema p/ o LLM escolher) - a execucao real e feita por _extrair_direto
# ---------------------------------------------------------------------------
def _param_caminho(desc):
    return {"type": "object",
            "properties": {"caminho": {"type": "string", "description": desc}},
            "required": ["caminho"]}


def _ferramentas():
    """Define as 3 ferramentas (apenas o ESQUEMA; usado para o tool-calling)."""
    from haystack.tools import Tool

    # function e obrigatorio no Tool, mas nao sera invocado (so usamos a ESCOLHA do LLM)
    nada = lambda caminho="": ""
    return [
        Tool(name="extrair_planilha", description="Use para PLANILHAS: .xlsx, .xls, .csv, .tsv.",
             parameters=_param_caminho("caminho do arquivo"), function=nada),
        Tool(name="extrair_texto",
             description="Use para PDF nativo, DOCX ou TXT que JA possuem camada de texto (sem OCR).",
             parameters=_param_caminho("caminho do arquivo"), function=nada),
        Tool(name="extrair_com_ocr",
             description="Use para IMAGENS, PDFs ESCANEADOS (pouco texto + imagens) ou paginas com FIGURAS.",
             parameters=_param_caminho("caminho do arquivo"), function=nada),
    ]


def _llm_escolhe_tecnica(caminho, sinais):
    """Uma unica chamada ao Groq com tool_choice='required' -> retorna (tecnica, motivo)."""
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.utils import Secret

    api_key, modelo, base_url = config.config_llm()
    gerador = OpenAIChatGenerator(
        api_key=Secret.from_token(api_key), model=modelo, api_base_url=base_url,
        tools=_ferramentas(),
        # tool_choice='required' forca o tool-call ESTRUTURADO (corrige o tool_use_failed do Groq)
        generation_kwargs={"temperature": 0.0, "max_tokens": 300, "tool_choice": "required"})
    prompt = (f"Sinais do documento: {json.dumps(sinais, ensure_ascii=False)}\n"
              "Selecione a ferramenta de extracao adequada.")
    system = prompts.get_prompts()["extracao_system"]   # editavel na aba Configuracoes
    msgs = [ChatMessage.from_system(system), ChatMessage.from_user(prompt)]
    reply = gerador.run(messages=msgs)["replies"][0]
    chamadas = reply.tool_calls or []
    if not chamadas:
        return None, ""
    nome = chamadas[0].tool_name
    tecnica = MAPA_TOOL_TECNICA.get(nome)
    return tecnica, f"o LLM escolheu '{nome}' a partir dos sinais do documento"


# ---------------------------------------------------------------------------
# 3) Orquestracao: LLM decide -> extrai (com fallback heuristico)
# ---------------------------------------------------------------------------
def decidir_e_extrair(caminho):
    """Decide a tecnica (LLM) e extrai. Devolve (sinais, tecnica, complexidade, motivo, dados).

    Se o LLM falhar (ex.: instabilidade de tool-calling do Groq) ou nao escolher nada,
    cai num FALLBACK HEURISTICO deterministico pelos sinais do probe.
    """
    log.info("Iniciando extracao: %s", caminho)
    sinais = probe(caminho)
    log.debug("Sinais do probe: %s", json.dumps(sinais, ensure_ascii=False))

    tecnica, motivo = None, ""
    try:
        log.info("Consultando o LLM (Groq, tool-calling) para escolher a tecnica...")
        tecnica, motivo = _llm_escolhe_tecnica(caminho, sinais)
        log.info("LLM escolheu a tecnica: %s", tecnica or "(nenhuma ferramenta)")
    except Exception as e:
        log.warning("LLM falhou (%s) -> usando fallback heuristico", e)
        motivo = f"(fallback heuristico: LLM falhou - {e})"

    if not tecnica:
        tecnica = escolher_por_sinais(sinais)
        log.info("FALLBACK heuristico: tecnica '%s' escolhida pelos sinais", tecnica)
        if not motivo:
            motivo = f"(fallback heuristico: tecnica '{tecnica}' escolhida pelos sinais)"

    dados = _extrair_direto(caminho, tecnica)
    complexidade = MAPA_COMPLEXIDADE.get(tecnica, "texto_simples")
    log.info("Extracao finalizada: tecnica=%s, complexidade=%s", tecnica, complexidade)
    return sinais, tecnica, complexidade, motivo, dados


def limpar_cache(caminho=None):
    if caminho:
        _CACHE.pop(caminho, None)
    else:
        _CACHE.clear()
