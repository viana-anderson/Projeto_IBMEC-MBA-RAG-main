"""
prompts.py - Prompts do app, EDITAVEIS em runtime (aba Configuracoes do Gradio).

Mantem os textos dos prompts num so lugar, com valores padrao + override que o aluno
edita pela interface. O override e persistido em prompts.json (sobrevive a reinicios).

Chaves:
  - rag             : prompt FINAL da busca (usa {{ documents }} e {{ pergunta }}).
  - variacoes       : gera variacoes da pergunta (multi_query / rag_fusion) - usa {{ pergunta }}.
  - stepback        : gera a pergunta mais geral (step_back) - usa {{ pergunta }}.
  - extracao_system : system prompt do agente que escolhe a tecnica de extracao (texto puro).

IMPORTANTE: os marcadores {{ documents }} / {{ pergunta }} sao obrigatorios nos prompts de
busca (sintaxe Jinja do Haystack). Se o aluno remove-los, a busca pode falhar.
"""

import json

from . import config
from .log import obter_logger

log = obter_logger(__name__)

ARQUIVO = config.PASTA_PROJETO / "prompts.json"

DEFAULTS = {
    "rag": (
        "Voce e um assistente juridico. Responda APENAS com base nos trechos abaixo, de "
        "forma objetiva. Se nao constar, diga que nao consta.\n\n"
        "Trechos:\n{% for d in documents %}- {{ d.content }}\n{% endfor %}\n"
        "Pergunta: {{ pergunta }}\nResposta:"
    ),
    "variacoes": (
        "Gere 4 reformulacoes ALTERNATIVAS da pergunta abaixo, uma por linha, sem numerar "
        "e sem comentarios, preservando a intencao mas variando termos e foco.\n\n"
        "Pergunta: {{ pergunta }}"
    ),
    "stepback": (
        "Dada a pergunta especifica abaixo, escreva UMA pergunta mais ampla e geral "
        "(step-back) que ajude a recuperar contexto de fundo. Responda APENAS com a "
        "pergunta.\n\nPergunta: {{ pergunta }}"
    ),
    "extracao_system": (
        "Voce e um classificador de documentos. A partir dos sinais fornecidos, selecione a "
        "ferramenta de extracao mais adequada:\n"
        "- extrair_planilha: planilhas (.xlsx, .xls, .csv, .tsv).\n"
        "- extrair_com_ocr: imagens, PDFs escaneados (pouco texto e com imagens) ou paginas com figuras.\n"
        "- extrair_texto: PDFs nativos, DOCX ou TXT que ja possuem camada de texto.\n"
        "Selecione exatamente uma ferramenta."
    ),
}

_ATUAIS = dict(DEFAULTS)


def _carregar():
    global _ATUAIS
    try:
        if ARQUIVO.exists():
            dados = json.loads(ARQUIVO.read_text(encoding="utf-8"))
            _ATUAIS = {k: dados.get(k) or DEFAULTS[k] for k in DEFAULTS}
            log.info("Prompts carregados de %s", ARQUIVO)
    except Exception as e:
        log.warning("Falha ao carregar prompts (%s) -> usando defaults", e)


_carregar()


def get_prompts():
    """Prompts atuais (override do aluno ou default)."""
    return dict(_ATUAIS)


def set_prompts(novos):
    """Atualiza so as chaves enviadas (texto nao-vazio) e persiste em prompts.json."""
    for k in DEFAULTS:
        v = (novos or {}).get(k)
        if isinstance(v, str) and v.strip():
            _ATUAIS[k] = v
    try:
        ARQUIVO.write_text(json.dumps(_ATUAIS, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Prompts salvos em %s", ARQUIVO)
    except Exception as e:
        log.warning("Falha ao salvar prompts: %s", e)
    return dict(_ATUAIS)


def reset():
    """Restaura os prompts padrao (apaga o override)."""
    global _ATUAIS
    _ATUAIS = dict(DEFAULTS)
    try:
        if ARQUIVO.exists():
            ARQUIVO.unlink()
    except Exception as e:
        log.warning("Falha ao remover %s: %s", ARQUIVO, e)
    return dict(_ATUAIS)
