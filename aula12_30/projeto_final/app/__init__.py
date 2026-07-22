"""Projeto Final (Aula 12) - API de Ingestao Inteligente + RAG.

IMPORTANTE: preparamos o tracing do Langfuse AQUI, no __init__ do pacote, porque ele
roda ANTES de qualquer import do Haystack (a auto-instrumentacao do Haystack le a
variavel HAYSTACK_CONTENT_TRACING_ENABLED no momento do import). So liga se as chaves
do Langfuse estiverem no .env.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# carrega o .env do projeto (mesmo arquivo que o config.py usa)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    # liga o tracing de conteudo do Haystack (precisa estar setado antes de importar haystack)
    os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
    # o Langfuse usa LANGFUSE_HOST; aceitamos LANGFUSE_BASE_URL como alias (padrao das aulas)
    base = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
    if base:
        os.environ.setdefault("LANGFUSE_HOST", base)
