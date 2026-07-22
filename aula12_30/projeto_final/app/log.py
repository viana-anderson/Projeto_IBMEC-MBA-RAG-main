"""
log.py - Configuracao central de logging do Projeto Final (Aula 12).

Verbosidade controlada por LOG_LEVEL no .env (DEBUG | INFO | WARNING | ERROR).
  - INFO  (padrao): mostra as DECISOES (extracao, destino, chunking, consulta).
  - DEBUG          : mostra tambem detalhes (sinais do probe, trechos, libs ruidosas).

Uso nos modulos:
    from .log import obter_logger
    log = obter_logger(__name__)
    log.info("mensagem")
"""

import logging
import os

# bibliotecas barulhentas: so aparecem se LOG_LEVEL=DEBUG
_LIBS_RUIDOSAS = ["httpx", "httpcore", "openai", "urllib3", "docling", "haystack",
                  "opensearch", "lightrag", "nano-vectordb", "asyncio"]

_CONFIGURADO = False


def configurar_logging(nivel=None):
    """Configura o logging uma unica vez. Le LOG_LEVEL do .env (default INFO)."""
    global _CONFIGURADO
    if _CONFIGURADO:
        return
    nivel = (nivel or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, nivel, logging.INFO),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # silencia libs barulhentas, exceto em DEBUG
    nivel_libs = logging.DEBUG if nivel == "DEBUG" else logging.WARNING
    for nome in _LIBS_RUIDOSAS:
        logging.getLogger(nome).setLevel(nivel_libs)
    _CONFIGURADO = True
    logging.getLogger("projeto_final").info("Logging configurado em nivel %s", nivel)


def obter_logger(nome):
    """Logger filho de 'projeto_final' (um so LOG_LEVEL controla tudo)."""
    configurar_logging()
    curto = nome.split(".")[-1]  # ex.: app.extracao -> extracao
    return logging.getLogger(f"projeto_final.{curto}")
