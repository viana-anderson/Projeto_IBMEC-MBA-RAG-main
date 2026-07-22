"""
modelos.py - Schemas Pydantic da API (entrada/saida dos endpoints).
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class RelatorioIngestao(BaseModel):
    """O que a API devolve apos ingerir um documento (mostra a DECISAO tomada)."""
    arquivo: str
    complexidade: str = Field(..., description="planilha | texto_simples | complexo | imagem")
    tecnica_extracao: str = Field(..., description="ferramenta escolhida pelo agente")
    motivo_extracao: str = ""
    estrutura: dict = Field(default_factory=dict, description="sinais do documento")
    destino: str = Field(..., description="opensearch | grafo (LightRAG)")
    motivo_destino: str = ""
    chunking: str = Field("", description="tecnica de chunking escolhida (so no OpenSearch)")
    motivo_chunking: str = ""
    n_chunks: int = 0
    n_caracteres: int = 0


class ConsultaRequest(BaseModel):
    pergunta: str
    destino: str = Field("auto", description="auto | opensearch | grafo")
    top_k: int = 5
    tecnica: str = Field("baseline",
                         description="baseline | multi_query | rag_fusion | step_back (so no OpenSearch)")
    rerank: str = Field("rrf", description="rrf | minmax | modelo (so no OpenSearch)")
    dataset_nome: Optional[str] = Field(None, description="dataset RAGAS p/ gabarito das metricas (opcional)")
    com_metricas: bool = Field(False, description="calcular metricas (retrieval + RAGAS) desta pergunta")
    gabarito_pergunta: str = Field("auto", description="'auto' (semantico) ou a pergunta-gabarito escolhida")


class ConsultaResponse(BaseModel):
    pergunta: str
    resposta: str
    destino_usado: str
    fontes: list = Field(default_factory=list)
    metricas: Optional[dict] = None


class IngestaoResponse(BaseModel):
    ok: bool
    relatorio: Optional[RelatorioIngestao] = None
    erro: Optional[str] = None


class PromptsConfig(BaseModel):
    """Edicao dos prompts (todos opcionais; so os enviados sao atualizados)."""
    rag: Optional[str] = None
    variacoes: Optional[str] = None
    stepback: Optional[str] = None
    extracao_system: Optional[str] = None


class GerarDatasetRequest(BaseModel):
    nome: str = Field(..., description="nome do dataset RAGAS a criar")
    origem: str = Field("opensearch", description="opensearch | grafo")
    n: int = Field(15, description="quantos documentos usar para gerar perguntas")


class AvaliarLoteRequest(BaseModel):
    perguntas: List[str] = Field(default_factory=list, description="perguntas do CSV (opcional)")
    dataset_nome: Optional[str] = Field(None, description="dataset RAGAS p/ gabarito (ou nenhum)")
    tecnica: str = "baseline"
    rerank: str = "rrf"
    top_k: int = 5
