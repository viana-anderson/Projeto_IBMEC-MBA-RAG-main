# Bancada de Avaliação de Embeddings (Gradio + API)

Ferramenta para descobrir **qual modelo de embedding é o melhor para um corpus/domínio**,
comparando modelos do **Ollama** por três famílias de métricas:

- **Retrieval:** Hit@K, Recall@K
- **Ranking:** MRR, NDCG@K (usa a relevância graduada do gabarito)
- **Separabilidade:** AUC-ROC (os scores separam relevante de não-relevante?)

A saída é uma **tabela**, **gráficos** (barras por métrica + radar) e uma **explicação
automática** do melhor modelo e por quê.

## Como funciona

```
Para cada modelo (Ollama):
  embeda o corpus e as queries  →  similaridade cosseno (em memória, numpy)
  → calcula Hit@K, Recall@K, MRR, NDCG@K, AUC contra o gabarito
  → registra dimensão e latência
Ranqueia os modelos (por NDCG@K) e explica o vencedor.
```

A **API (FastAPI)** faz o cálculo; a **interface (Gradio)** só consome a API e mostra os
resultados — separando "motor de avaliação" de "visualização".

## Estrutura

```
bench_embeddings/
├── app/
│   ├── api.py          # FastAPI: POST /avaliar, GET /modelos, /health
│   ├── avaliacao.py    # orquestra embed -> cosseno -> métricas -> melhor + explicação
│   ├── metricas.py     # Hit@K, Recall@K, MRR, NDCG@K, AUC-ROC (numpy + sklearn)
│   ├── embeddings.py   # API de embeddings do Ollama (/api/embed)
│   └── dados.py        # carrega o corpus + queries com gabarito
├── interface.py        # Gradio (chama a API)
├── requirements.txt
├── .env.example
└── README.md
```

## Dataset (corpus + gabarito)

Por padrão usa `aula11/datasets/corpus_juridico_benchmark.json` (30 documentos + 20
queries com relevância graduada). Você pode enviar o seu, no mesmo formato:

```json
{
  "documentos": [{"id": "D01", "texto": "..."}, ...],
  "queries_benchmark": [
    {"id": "Q01", "query": "...", "relevancia": {"D05": 1, "D06": 2}}
  ]
}
```

`relevancia` pode ser `{id: nota}` (graduada, melhor para NDCG) ou uma lista de ids
(`["D05","D06"]`, tratada como nota 1). O gabarito é **obrigatório** — sem ele não há como
medir.

## Como rodar

```bash
# 1) modelos no Ollama
ollama pull nomic-embed-text
ollama pull bge-m3

# 2) dependências
pip install -r requirements.txt

# 3) suba a API (terminal 1, de dentro de bench_embeddings/)
uvicorn app.api:app --port 8000

# 4) suba a interface (terminal 2)
python interface.py        # abre em http://localhost:7860
```

Na interface: escolha os modelos, o K, o dataset (padrão ou upload) e clique **Avaliar**.

## Como ler os resultados

- **NDCG@K** é a métrica-síntese de **qualidade de ranking** (coloca os relevantes no
  topo, ponderando a nota) — usamos como critério principal do "melhor".
- **Recall@K** mede **cobertura** (quantos relevantes entraram no top-K).
- **MRR** olha só a **posição do 1º relevante**.
- **AUC-ROC** mede **separabilidade** (quão bem os scores distinguem relevante de não).
- **dim** e **latência** são o custo: um modelo que empata na qualidade mas é menor/mais
  rápido pode ser o melhor para produção.

## Observações

- Tudo **em memória** (numpy) — adequado para benchmark, pois cada modelo gera vetores
  diferentes (reembeda de qualquer forma). Sem OpenSearch.
- A 1ª execução de cada modelo baixa os pesos no Ollama.
- Em corpora grandes, a matriz de similaridade Q×D cresce; para milhares de docs, troque a
  comparação por kNN/ANN.
