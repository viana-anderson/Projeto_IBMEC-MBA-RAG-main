# reusa a logica de bench_embeddings/app/metricas.py

import csv
from pathlib import Path
from app import busca_avancada
from app.metricas import metricas_por_query
#from app.avaliacao_ragas import carregar_dataset # ou chama a API POST /consulta

import json

def carregar_dataset(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)

TOP_K = 5

TECNICAS = [
    "baseline",
    "multi_query",
    "rag_fusion",
    "step_back",
]

def salvar_linha_csv(arquivo, linha):
    arquivo = Path(arquivo)
    existe = arquivo.exists()
    with open(arquivo, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=linha.keys()
        )

        if not existe:
            writer.writeheader()
        writer.writerow(linha)

gold = carregar_dataset("avaliacao/dataset.json")
print(gold)

for tecnica in TECNICAS:
    print(f"\nExecutando {tecnica}")
    for pergunta in gold["queries_benchmark"]:
        pipe, inputs, chave = busca_avancada.construir(tecnica=tecnica, top_k=TOP_K, pergunta=pergunta["query"],)
        resposta = pipe.run(inputs, include_outputs_from={chave},)
        docs = resposta[chave]["documents"]
        ids = [d.meta.get("id_original") or d.id for d in docs] 

# for q in gold["queries_benchmark"]:
#     pipe, inputs, chave = busca_avancada.construir(TECNICA, TOP_K, q["query"])
#     docs = pipe.run(inputs, include_outputs_from={chave})[chave]["documents"]
#     ids = [d.meta.get("id_original") or d.id for d in docs]

        scores = [d.score for d in docs]

        metricas = metricas_por_query(
            ids_ranqueados=ids,
            scores=scores,
            relevancia=pergunta["relevancia"],
            k=TOP_K,
        )

linha = {
    "exp": 1,
    "fase": "Baseline",
    "mudança": "Busca padrão",
    "Hit@5": medias["hit@5"],
    "Recall@5": medias["recall@5"],
    "MRR": medias["mrr"],
    "NDCG@10": medias["ndcg@10"],
    "RAGAS_faith": faith,
    "RAGAS_ans_rel": ans_rel,
    "RAGAS_ctx_recall": ctx,
    "latência(s)": tempo_medio,
    "custo": custo,
    "observação": "",
}

salvar_linha_csv("avaliacao/resultados.csv",linha,)

# -> hit@k, recall@k, mrr, ndcg@k usando q["relevancia"]
# salvar_linha_csv("avaliacao/resultados.csv", experimento, metricas)