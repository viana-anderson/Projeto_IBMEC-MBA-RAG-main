import json
import csv
from app import busca_avancada  

def carregar_dataset(caminho):
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)

def salvar_linha_csv(caminho_csv, experimento, metricas):
    # Salva os resultados linha por linha na planilha de experimentos
    pass

def rodar_avaliacao_recuperacao(tecnica_nome, top_k=5):
    dataset = carregar_dataset("avaliacao/dataset.json")
    
    hits, recalls, mrrs, ndcgs = [], [], [], []
    
    for q in dataset["queries_benchmark"]:
        pergunta = q["query"]
        gabarito_relevancia = q["relevancia"] # Ex: {"D01": 2, "D02": 1}
        
        # Ajustado para a chamada padrão do módulo de busca avançada do projeto
        pipe, inputs, chave = busca_avancada.construir(tecnica_nome, top_k, pergunta)
        resultado_exec = pipe.run(inputs)
        
        # Obtém os documentos da chave de saída retornada pelo pipeline do Haystack
        docs = resultado_exec.get(chave, {}).get("documents", [])
        
        # Extrai os IDs dos documentos recuperados
        ids_recuperados = [d.meta.get("id_original") or d.id for d in docs]
        
        # TODO: Calcule as métricas comparando `ids_recuperados` com `gabarito_relevancia`
        
    print(f"Avaliação da recuperação concluída para a técnica: {tecnica_nome}")

if __name__ == "__main__":
    rodar_avaliacao_recuperacao(tecnica_nome="baseline", top_k=5)