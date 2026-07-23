import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_groq import ChatGroq  # Essencial para configurar a Groq como LLM juiz no RAGAS
from app import consulta  # Importa o módulo de consulta/RAG completo do projeto

def rodar_ragas():
    # 1. Carregar o dataset com perguntas e respostas de referência
    with open("avaliacao/dataset.json", 'r', encoding='utf-8') as f:
        dados = json.load(f)
    
    perguntas = []
    respostas_geradas = []
    contextos_recuperados = []
    respostas_referencia = []
    
    for q in dados["queries_benchmark"]:
        pergunta = q["query"]
        ref = q["resposta_referencia"]
        
        # Chamada real ao pipeline de RAG do projeto (conforme app/consulta.py)
        resultado_rag = consulta.executar(pergunta) # Ajuste o método conforme a assinatura real em consulta.py
        
        resposta_do_modelo = resultado_rag.get("resposta", "")
        # Extrai o texto dos documentos retornados pelo pipeline
        contextos = [d.content for d in resultado_rag.get("documentos", [])]
        
        perguntas.append(pergunta)
        respostas_geradas.append(resposta_do_modelo)
        contextos_recuperados.append(contextos)
        respostas_referencia.append(ref)
    
    # 2. Montar o formato de dataset exigido pela biblioteca RAGAS
    data = {
        "question": perguntas,
        "answer": respostas_geradas,
        "contexts": contextos_recuperados,
        "ground_truth": respostas_referencia
    }
    dataset_ragas = Dataset.from_dict(data)
    
    # 3. Configurar a métrica de relevância com strictness=1 (exigência da Groq)
    ans_rel = answer_relevancy
    ans_rel.strictness = 1
    
    # 4. Configurar a Groq como LLM e embeddings para o RAGAS
    # O RAGAS utiliza LangChain por baixo dos panos para os juízes
    llm_juiz = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0) # Utilize o modelo adequado configurado no projeto
    
    # 5. Executar a avaliação passando o avaliador configurado
    resultado = evaluate(
        dataset=dataset_ragas,
        metrics=[
            faithfulness,
            ans_rel,
            context_precision,
            context_recall
        ],
        llm=llm_juiz
    )
    
    print("Resultados RAGAS:", resultado)
    df_resultado = resultado.to_pandas()
    os.makedirs("avaliacao", exist_ok=True)
    df_resultado.to_csv("avaliacao/resultado_ragas.csv", index=False)

if __name__ == "__main__":
    rodar_ragas()