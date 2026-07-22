"""
interface.py - Interface Gradio para a API do Projeto Final (Aula 12).

Facilita o uso da API sem curl/Postman. Tem duas abas:
  1) Ingestao  - envia um documento e mostra o RELATORIO da decisao
                 (tecnica de extracao do agente + destino + tecnica de chunking + motivos).
  2) Consulta  - faz uma pergunta e mostra a resposta RAG + fontes.

A interface NAO faz logica de RAG: ela so chama a API (separacao clara
"motor" x "interface"). Configure a URL da API em API_URL (.env), padrao http://localhost:8000.

Rodar (com a API ja no ar):
    python interface.py        # abre em http://localhost:7860
"""

import os

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8001")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
API_KEY = os.getenv("API_KEY", "")  # so se a API exigir X-API-Key

CHUNKING_OPCOES = ["auto", "fixo", "recursivo", "sentenca_janela", "semantico", "hierarquico"]
DESTINO_OPCOES = ["auto", "opensearch", "grafo"]
TECNICA_OPCOES = ["baseline", "multi_query", "rag_fusion", "step_back"]  # query enhancement (OpenSearch)
RERANK_OPCOES = ["rrf", "minmax", "modelo"]                              # rerank/fusao (OpenSearch)
ORIGEM_OPCOES = ["opensearch", "grafo"]


def _headers():
    return {"X-API-Key": API_KEY} if API_KEY else {}


# ---------------------------------------------------------------------------
# Aba 1 - Ingestao
# ---------------------------------------------------------------------------
def ingerir(arquivo, estrategia, chunking, limpar):
    # outputs: resumo, json, aba_grafo(update), grafo_md, grafo_html, grafo_json
    sem_grafo = (gr.update(), gr.update(), gr.update(), gr.update())  # nao mexe na aba/conteudo
    if not arquivo:
        return ("Selecione um arquivo.", {}, *sem_grafo)
    aviso_limpeza = ""
    if limpar:
        try:
            r = requests.post(f"{API_URL}/admin/limpar", headers=_headers(), timeout=120).json()
            aviso_limpeza = f"🧹 Limpeza: OpenSearch={r.get('opensearch')} | Grafo={r.get('grafo')}\n\n"
        except Exception as e:
            aviso_limpeza = f"⚠️ Falha ao limpar: {e}\n\n"
    try:
        with open(arquivo, "rb") as f:
            resp = requests.post(
                f"{API_URL}/ingestao",
                params={"estrategia": estrategia, "chunking": chunking},
                files={"arquivo": (os.path.basename(arquivo), f)},
                headers=_headers(), timeout=600)
        resp.raise_for_status()
        dados = resp.json()
    except Exception as e:
        return (f"Erro ao chamar a API: {e}", {}, *sem_grafo)

    if not dados.get("ok"):
        return (f"Falha na ingestao: {dados.get('erro')}", dados, *sem_grafo)

    r = dados["relatorio"]
    resumo = (
        aviso_limpeza +
        f"### Decisao da ingestao\n"
        f"- **Arquivo:** {r['arquivo']} ({r['n_caracteres']} caracteres)\n"
        f"- **Complexidade:** {r['complexidade']}\n"
        f"- **Extracao (agente):** `{r['tecnica_extracao']}` — {r.get('motivo_extracao','')}\n"
        f"- **Destino:** `{r['destino']}` — {r.get('motivo_destino','')}\n"
        f"- **Chunking:** `{r.get('chunking','')}` — {r.get('motivo_chunking','')}\n"
        f"- **Chunks indexados:** {r['n_chunks']}\n"
    )

    # se agora existe grafo (ex.: destino=grafo), MOSTRA a aba e ja carrega o conteudo
    existe, _ = grafo_info()
    if existe:
        gmd, ghtml, gjson = atualizar_grafo()
        return (resumo, dados, gr.update(visible=True), gmd, ghtml, gjson)
    return (resumo, dados, *sem_grafo)


# ---------------------------------------------------------------------------
# Aba 2 - Consulta
# ---------------------------------------------------------------------------
def _fmt_metricas(met):
    """Formata as metricas de uma pergunta (retrieval + RAGAS) em markdown."""
    if not met:
        return ""
    partes = ["#### 📊 Métricas desta pergunta"]
    if met.get("gabarito"):
        partes.append(f"_{met['gabarito']}_")
    r = met.get("retrieval")
    if r:
        partes.append("**Recuperação:** " + "  |  ".join(f"{k}={v}" for k, v in r.items()))
    else:
        partes.append("**Recuperação:** _sem gabarito casado — selecione um dataset e o gabarito "
                      "(ou deixe 'auto') para medir Hit@K/Recall@K/MRR/NDCG/AUC._")
    rg = met.get("ragas") or {}
    if rg.get("erro"):
        partes.append(f"**RAGAS:** _{rg['erro']}_")
    elif rg:
        partes.append("**RAGAS:** " + "  |  ".join(f"{k}={v}" for k, v in rg.items()))
    return "\n\n".join(partes)


def perguntas_do_dataset(nome):
    """['auto (semântico)'] + perguntas do dataset (para o dropdown de gabarito)."""
    if not nome or nome == "nenhum":
        return gr.update(choices=["auto (semântico)"], value="auto (semântico)")
    try:
        d = requests.get(f"{API_URL}/ragas/perguntas", params={"nome": nome},
                         headers=_headers(), timeout=30).json()
        return gr.update(choices=["auto (semântico)"] + d.get("perguntas", []),
                         value="auto (semântico)")
    except Exception:
        return gr.update(choices=["auto (semântico)"], value="auto (semântico)")


def consultar(pergunta, destino, top_k, tecnica, rerank, dataset_nome, medir, gabarito):
    if not pergunta.strip():
        return "Digite uma pergunta.", "", {}
    gab = "auto" if (gabarito or "").startswith("auto") else gabarito
    payload = {"pergunta": pergunta, "destino": destino, "top_k": int(top_k),
               "tecnica": tecnica, "rerank": rerank, "dataset_nome": dataset_nome,
               "com_metricas": bool(medir) and destino != "grafo", "gabarito_pergunta": gab}
    try:
        resp = requests.post(f"{API_URL}/consulta", json=payload, headers=_headers(), timeout=600)
        resp.raise_for_status()
        dados = resp.json()
    except Exception as e:
        return f"Erro ao chamar a API: {e}", "", {}

    fontes = "\n".join(f"- {f}" for f in dados.get("fontes", [])) or "(sem fontes)"
    texto = (f"### Resposta (destino: {dados.get('destino_usado','?')} | técnica: {tecnica} "
             f"| rerank: {rerank})\n\n{dados.get('resposta','')}\n\n**Fontes:**\n{fontes}")
    return texto, _fmt_metricas(dados.get("metricas")), dados


# ---------------------------------------------------------------------------
# RAGAS: datasets e geracao
# ---------------------------------------------------------------------------
def listar_datasets():
    """Retorna ['nenhum', <datasets...>] para os combos."""
    try:
        d = requests.get(f"{API_URL}/ragas/datasets", headers=_headers(), timeout=30).json()
        return ["nenhum"] + d.get("datasets", [])
    except Exception:
        return ["nenhum"]


def gerar_dataset_ragas(nome, origem, n):
    if not (nome or "").strip():
        return "Informe o nome do dataset.", gr.update()
    try:
        r = requests.post(f"{API_URL}/ragas/gerar_dataset",
                          json={"nome": nome, "origem": origem, "n": int(n)},
                          headers=_headers(), timeout=1800)
        r.raise_for_status()
        d = r.json()
        msg = (f"✅ Dataset **{d['nome']}** criado ({d['n_itens']} perguntas, origem "
               f"{d['origem']}).\nArquivo: `{d['arquivo']}`")
    except Exception as e:
        return f"❌ Erro ao gerar dataset: {e}", gr.update()
    # atualiza os combos de dataset (aba Consulta)
    return msg, gr.update(choices=listar_datasets())


# ---------------------------------------------------------------------------
# Avaliacao em lote (CSV + dataset -> metricas)
# ---------------------------------------------------------------------------
def _perguntas_do_csv(caminho):
    import pandas as pd

    df = pd.read_csv(caminho)
    col = next((c for c in df.columns if c.lower() in ("pergunta", "perguntas", "question", "query")),
               df.columns[0])
    return [str(x) for x in df[col].dropna().tolist() if str(x).strip()]


def avaliar_lote(arquivo_csv, dataset_nome, tecnica, rerank, top_k):
    import pandas as pd

    perguntas = []
    if arquivo_csv:
        try:
            perguntas = _perguntas_do_csv(arquivo_csv)
        except Exception as e:
            return f"❌ Erro lendo o CSV: {e}", None
    if not perguntas and (not dataset_nome or dataset_nome == "nenhum"):
        return "Envie um CSV de perguntas **ou** selecione um dataset RAGAS.", None
    try:
        r = requests.post(f"{API_URL}/avaliar_lote",
                          json={"perguntas": perguntas, "dataset_nome": dataset_nome,
                                "tecnica": tecnica, "rerank": rerank, "top_k": int(top_k)},
                          headers=_headers(), timeout=3600)
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        return f"❌ Erro na avaliacao: {e}", None
    if not d.get("ok"):
        return f"❌ {d.get('erro')}", None

    df = pd.DataFrame(d.get("linhas", []))
    med = d.get("medias_retrieval", {}) or {}
    ragas = d.get("ragas") or {}
    linhas_med = " | ".join(f"{k}={v}" for k, v in med.items()) or "(sem gabarito)"
    txt = (f"### Resultado (técnica: **{d['tecnica']}** | rerank: **{d['rerank']}** | "
           f"top_k={d['top_k']})\n"
           f"- Perguntas: {d['n_perguntas']} (com gabarito: {d['com_gabarito']})\n"
           f"- **Médias retrieval:** {linhas_med}\n")
    if ragas.get("medias"):
        txt += "- **Médias RAGAS:** " + " | ".join(f"{k}={v}" for k, v in ragas["medias"].items()) + "\n"
    elif ragas.get("erro"):
        txt += f"- RAGAS: _{ragas['erro']}_\n"
    return txt, df


def status():
    try:
        return requests.get(f"{API_URL}/health", headers=_headers(), timeout=30).json()
    except Exception as e:
        return {"erro": str(e), "dica": f"A API esta no ar em {API_URL}?"}


# ---------------------------------------------------------------------------
# Aba Grafo (LightRAG) - so aparece se existir grafo
# ---------------------------------------------------------------------------
def grafo_info():
    """Consulta /graph. Retorna (existe, dict). Usado no startup e no botao Atualizar."""
    try:
        d = requests.get(f"{API_URL}/graph", headers=_headers(), timeout=60).json()
        return bool(d.get("exists")), d
    except Exception as e:
        return False, {"erro": str(e)}


def _iframe_grafo():
    """HTML com um iframe que carrega a visualizacao interativa servida pela API."""
    return (f'<iframe src="{API_URL}/graph/html" '
            'style="width:100%;height:600px;border:1px solid #ddd;border-radius:8px;"></iframe>')


def atualizar_grafo():
    existe, d = grafo_info()
    if not existe:
        return "Nenhum grafo no LightRAG ainda. Ingira um documento com destino 'grafo'.", "", {}
    hubs = "\n".join(f"- {h['no']} (grau {h['grau']})" for h in d.get("top_hubs", [])) or "(sem hubs)"
    md = (f"### Grafo de Conhecimento (LightRAG)\n"
          f"- **Nos:** {d['n_nodes']}  |  **Arestas:** {d['n_edges']}  "
          f"(exibindo {d.get('exibindo_nos', 0)} mais conectados)\n\n"
          f"**Entidades mais conectadas:**\n{hubs}")
    return md, _iframe_grafo(), d


# checagem no startup: a aba so e criada se houver grafo
_GRAFO_EXISTE, _GRAFO_DADOS = grafo_info()


# ---------------------------------------------------------------------------
# Aba Configuracoes - prompts editaveis
# ---------------------------------------------------------------------------
def carregar_prompts():
    """Le os prompts atuais da API. Retorna (rag, variacoes, stepback, extracao_system)."""
    try:
        p = requests.get(f"{API_URL}/config/prompts", headers=_headers(), timeout=30).json()
    except Exception as e:
        return (f"(erro ao carregar: {e})", "", "", "")
    return (p.get("rag", ""), p.get("variacoes", ""), p.get("stepback", ""),
            p.get("extracao_system", ""))


def salvar_prompts(rag, variacoes, stepback, extracao_system):
    corpo = {"rag": rag, "variacoes": variacoes, "stepback": stepback,
             "extracao_system": extracao_system}
    try:
        r = requests.put(f"{API_URL}/config/prompts", json=corpo, headers=_headers(), timeout=30)
        r.raise_for_status()
        return "✅ Prompts salvos. As proximas buscas/ingestoes ja usam os novos textos."
    except Exception as e:
        return f"❌ Erro ao salvar: {e}"


def restaurar_prompts():
    try:
        requests.post(f"{API_URL}/config/prompts/reset", headers=_headers(), timeout=30).raise_for_status()
    except Exception as e:
        return (*carregar_prompts(), f"❌ Erro ao restaurar: {e}")
    return (*carregar_prompts(), "♻️ Prompts restaurados ao padrao.")


_PROMPTS_INI = carregar_prompts()   # valores iniciais (no startup)
_DATASETS_INI = listar_datasets()   # datasets RAGAS disponiveis (no startup)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Projeto Final - RAG Juridico (Aula 12)") as demo:
    gr.Markdown(
        "# Projeto Final — Ingestao Inteligente + RAG\n"
        "Interface da API: o **agente** decide a extracao e a **heuristica** decide o "
        "**destino** (OpenSearch/grafo) e a **tecnica de chunking**. "
        f"API: `{API_URL}`")

    with gr.Tab("1) Ingestao"):
        arquivo = gr.File(label="Documento (PDF, DOCX, XLSX, imagem, TXT)", type="filepath")
        with gr.Row():
            estrategia = gr.Dropdown(DESTINO_OPCOES, value="auto", label="Destino (override)")
            chunking = gr.Dropdown(CHUNKING_OPCOES, value="auto", label="Chunking (override)")
        limpar = gr.Checkbox(value=False,
                             label="🧹 Limpar índice OpenSearch e grafo LightRAG antes de indexar")
        btn_ing = gr.Button("Ingerir", variant="primary")
        out_ing_md = gr.Markdown()
        out_ing_json = gr.JSON(label="Relatorio completo (JSON)")
        # (o .click e registrado no fim, quando os componentes do grafo ja existem)

    with gr.Tab("2) Consulta"):
        pergunta = gr.Textbox(label="Pergunta", lines=2, placeholder="Ex.: Qual o prazo do recurso?")
        with gr.Row():
            destino = gr.Dropdown(DESTINO_OPCOES, value="auto", label="Buscar em")
            tecnica = gr.Dropdown(TECNICA_OPCOES, value="baseline", visible=False,
                                  label="Técnica (query enhancement, só OpenSearch)")
            rerank = gr.Dropdown(RERANK_OPCOES, value="rrf", visible=False,
                                 label="Rerank/fusão (só OpenSearch)")
            top_k = gr.Slider(1, 15, value=5, step=1, label="top_k")
        with gr.Row():
            dataset_combo = gr.Dropdown(_DATASETS_INI, value="nenhum",
                                        label="Dataset RAGAS (gabarito p/ métricas) — ou 'nenhum'")
            gabarito_combo = gr.Dropdown(["auto (semântico)"], value="auto (semântico)",
                                         label="Gabarito (necessidade de informação)")
            medir = gr.Checkbox(value=True, label="📊 Medir métricas (retrieval + RAGAS)")
        gr.Markdown("_Digite uma **paráfrase** e escolha o gabarito (ou deixe **auto**): a métrica "
                    "compara sua busca contra os documentos relevantes daquela necessidade — "
                    "sem precisar repetir a pergunta exata do dataset._")
        btn_q = gr.Button("Perguntar", variant="primary")
        out_q_md = gr.Markdown()
        out_q_metricas = gr.Markdown()
        out_q_json = gr.JSON(label="Resposta completa (JSON)")
        # Tecnica e Rerank so aparecem quando "Buscar em" = opensearch
        destino.change(lambda d: (gr.update(visible=(d == "opensearch")),
                                  gr.update(visible=(d == "opensearch"))),
                       inputs=destino, outputs=[tecnica, rerank])
        # ao trocar o dataset, popula o dropdown de gabarito com as perguntas dele
        dataset_combo.change(perguntas_do_dataset, inputs=dataset_combo, outputs=gabarito_combo)
        btn_q.click(consultar,
                    [pergunta, destino, top_k, tecnica, rerank, dataset_combo, medir, gabarito_combo],
                    [out_q_md, out_q_metricas, out_q_json])

        gr.Markdown("---\n#### Avaliação em lote (CSV de perguntas + métricas)")
        csv_lote = gr.File(label="CSV de perguntas (coluna 'pergunta')", type="filepath")
        btn_lote = gr.Button("Avaliar em lote", variant="primary")
        out_lote_md = gr.Markdown()
        out_lote_tab = gr.Dataframe(label="Métricas por pergunta", wrap=True)
        btn_lote.click(avaliar_lote, [csv_lote, dataset_combo, tecnica, rerank, top_k],
                       [out_lote_md, out_lote_tab])

    with gr.Tab("3) RAGAS (dataset)"):
        gr.Markdown(
            "Gera um **dataset de avaliação** lendo o índice OpenSearch (ou o grafo LightRAG): "
            "para cada documento, a LLM cria uma pergunta + resposta de referência e registra o "
            "documento relevante (gabarito). Use-o depois na **Avaliação em lote** (aba Consulta).")
        with gr.Row():
            ragas_nome = gr.Textbox(label="Nome do dataset", placeholder="ex.: meu_corpus_v1")
            ragas_origem = gr.Dropdown(ORIGEM_OPCOES, value="opensearch", label="Origem")
            ragas_n = gr.Slider(3, 100, value=15, step=1, label="Nº de documentos")
        btn_ragas = gr.Button("Gerar Dataset RAGAS", variant="primary")
        out_ragas_md = gr.Markdown()

    # Aba do grafo: SEMPRE criada, mas comeca OCULTA se ainda nao existe grafo.
    # Apos uma ingestao com destino 'grafo', ela fica visivel automaticamente (sem reiniciar).
    aba_grafo = gr.Tab("4) Grafo (LightRAG)", visible=_GRAFO_EXISTE)
    with aba_grafo:
        gr.Markdown("Visualizacao interativa do grafo de conhecimento construido pelo LightRAG.")
        btn_g = gr.Button("Atualizar grafo", variant="primary")
        out_g_md = gr.Markdown()
        out_g_html = gr.HTML(_iframe_grafo() if _GRAFO_EXISTE else "")
        out_g_json = gr.JSON(label="Estatisticas (JSON)")
        btn_g.click(atualizar_grafo, None, [out_g_md, out_g_html, out_g_json])

    with gr.Tab("⚙️ Configuracoes (prompts)"):
        gr.Markdown(
            "Edite os **prompts** usados pelo app. As mudancas valem para as proximas "
            "buscas/ingestoes e ficam salvas (prompts.json).\n\n"
            "**Mantenha os marcadores** `{{ documents }}` e `{{ pergunta }}` nos prompts de "
            "busca (sintaxe Jinja). Removê-los pode quebrar a geracao.")
        cfg_rag = gr.Textbox(label="Prompt do RAG (resposta final) — usa {{ documents }} e {{ pergunta }}",
                             value=_PROMPTS_INI[0], lines=8)
        cfg_var = gr.Textbox(label="Prompt de variacoes (multi_query / rag_fusion) — usa {{ pergunta }}",
                             value=_PROMPTS_INI[1], lines=4)
        cfg_sb = gr.Textbox(label="Prompt step-back — usa {{ pergunta }}",
                            value=_PROMPTS_INI[2], lines=4)
        cfg_ext = gr.Textbox(label="System prompt do agente de extracao (texto puro)",
                             value=_PROMPTS_INI[3], lines=6)
        with gr.Row():
            btn_salvar = gr.Button("Salvar prompts", variant="primary")
            btn_restaurar = gr.Button("Restaurar padrao")
            btn_recarregar = gr.Button("Recarregar")
        cfg_status = gr.Markdown()
        btn_salvar.click(salvar_prompts, [cfg_rag, cfg_var, cfg_sb, cfg_ext], cfg_status)
        btn_restaurar.click(restaurar_prompts, None,
                            [cfg_rag, cfg_var, cfg_sb, cfg_ext, cfg_status])
        btn_recarregar.click(carregar_prompts, None, [cfg_rag, cfg_var, cfg_sb, cfg_ext])

    with gr.Tab("Status"):
        btn_s = gr.Button("Checar /health")
        out_s = gr.JSON()
        btn_s.click(status, None, out_s)

    # botao RAGAS: gera o dataset e ATUALIZA o combo de datasets da aba Consulta
    btn_ragas.click(gerar_dataset_ragas, [ragas_nome, ragas_origem, ragas_n],
                    [out_ragas_md, dataset_combo])

    # registra o click da ingestao aqui: alem do relatorio, ele revela/atualiza a aba Grafo
    btn_ing.click(ingerir, [arquivo, estrategia, chunking, limpar],
                  [out_ing_md, out_ing_json, aba_grafo, out_g_md, out_g_html, out_g_json])

if __name__ == "__main__":
    demo.launch(server_port=GRADIO_PORT)
