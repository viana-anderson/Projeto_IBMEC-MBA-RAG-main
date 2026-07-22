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


def _headers():
    return {"X-API-Key": API_KEY} if API_KEY else {}


# ---------------------------------------------------------------------------
# Aba 1 - Ingestao
# ---------------------------------------------------------------------------
def ingerir(arquivo, estrategia, chunking):
    # outputs: resumo, json, aba_grafo(update), grafo_md, grafo_html, grafo_json
    sem_grafo = (gr.update(), gr.update(), gr.update(), gr.update())  # nao mexe na aba/conteudo
    if not arquivo:
        return ("Selecione um arquivo.", {}, *sem_grafo)
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
def consultar(pergunta, destino, top_k, tecnica):
    if not pergunta.strip():
        return "Digite uma pergunta.", {}
    try:
        resp = requests.post(
            f"{API_URL}/consulta",
            json={"pergunta": pergunta, "destino": destino, "top_k": int(top_k),
                  "tecnica": tecnica},
            headers=_headers(), timeout=600)
        resp.raise_for_status()
        dados = resp.json()
    except Exception as e:
        return f"Erro ao chamar a API: {e}", {}

    fontes = "\n".join(f"- {f}" for f in dados.get("fontes", [])) or "(sem fontes)"
    texto = (f"### Resposta (destino: {dados.get('destino_usado','?')} | técnica: {tecnica})\n\n"
             f"{dados.get('resposta','')}\n\n**Fontes:**\n{fontes}")
    return texto, dados


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
            top_k = gr.Slider(1, 15, value=5, step=1, label="top_k")
        btn_q = gr.Button("Perguntar", variant="primary")
        out_q_md = gr.Markdown()
        out_q_json = gr.JSON(label="Resposta completa (JSON)")
        # a Tecnica so aparece quando "Buscar em" = opensearch (oculta em auto/grafo)
        destino.change(lambda d: gr.update(visible=(d == "opensearch")),
                       inputs=destino, outputs=tecnica)
        btn_q.click(consultar, [pergunta, destino, top_k, tecnica], [out_q_md, out_q_json])

    # Aba do grafo: SEMPRE criada, mas comeca OCULTA se ainda nao existe grafo.
    # Apos uma ingestao com destino 'grafo', ela fica visivel automaticamente (sem reiniciar).
    aba_grafo = gr.Tab("3) Grafo (LightRAG)", visible=_GRAFO_EXISTE)
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

    # registra o click da ingestao aqui: alem do relatorio, ele revela/atualiza a aba Grafo
    btn_ing.click(ingerir, [arquivo, estrategia, chunking],
                  [out_ing_md, out_ing_json, aba_grafo, out_g_md, out_g_html, out_g_json])

if __name__ == "__main__":
    demo.launch(server_port=GRADIO_PORT)
