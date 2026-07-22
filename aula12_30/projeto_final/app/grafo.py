"""
grafo.py - Leitura e visualizacao do grafo de conhecimento do LightRAG.

O LightRAG grava o grafo em arquivo (working_dir = config.PASTA_RAG_STORAGE), no
arquivo GraphML 'graph_chunk_entity_relation.graphml'. Aqui:
  - existe()      : ha grafo gravado?
  - ler_grafo()   : devolve nos/arestas + estatisticas (para o endpoint /graph)
  - html_vis()    : pagina HTML interativa (vis-network) para o endpoint /graph/html

Leitura robusta do GraphML: o arquivo pode vir com bytes nulos / lixo apos
'</graphml>' (artefato do Ollama/LightRAG); por isso limpamos antes de parsear.
"""

import json
from pathlib import Path

from . import config

NOME_GRAPHML = "graph_chunk_entity_relation.graphml"


def caminho_graphml():
    return config.PASTA_RAG_STORAGE / NOME_GRAPHML


def existe():
    p = caminho_graphml()
    return p.exists() and p.stat().st_size > 0


def _ler_nx():
    """Le o GraphML de forma robusta e devolve um networkx.Graph."""
    import networkx as nx

    bruto = Path(caminho_graphml()).read_text(encoding="utf-8", errors="ignore")
    bruto = bruto.replace("\x00", "")
    fim = bruto.find("</graphml>")
    if fim != -1:
        bruto = bruto[:fim + len("</graphml>")]
    return nx.parse_graphml(bruto)


def ler_grafo(limite_nos=150):
    """Devolve {n_nodes, n_edges, top_hubs, nodes, edges}.

    Para grafos grandes, mostra apenas os 'limite_nos' nos de maior grau (e as
    arestas entre eles), para a visualizacao ficar legivel.
    """
    if not existe():
        return {"exists": False, "n_nodes": 0, "n_edges": 0,
                "top_hubs": [], "nodes": [], "edges": []}

    g = _ler_nx()
    n_nodes_total, n_edges_total = g.number_of_nodes(), g.number_of_edges()

    # ordena por grau e mantem os mais conectados
    graus = dict(g.degree())
    principais = sorted(graus, key=graus.get, reverse=True)[:limite_nos]
    conjunto = set(principais)

    nodes = [{"id": str(n), "label": str(n)[:40], "grau": int(graus[n])} for n in principais]
    edges = []
    for u, v, d in g.edges(data=True):
        if u in conjunto and v in conjunto:
            edges.append({"source": str(u), "target": str(v),
                          "label": str(d.get("keywords", ""))[:30]})

    top_hubs = [{"no": str(n), "grau": int(graus[n])} for n in principais[:10]]
    return {"exists": True, "n_nodes": n_nodes_total, "n_edges": n_edges_total,
            "exibindo_nos": len(nodes), "top_hubs": top_hubs, "nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Visualizacao interativa (vis-network via CDN) - pagina HTML auto-contida
# ---------------------------------------------------------------------------
_TEMPLATE_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"/>
<title>Grafo de Conhecimento (LightRAG)</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
  html, body { margin: 0; height: 100%; font-family: Arial, sans-serif; }
  #info { padding: 6px 10px; font-size: 13px; color: #333; background: #f4f4f4; }
  #rede { width: 100%; height: calc(100% - 32px); border: 0; }
</style>
</head>
<body>
<div id="info">__INFO__</div>
<div id="rede"></div>
<script>
  const nodes = new vis.DataSet(__NODES__);
  const edges = new vis.DataSet(__EDGES__);
  const container = document.getElementById('rede');
  const data = { nodes: nodes, edges: edges };
  const options = {
    nodes: { shape: 'dot', scaling: { min: 6, max: 40 },
             font: { size: 12 }, color: { background: '#7aa6e0', border: '#3f6fb0' } },
    edges: { color: { color: '#bbb' }, smooth: { type: 'continuous' },
             font: { size: 10, align: 'middle' }, arrows: { to: { enabled: false } } },
    physics: { stabilization: true, barnesHut: { gravitationalConstant: -8000,
               springLength: 120 } },
    interaction: { hover: true, tooltipDelay: 120 }
  };
  new vis.Network(container, data, options);
</script>
</body>
</html>"""


def html_vis(limite_nos=150):
    """Monta a pagina HTML interativa do grafo (vis-network)."""
    dados = ler_grafo(limite_nos=limite_nos)
    if not dados["exists"]:
        return "<html><body style='font-family:Arial'><p>Ainda nao existe grafo no LightRAG. " \
               "Faca uma ingestao cujo destino seja 'grafo'.</p></body></html>"

    vis_nodes = [{"id": n["id"], "label": n["label"], "value": n["grau"],
                  "title": f"{n['id']} (grau {n['grau']})"} for n in dados["nodes"]]
    vis_edges = [{"from": e["source"], "to": e["target"], "label": e["label"]}
                 for e in dados["edges"]]
    info = (f"Grafo LightRAG &mdash; {dados['n_nodes']} nos, {dados['n_edges']} arestas "
            f"(exibindo os {dados['exibindo_nos']} mais conectados).")
    return (_TEMPLATE_HTML
            .replace("__NODES__", json.dumps(vis_nodes, ensure_ascii=False))
            .replace("__EDGES__", json.dumps(vis_edges, ensure_ascii=False))
            .replace("__INFO__", info))
