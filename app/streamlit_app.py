# app/streamlit_app.py
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network


def chips(items, max_items=12):
    items = [str(x) for x in (items or []) if x]
    if not items:
        return
    items = items[:max_items]
    html = ""
    for it in items:
        html += (
            "<span style='display:inline-block;"
            "padding:4px 10px;margin:3px;border-radius:999px;"
            "background:#f2f4f7;font-size:12px'>"
            + it +
            "</span>"
        )
    st.markdown(html, unsafe_allow_html=True)


def show_list(title, items, max_items=12):
    items = [x for x in (items or []) if x not in (None, "")]
    if not items:
        return
    st.markdown(f"**{title}**")
    chips(items, max_items=max_items)


# -----------------------------
# Config
# -----------------------------
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")  # ✅ 절대 하드코딩 X (.env 권장)

ONTO_REL_TYPES = [
    "USES_METHOD",
    "USES_SOFTWARE",
    "HAS_DATA",
    "HAS_RESEARCH_TASK",
    "STUDIES",
]

# ✅ Assumption 제거 (UI/쿼리/라벨리스트 모두에서 제거)
ONTO_NODE_LABELS = ["Method", "DataType", "Software", "Context", "ResearchTask", "Taxon"]

REL_TYPES_JOIN = "|".join(ONTO_REL_TYPES)

REL_STYLE = {
    "USES_METHOD": {"width": 2},
    "USES_SOFTWARE": {"width": 2},
    "HAS_DATA": {"width": 2},
    "HAS_RESEARCH_TASK": {"width": 3},
    "STUDIES": {"width": 3},
    "CITES": {"width": 1, "dashes": True},
}

NODE_STYLE = {
    "Paper": {"shape": "dot", "size": 28},
    "Method": {"shape": "dot", "size": 18},
    "DataType": {"shape": "dot", "size": 18},
    "Software": {"shape": "dot", "size": 18},
    "Context": {"shape": "dot", "size": 18},
    "ResearchTask": {"shape": "dot", "size": 20},
    "Taxon": {"shape": "dot", "size": 20},
    # "Assumption": {"shape": "dot", "size": 18},  # ✅ 제거
}

# -----------------------------
# Neo4j driver
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def run_cypher(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(query, params or {})
        return [r.data() for r in result]


# -----------------------------
# 3) Coverage / Metrics (Neo4j 5+ 안전)
# -----------------------------
def get_coverage_stats() -> Dict[str, Any]:
    q_total = "MATCH (p:Paper) RETURN count(p) AS total"
    total_rows = run_cypher(q_total)
    total = total_rows[0]["total"] if total_rows else 0

    q_tagged = f"""
    MATCH (p:Paper)
    WHERE COUNT {{ (p)-[:{REL_TYPES_JOIN}]->() }} > 0
    RETURN count(p) AS tagged
    """
    tagged_rows = run_cypher(q_tagged)
    tagged = tagged_rows[0]["tagged"] if tagged_rows else 0

    rel_rows = []
    for rel in ONTO_REL_TYPES:
        q = f"""
        MATCH (p:Paper)-[:{rel}]->()
        RETURN '{rel}' AS rel, count(DISTINCT p) AS papers
        """
        out = run_cypher(q)
        rel_rows.append(out[0] if out else {"rel": rel, "papers": 0})

    rel_df = pd.DataFrame(rel_rows).sort_values("papers", ascending=False)

    return {
        "total": total,
        "tagged": tagged,
        "coverage": (tagged / total * 100.0) if total else 0.0,
        "rel_df": rel_df,
    }


# -----------------------------
# 1) Search with evidence (안정 버전)
# -----------------------------
def search_papers_with_evidence(keyword: str, k: int) -> pd.DataFrame:
    kw = keyword.strip()
    if not kw:
        return pd.DataFrame(columns=["openalexId", "title", "year", "reason", "score"])

    q_title = """
    MATCH (p:Paper)
    WHERE toLower(coalesce(p.title,"")) CONTAINS toLower($kw)
    RETURN p.openalexId AS openalexId, p.title AS title, p.year AS year,
           "title match" AS reason, 1.0 AS score
    LIMIT $k
    """

    q_tag = f"""
    MATCH (t)
    WHERE any(lbl IN labels(t) WHERE lbl IN $onto_labels)
      AND (
        toLower(coalesce(t.name,""))  CONTAINS toLower($kw) OR
        toLower(coalesce(t.label,"")) CONTAINS toLower($kw)
      )
    WITH t, labels(t)[0] AS tLabel, coalesce(t.name, t.label, "unknown") AS tName
    MATCH (p:Paper)-[r:{REL_TYPES_JOIN}]->(t)
    RETURN p.openalexId AS openalexId, p.title AS title, p.year AS year,
           ("tag match: " + type(r) + " -> " + tLabel + " / " + tName) AS reason,
           2.0 AS score
    LIMIT $k
    """

    rows = []
    rows += run_cypher(q_title, {"kw": kw, "k": k})
    rows += run_cypher(q_tag, {"kw": kw, "k": k, "onto_labels": ONTO_NODE_LABELS})

    if not rows:
        return pd.DataFrame(columns=["openalexId", "title", "year", "reason", "score"])

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["openalexId", "title", "year"], dropna=False)
        .agg(
            score=("score", "sum"),
            reason=("reason", lambda x: " | ".join(list(dict.fromkeys(x)))),
        )
        .reset_index()
        .sort_values(["score", "year"], ascending=[False, False])
        .head(k)
    )
    return agg


# -----------------------------
# Paper detail (tags)  ✅ Assumption 완전 제거
# -----------------------------
def get_paper_detail(openalex_id: str) -> Dict[str, Any]:
    q = """
    MATCH (p:Paper {openalexId:$oid})
    OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
    OPTIONAL MATCH (p)-[:USES_SOFTWARE]->(s:Software)
    OPTIONAL MATCH (p)-[:HAS_CONTEXT]->(c:Context)
    OPTIONAL MATCH (p)-[:HAS_RESEARCH_TASK]->(rt:ResearchTask)
    OPTIONAL MATCH (p)-[:STUDIES]->(t:Taxon)
    OPTIONAL MATCH (p)-[:HAS_DATA]->(d:DataType)

    RETURN
      p.openalexId AS openalexId,
      p.title AS title,
      p.year AS year,
      collect(DISTINCT coalesce(m.name, m.label)) AS methods,
      collect(DISTINCT coalesce(t.name, t.label)) AS taxa,
      collect(DISTINCT coalesce(s.name, s.label)) AS software,
      collect(DISTINCT coalesce(c.name, c.label)) AS contexts,
      collect(DISTINCT coalesce(rt.name, rt.label)) AS researchTasks,
      collect(DISTINCT coalesce(d.name, d.label)) AS dataTypes
    """
    rows = run_cypher(q, {"oid": openalex_id})
    if not rows:
        return {
            "openalexId": openalex_id,
            "title": None,
            "year": None,
            "methods": [],
            "taxa": [],
            "software": [],
            "contexts": [],
            "researchTasks": [],
            "dataTypes": [],
        }

    detail = rows[0]
    for key in ["methods", "taxa", "software", "contexts", "researchTasks", "dataTypes"]:
        detail[key] = [x for x in detail.get(key, []) if x not in (None, "")]
    return detail


# -----------------------------
# Similar papers + evidence
# -----------------------------
def get_similar_papers(openalex_id: str, k: int) -> pd.DataFrame:
    q = f"""
    MATCH (p:Paper {{openalexId:$oid}})
    MATCH (p)-[r:{REL_TYPES_JOIN}]->(t)
    WITH p, collect(DISTINCT t) AS tags

    MATCH (p2:Paper)
    WHERE p2 <> p

    OPTIONAL MATCH (p2)-[r2:{REL_TYPES_JOIN}]->(t2)
    WHERE t2 IN tags

    WITH p2,
         collect(DISTINCT {{ rel:type(r2), label:labels(t2)[0], name:coalesce(t2.name,t2.label) }}) AS shared,
         count(DISTINCT t2) AS score
    WHERE score > 0
    RETURN p2.openalexId AS openalexId, p2.title AS title, p2.year AS year,
           score AS score,
           shared AS evidence
    ORDER BY score DESC, year DESC
    LIMIT $k
    """
    rows = run_cypher(q, {"oid": openalex_id, "k": k})
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["openalexId", "title", "year", "score", "evidence"])
    )


# -----------------------------
# 2) Graph 1-hop (✅ elementId로 통일)
# -----------------------------
def get_graph_1hop(openalex_id: str, max_edges: int) -> Tuple[List[Dict], List[Dict]]:
    q = f"""
    MATCH (p:Paper {{openalexId:$oid}})
    OPTIONAL MATCH (p)-[r:{REL_TYPES_JOIN}]->(t)
    RETURN
      elementId(p) AS p_id,
      labels(p) AS p_labels,
      p.title AS p_title,
      p.openalexId AS p_oid,
      collect(DISTINCT {{
        id: elementId(t),
        labels: labels(t),
        display: coalesce(t.name, t.label, head(labels(t)))
      }}) AS nodes_info,
      collect({{
        from: elementId(p),
        to: elementId(t),
        type: type(r)
      }}) AS rels_info
    """
    rows = run_cypher(q, {"oid": openalex_id})
    if not rows or not rows[0].get("p_id"):
        return [], []

    data = rows[0]
    node_list = [
        {
            "id": data["p_id"],
            "labels": data["p_labels"],
            "display": data.get("p_title") or data.get("p_oid") or "Paper",
            "title": data.get("p_title") or "",
            "openalexId": data.get("p_oid"),
        }
    ]
    seen = {data["p_id"]}

    for n in data.get("nodes_info", []):
        if not n.get("id"):
            continue
        if n["id"] in seen:
            continue
        node_list.append(
            {
                "id": n["id"],
                "labels": n.get("labels", ["Node"]),
                "display": n.get("display", "Node"),
            }
        )
        seen.add(n["id"])

    edges_raw = data.get("rels_info", [])
    edge_list = [
        {"from": e["from"], "to": e["to"], "type": e.get("type", "REL")}
        for e in edges_raw
        if e.get("to")
    ][:max_edges]

    return node_list, edge_list


def render_pyvis_graph(nodes: List[Dict], edges: List[Dict], height_px: int = 520) -> None:
    if not nodes or not edges:
        st.info("그래프에 표시할 연결이 없습니다.")
        return

    net = Network(height=f"{height_px}px", width="100%", directed=True)

    for n in nodes:
        labels = n.get("labels", [])
        primary = labels[0] if labels else "Node"
        style = NODE_STYLE.get(primary, {"shape": "dot", "size": 16})

        net.add_node(
            n["id"],
            label=str(n.get("display", ""))[:80],
            title=(n.get("title") or n.get("display") or ""),
            shape=style.get("shape", "dot"),
            size=style.get("size", 16),
        )

    for e in edges:
        rtype = e.get("type", "REL")
        stl = REL_STYLE.get(rtype, {"width": 1})
        net.add_edge(
            e["from"],
            e["to"],
            label=rtype,
            title=rtype,
            width=stl.get("width", 1),
            arrows="to",
            dashes=stl.get("dashes", False),
        )

    net.set_options(
        """
    var options = {
      "edges": {"smooth": {"type": "dynamic"}, "font": {"size": 10}},
      "nodes": {"font": {"size": 12}},
      "physics": {"barnesHut": {"gravitationalConstant": -25000, "springLength": 120}}
    }
    """
    )
    components.html(net.generate_html(), height=height_px + 40, scrolling=False)


# -----------------------------
# Graph-based expansion (태그 → 관련 논문)
# -----------------------------
def get_papers_by_tag(tag_label: str, tag_name: str, k: int) -> pd.DataFrame:
    if tag_label not in ONTO_NODE_LABELS:
        return pd.DataFrame(columns=["openalexId", "title", "year", "reason", "score"])

    q = f"""
    MATCH (t:{tag_label})
    WHERE toLower(coalesce(t.name,t.label,"")) = toLower($tname)
    MATCH (p:Paper)-[r:{REL_TYPES_JOIN}]->(t)
    RETURN p.openalexId AS openalexId,
           p.title AS title,
           p.year AS year,
           ("via " + type(r) + " -> {tag_label} / " + coalesce(t.name,t.label,"")) AS reason,
           1.0 AS score
    ORDER BY year DESC
    LIMIT $k
    """
    rows = run_cypher(q, {"tname": tag_name, "k": k})
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["openalexId", "title", "year", "reason", "score"])
    )


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="SEMANTICA", layout="wide")
st.title("SEMANTICA (Neo4j)")

left, mid, right = st.columns([1.2, 2.2, 2.6], gap="large")

with left:
    st.subheader("검색")
    keyword = st.text_input("키워드 (Paper title / Ontology node)", value="genomic")
    k = st.slider("결과 수", 5, 50, 20, step=5)

    st.divider()

    cov = get_coverage_stats()
    st.subheader("데이터 상태")
    st.metric("전체 Paper", cov["total"])
    st.metric("Tagged", cov["tagged"])
    st.metric("Coverage", f"{cov['coverage']:.1f}%")

    with st.expander("관계 타입별 커버리지"):
        st.dataframe(cov["rel_df"], use_container_width=True, hide_index=True)

st.divider()

with mid:
    st.subheader("검색 결과 (근거 포함)")
    results_df = search_papers_with_evidence(keyword, k)

    if results_df.empty:
        st.warning("검색 결과가 없습니다.")
        st.stop()

    show = results_df.copy()
    show["title_short"] = show["title"].fillna("").str.slice(0, 90) + "…"
    show["year_str"] = show["year"].fillna("N/A").astype(str)

    options = [f"{row.year_str} · {row.title_short}" for _, row in show.iterrows()]
    idx = st.radio(
        "결과 선택",
        options=list(range(len(options))),
        format_func=lambda i: options[i],
        label_visibility="collapsed",
    )
    chosen_id = results_df.iloc[idx]["openalexId"]

    with st.expander("표로 보기", expanded=False):
        st.dataframe(
            results_df[["openalexId", "title", "year", "reason"]],
            use_container_width=True,
            hide_index=True,
        )

    chosen_row = results_df.iloc[idx]
    with st.expander("선택 논문: 검색 근거", expanded=True):
        st.markdown(f"**{chosen_row['title']}** ({chosen_row['year']})")
        reasons = str(chosen_row["reason"]).split(" | ")
        chips(reasons, max_items=8)

with right:
    tab1, tab2, tab3 = st.tabs(["📄 상세", "🕸 그래프", "🔎 유사 논문"])

    with tab1:
        st.subheader("논문 상세 / 태그")
        detail = get_paper_detail(chosen_id)

        st.markdown(f"### {detail.get('title','(no title)')} ({detail.get('year','N/A')})")
        st.caption(f"OpenAlex: {detail.get('openalexId')}")

        show_list("Methods", detail["methods"])
        show_list("Taxa", detail["taxa"])
        show_list("Software", detail["software"])
        show_list("DataTypes", detail["dataTypes"])
        show_list("Contexts", detail["contexts"])
        show_list("Research Tasks", detail["researchTasks"])

        if (
            not detail["methods"]
            and not detail["taxa"]
            and not detail["software"]
            and not detail["contexts"]
            and not detail["researchTasks"]
            and not detail["dataTypes"]
        ):
            st.info("이 논문은 아직 온톨로지 태그가 할당되지 않았습니다.")

    with tab2:
        st.subheader("그래프 탐색 (1-hop)")
        max_edges = st.slider("그래프 엣지 최대", 10, 200, 80, step=10, key="max_edges")
        nodes, edges = get_graph_1hop(chosen_id, max_edges)
        render_pyvis_graph(nodes, edges, height_px=540)

    with tab3:
        st.subheader("유사 논문 추천")
        sim_k = st.slider("추천 개수", 5, 30, 10, step=5, key="sim_k")
        sim_df = get_similar_papers(chosen_id, sim_k)

        if sim_df.empty:
            st.info("유사 논문이 없습니다.")
        else:
            st.dataframe(
                sim_df[["openalexId", "title", "year", "score"]],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("추천 근거(공유 태그) 보기"):
                for _, row in sim_df.head(10).iterrows():
                    st.markdown(f"**{row['title']}** ({row['year']}) — score={row['score']}")
                    for e in (row["evidence"] or [])[:12]:
                        st.write(f"- {e.get('rel')} → {e.get('label')} / {e.get('name')}")
                    st.markdown("---")
