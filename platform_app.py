import re
import sqlite3
import os
import time
import html as html_lib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import feedparser
import pandas as pd
import requests
import streamlit as st
from dateutil import parser as date_parser
from zoneinfo import ZoneInfo


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "war_watch.db"
BEIJING_TZ = ZoneInfo("Asia/Shanghai")

OPENAI_MODEL = os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
          background: radial-gradient(1200px 800px at 10% 0%, rgba(60, 90, 255, 0.12), transparent 55%),
                      radial-gradient(1200px 800px at 90% 0%, rgba(255, 120, 80, 0.10), transparent 55%),
                      linear-gradient(180deg, rgba(255,255,255,0.92), rgba(248,249,253,0.92));
        }
        [data-testid="stHeader"] {
          background: transparent;
        }
        .block-container {
          padding-top: 1.2rem;
          padding-bottom: 2.2rem;
          max-width: 1400px;
        }
        h1, h2, h3 {
          letter-spacing: -0.02em;
        }
        .stTabs [data-baseweb="tab"] {
          border-radius: 12px;
          padding: 10px 14px;
        }
        .stTabs [aria-selected="true"] {
          background: rgba(20, 40, 120, 0.08);
        }
        [data-testid="stMetric"] {
          background: rgba(255,255,255,0.70);
          border: 1px solid rgba(0,0,0,0.06);
          padding: 10px 12px;
          border-radius: 12px;
        }
        .ww-day {
          font-weight: 700;
          margin: 18px 0 8px 0;
          color: rgba(10, 20, 45, 0.92);
        }
        .ww-card {
          background: rgba(255,255,255,0.80);
          border: 1px solid rgba(0,0,0,0.06);
          border-radius: 14px;
          padding: 14px 14px 12px 14px;
          margin: 10px 0;
          box-shadow: 0 6px 18px rgba(18, 24, 40, 0.06);
        }
        .ww-title {
          font-size: 0.98rem;
          font-weight: 700;
          margin: 2px 0 6px 0;
          line-height: 1.25rem;
        }
        .ww-meta {
          font-size: 0.82rem;
          color: rgba(20, 30, 60, 0.68);
          margin-bottom: 10px;
        }
        .ww-chips {
          display: flex;
          gap: 6px;
          flex-wrap: wrap;
          margin: 6px 0 10px 0;
        }
        .ww-chip {
          font-size: 0.75rem;
          padding: 3px 8px;
          border-radius: 999px;
          background: rgba(20, 40, 120, 0.08);
          border: 1px solid rgba(20, 40, 120, 0.10);
          color: rgba(20, 30, 60, 0.85);
        }
        .ww-chip-warn {
          background: rgba(255, 120, 80, 0.10);
          border: 1px solid rgba(255, 120, 80, 0.18);
          color: rgba(120, 40, 20, 0.92);
        }
        .ww-body {
          font-size: 0.90rem;
          color: rgba(10, 20, 45, 0.88);
          line-height: 1.35rem;
          white-space: pre-wrap;
        }
        .ww-link a {
          text-decoration: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def openai_translate_to_zh(text: str) -> str:
    api_key = (st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        return ""

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional translator. Translate to Simplified Chinese. Keep domain terms and abbreviations, and append original term in parentheses the first time it appears. Do not add new facts.",
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
    }

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()


def openai_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    api_key = (st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        return ""
    payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": temperature}
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()


def ai_extract_terms(original_text: str) -> Dict[str, object]:
    content = openai_chat(
        [
            {
                "role": "system",
                "content": "Extract 6-12 key terms/phrases from the text. Return JSON with keys: terms_original (array of strings), terms_zh (array of strings). Terms must be short phrases, not full sentences.",
            },
            {"role": "user", "content": original_text},
        ],
        temperature=0.1,
    )
    if not content:
        return {"terms_original": [], "terms_zh": []}
    try:
        import json

        obj = json.loads(content)
        return {"terms_original": obj.get("terms_original", []) or [], "terms_zh": obj.get("terms_zh", []) or []}
    except Exception:
        return {"terms_original": [], "terms_zh": []}


def ai_translate_and_expand(title_original: str, summary_original: str) -> Dict[str, str]:
    content = openai_chat(
        [
            {
                "role": "system",
                "content": "Return STRICT JSON only. Based only on the given title/summary (may be short), produce Chinese outputs without adding unverifiable facts. Keys: title_zh, summary_zh (4-10 bullets, detailed but grounded), terms_original (6-16 short phrases), terms_zh (6-16 short phrases). Do NOT include generic impact analysis.",
            },
            {"role": "user", "content": f"Title: {title_original}\nSummary: {summary_original}"},
        ],
        temperature=0.2,
    )
    if not content:
        return {}
    try:
        import json

        obj = json.loads(content)
        title_zh = str(obj.get("title_zh", "") or "").strip()
        summary_zh = obj.get("summary_zh", "")
        if isinstance(summary_zh, list):
            summary_zh = "\n".join([f"- {str(x).strip()}" for x in summary_zh if str(x).strip()])
        summary_zh = str(summary_zh or "").strip()
        terms_o = obj.get("terms_original", [])
        if not isinstance(terms_o, list):
            terms_o = []
        terms_zh = obj.get("terms_zh", [])
        if not isinstance(terms_zh, list):
            terms_zh = []
        return {
            "title_zh_ai": title_zh,
            "summary_zh_ai": summary_zh,
            "impact_ai": "",
            "terms_original": ", ".join([str(x).strip() for x in terms_o if str(x).strip()]),
            "terms_zh": "、".join([str(x).strip() for x in terms_zh if str(x).strip()]),
        }
    except Exception:
        return {}


def ai_generate_milestones(conflict: str, year: int, count: int) -> List[Dict[str, object]]:
    prompt = (
        "你将为‘战争里程碑时间轴’生成重要事件列表。要求：\n"
        "1) 只收录该年份的关键节点（停火/谈判/重大攻防与战役进程/跨境外溢/重要制裁与援助框架/重大国际机制与法律进程/领导层变化等）。\n"
        "2) 以中国大陆官方书面中文表述，避免口语，避免夸张。\n"
        "3) 每条必须包含：occurred_utc(ISO8601含+00:00)、date_precision(day/month)、title_zh、summary_zh(8-14条要点，以'- '开头；说明当时发生了什么、直接造成什么影响/后果)、category、phase、tags(用'|'分隔)、importance(50-100整数)。\n"
        "4) 不要编造具体伤亡数字、精确装备数量等无法从常识确认的信息；如日期只能确定到月份，可用该月第一天并将date_precision设为month。\n"
        "5) 仅返回严格JSON数组，不要任何额外文字。"
    )
    content = openai_chat(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"冲突：{conflict}\n年份：{year}\n生成数量：{count}"},
        ],
        temperature=0.2,
    )
    if not content:
        return []
    try:
        import json

        arr = json.loads(content)
        if not isinstance(arr, list):
            return []
        out: List[Dict[str, object]] = []
        for it in arr:
            if not isinstance(it, dict):
                continue
            if not str(it.get("occurred_utc", "")).strip():
                continue
            if not str(it.get("title_zh", "")).strip():
                continue
            if not str(it.get("summary_zh", "")).strip():
                continue
            it["conflict"] = conflict
            it["date_precision"] = str(it.get("date_precision", "day") or "day")
            it["importance"] = int(it.get("importance", 70) or 70)
            out.append(it)
        return out[:count]
    except Exception:
        return []


def ai_expand_milestone(title_zh: str, summary_zh: str) -> str:
    content = openai_chat(
        [
            {
                "role": "system",
                "content": "以中国大陆官方书面中文，将里程碑事件阐释扩写为10-16条要点（每条以'- '开头）。要求：写清‘发生了什么’、‘为何重要’、‘当时直接带来哪些影响/后果（战场/谈判/制裁援助/人道/外溢）’，避免编造精确伤亡或数量。仅基于给定标题与现有要点扩写。",
            },
            {"role": "user", "content": f"标题：{normalize_text(title_zh)}\n现有要点：{normalize_text(summary_zh)}"},
        ],
        temperature=0.2,
    )
    if not content:
        return ""
    lines = []
    for ln in str(content).splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("-"):
            lines.append(s if s.startswith("- ") else ("- " + s.lstrip("- ")))
        else:
            lines.append("- " + s)
    return "\n".join(lines[:16]).strip()


def expand_milestones(conn: sqlite3.Connection, ids: List[int]) -> int:
    if not ids:
        return 0
    if not _has_openai_key():
        return 0
    cur = conn.cursor()
    updated = 0
    prog = st.progress(0)
    for i, mid in enumerate(ids, start=1):
        row = cur.execute("SELECT title_zh, summary_zh FROM milestones WHERE id = ?", (int(mid),)).fetchone()
        if not row:
            prog.progress(i / len(ids))
            continue
        title_zh, summary_zh = row
        new_sum = ai_expand_milestone(str(title_zh or ""), str(summary_zh or ""))
        if new_sum:
            now = datetime.now(timezone.utc).isoformat()
            cur.execute("UPDATE milestones SET summary_zh = ?, updated_utc = ? WHERE id = ?", (new_sum, now, int(mid)))
            updated += 1
            conn.commit()
        prog.progress(i / len(ids))
    return updated


STANCE_DOC_SEEDS = [
    {
        "topic": "俄乌战争",
        "country": "中国",
        "issuer": "中华人民共和国外交部",
        "doc_type": "position_paper",
        "title_original": "关于政治解决乌克兰危机的中国立场",
        "language": "zh",
        "url": "https://www.fmprc.gov.cn/zyxw/202302/t20230224_11030707.shtml",
    },
    {
        "topic": "俄乌战争",
        "country": "乌克兰",
        "issuer": "Ministry of Foreign Affairs of Ukraine",
        "doc_type": "timeline",
        "title_original": "Timeline (Aggression of Russia against Ukraine)",
        "language": "uk",
        "url": "https://mfa.gov.ua/timeline?&type=posts",
    },
    {
        "topic": "俄乌战争",
        "country": "美国",
        "issuer": "U.S. Department of State",
        "doc_type": "policy_hub",
        "title_original": "Press Statements",
        "language": "en",
        "url": "https://www.state.gov/press-releases/",
    },
    {
        "topic": "俄乌战争",
        "country": "欧盟",
        "issuer": "European Council / Council of the EU",
        "doc_type": "policy_hub",
        "title_original": "Press releases",
        "language": "en",
        "url": "https://www.consilium.europa.eu/en/press/press-releases/",
    },
    {
        "topic": "俄乌战争",
        "country": "英国",
        "issuer": "UK Government",
        "doc_type": "policy_hub",
        "title_original": "Russia's invasion of Ukraine",
        "language": "en",
        "url": "https://www.gov.uk/government/topical-events/russias-invasion-of-ukraine",
    },
    {
        "topic": "俄乌战争",
        "country": "法国",
        "issuer": "Ministry for Europe and Foreign Affairs (France)",
        "doc_type": "country_file",
        "title_original": "Ukraine",
        "language": "en",
        "url": "https://www.diplomatie.gouv.fr/en/country-files/ukraine/",
    },
    {
        "topic": "俄乌战争",
        "country": "德国",
        "issuer": "Federal Foreign Office (Germany)",
        "doc_type": "country_file",
        "title_original": "Ukraine",
        "language": "en",
        "url": "https://www.auswaertiges-amt.de/en/aussenpolitik/laenderinformationen/ukraine-node",
    },
    {
        "topic": "俄乌战争",
        "country": "俄罗斯",
        "issuer": "Ministry of Foreign Affairs of Russia",
        "doc_type": "press_service",
        "title_original": "Press service",
        "language": "en",
        "url": "https://mid.ru/en/press_service/",
    },
    {
        "topic": "新一轮巴以冲突",
        "country": "中国",
        "issuer": "中华人民共和国外交部",
        "doc_type": "statement",
        "title_original": "习近平在金砖国家领导人巴以问题特别视频峰会上的讲话（全文）",
        "language": "zh",
        "url": "https://www.fmprc.gov.cn/zyxw/202311/t20231121_11184740.shtml",
    },
    {
        "topic": "新一轮巴以冲突",
        "country": "中国",
        "issuer": "中华人民共和国外交部",
        "doc_type": "joint_statement",
        "title_original": "中国和阿拉伯国家关于巴勒斯坦问题的联合声明（全文）",
        "language": "zh",
        "url": "https://www.mfa.gov.cn/ziliao_674904/1179_674909/202405/t20240531_11366712.shtml",
    },
    {
        "topic": "新一轮巴以冲突",
        "country": "美国",
        "issuer": "U.S. Department of State",
        "doc_type": "policy_hub",
        "title_original": "Press Releases",
        "language": "en",
        "url": "https://www.state.gov/press-releases/",
    },
    {
        "topic": "新一轮巴以冲突",
        "country": "以色列",
        "issuer": "Israel Ministry of Foreign Affairs",
        "doc_type": "policy_hub",
        "title_original": "Ministry of Foreign Affairs",
        "language": "en",
        "url": "https://www.gov.il/en/departments/ministry_of_foreign_affairs",
    },
    {
        "topic": "新一轮巴以冲突",
        "country": "卡塔尔",
        "issuer": "Ministry of Foreign Affairs (Qatar)",
        "doc_type": "policy_hub",
        "title_original": "News",
        "language": "en",
        "url": "https://www.mofa.gov.qa/en/",
    },
    {
        "topic": "新一轮巴以冲突",
        "country": "沙特",
        "issuer": "Ministry of Foreign Affairs (Saudi Arabia)",
        "doc_type": "policy_hub",
        "title_original": "News",
        "language": "en",
        "url": "https://www.mofa.gov.sa/en/Pages/default.aspx",
    },
    {
        "topic": "新一轮巴以冲突",
        "country": "伊朗",
        "issuer": "Ministry of Foreign Affairs of Iran",
        "doc_type": "policy_hub",
        "title_original": "News",
        "language": "en",
        "url": "https://en.mfa.ir/",
    },
]


def seed_stance_docs(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for d in STANCE_DOC_SEEDS:
        cur.execute(
            """
            INSERT OR IGNORE INTO stance_docs(
                topic, country, issuer, doc_type, title_original, language, url, published_utc, published_bj, summary_original, translation_zh
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, '', '')
            """,
            (d["topic"], d["country"], d["issuer"], d.get("doc_type", "statement"), d["title_original"], d["language"], d["url"]),
        )
    conn.commit()


def read_stance_docs(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT * FROM stance_docs
        ORDER BY country, topic, doc_type, id DESC
        """,
        conn,
    )


def translate_doc_row(conn: sqlite3.Connection, doc_id: int) -> None:
    row = conn.execute(
        "SELECT title_original, summary_original, language, translation_zh FROM stance_docs WHERE id = ?",
        (doc_id,),
    ).fetchone()
    if not row:
        return
    title_original, summary_original, language, translation_zh = row
    if translation_zh:
        return

    if (language or "").lower().startswith("zh"):
        conn.execute("UPDATE stance_docs SET translation_zh = title_original WHERE id = ?", (doc_id,))
        conn.commit()
        return

    translated = openai_translate_to_zh(f"Title: {title_original}\nSummary: {summary_original}")
    if translated:
        conn.execute("UPDATE stance_docs SET translation_zh = ? WHERE id = ?", (translated, doc_id))
        conn.commit()


def fetch_url_text(url: str) -> str:
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        txt = strip_html(r.text)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt[:6000]
    except Exception:
        return ""


def ai_summarize_stance_doc(title_original: str, body_text: str) -> str:
    content = openai_chat(
        [
            {
                "role": "system",
                "content": "以中国大陆官方书面中文，生成该文件的‘要点摘要’。仅基于提供的标题与正文摘录，不要编造。输出6-10条要点，每条以'- '开头。",
            },
            {"role": "user", "content": f"标题：{normalize_text(title_original)}\n正文摘录：{normalize_text(body_text)}"},
        ],
        temperature=0.2,
    )
    if not content:
        return ""
    lines = []
    for ln in str(content).splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("-"):
            lines.append(s if s.startswith("- ") else ("- " + s.lstrip("- ")))
        else:
            lines.append("- " + s)
    return "\n".join(lines[:12]).strip()


def summarize_doc_row(conn: sqlite3.Connection, doc_id: int) -> None:
    row = conn.execute(
        "SELECT title_original, url, summary_original, translation_zh FROM stance_docs WHERE id = ?",
        (doc_id,),
    ).fetchone()
    if not row:
        return
    title_original, url, summary_original, translation_zh = row
    title_original = str(title_original or "")
    body = str(summary_original or "").strip()
    if not body:
        body = fetch_url_text(str(url or ""))
    if not body:
        body = str(translation_zh or "")
    summary_zh = ai_summarize_stance_doc(title_original, body)
    if summary_zh:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("UPDATE stance_docs SET summary_zh = ?, updated_utc = ? WHERE id = ?", (summary_zh, now, doc_id))
        conn.commit()


@st.cache_data(ttl=3600)
def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        from io import StringIO

        df = pd.read_csv(StringIO(r.text))
    except Exception:
        return pd.DataFrame(columns=["date", "close"])
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame(columns=["date", "close"])
    df = df.rename(columns={"Date": "date", "Close": "close"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    return df[["date", "close"]]


def fetch_price_series(symbols: List[str], label: str) -> pd.DataFrame:
    for sym in symbols:
        df = fetch_stooq_daily(sym)
        if not df.empty:
            df = df.copy()
            df["series"] = label
            df["symbol"] = sym
            return df
    out = pd.DataFrame(columns=["date", "close", "series", "symbol"])
    return out


@st.cache_data(ttl=3600)
def fetch_fred_series(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        from io import StringIO

        df = pd.read_csv(StringIO(r.text))
    except Exception:
        return pd.DataFrame(columns=["date", "close"])
    if df.empty or "DATE" not in df.columns:
        return pd.DataFrame(columns=["date", "close"])
    value_col = [c for c in df.columns if c != "DATE"][0] if len(df.columns) > 1 else None
    if not value_col:
        return pd.DataFrame(columns=["date", "close"])
    df = df.rename(columns={"DATE": "date", value_col: "close"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    return df[["date", "close"]]


def fetch_price_auto(label: str) -> pd.DataFrame:
    if label == "原油期货":
        df = fetch_price_series(["brn.f", "cl.f"], "原油")
        if not df.empty:
            df["series"] = "原油（Stooq）"
            return df
        f = fetch_fred_series("DCOILBRENTEU")
        if not f.empty:
            f = f.copy()
            f["series"] = "原油（Brent, FRED）"
            f["symbol"] = "fred:DCOILBRENTEU"
            return f
        f2 = fetch_fred_series("DCOILWTICO")
        if not f2.empty:
            f2 = f2.copy()
            f2["series"] = "原油（WTI, FRED）"
            f2["symbol"] = "fred:DCOILWTICO"
            return f2
        return pd.DataFrame(columns=["date", "close", "series", "symbol"])
    if label == "小麦期货":
        df = fetch_price_series(["zw.f"], "小麦")
        if not df.empty:
            df["series"] = "小麦（Stooq）"
            return df
        f = fetch_fred_series("PWHEAMTUSDM")
        if not f.empty:
            f = f.copy()
            f["series"] = "小麦（FRED）"
            f["symbol"] = "fred:PWHEAMTUSDM"
            return f
        return pd.DataFrame(columns=["date", "close", "series", "symbol"])
    return pd.DataFrame(columns=["date", "close", "series", "symbol"])


def heuristic_terms(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    terms = []
    for m in re.findall(r"\b[A-Z][A-Z0-9\-]{1,10}\b", t):
        if m not in terms:
            terms.append(m)
    for m in re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\b", t):
        if m not in terms:
            terms.append(m)
    return terms[:10]


def read_event_ai(conn: sqlite3.Connection, event_ids: List[int]) -> Dict[int, dict]:
    if not event_ids:
        return {}
    qmarks = ",".join(["?"] * len(event_ids))
    rows = conn.execute(
        f"SELECT event_id, terms_original, terms_zh, impact_ai, title_zh_ai, summary_zh_ai FROM event_ai WHERE event_id IN ({qmarks})",
        tuple(event_ids),
    ).fetchall()
    out: Dict[int, dict] = {}
    for r in rows:
        out[int(r[0])] = {
            "terms_original": (r[1] or "").strip(),
            "terms_zh": (r[2] or "").strip(),
            "impact_ai": (r[3] or "").strip(),
            "title_zh_ai": (r[4] or "").strip(),
            "summary_zh_ai": (r[5] or "").strip(),
        }
    return out


def upsert_event_ai(conn: sqlite3.Connection, event_id: int, terms_original: str, terms_zh: str, impact_ai: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    _execute_with_retry(
        conn,
        """
        INSERT INTO event_ai(event_id, terms_original, terms_zh, impact_ai, updated_utc)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            terms_original=excluded.terms_original,
            terms_zh=excluded.terms_zh,
            impact_ai=excluded.impact_ai,
            updated_utc=excluded.updated_utc
        """,
        (int(event_id), terms_original, terms_zh, impact_ai, now),
    )
    conn.commit()


def upsert_event_ai_full(conn: sqlite3.Connection, event_id: int, payload: Dict[str, str]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    _execute_with_retry(
        conn,
        """
        INSERT INTO event_ai(event_id, terms_original, terms_zh, impact_ai, title_zh_ai, summary_zh_ai, updated_utc)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            terms_original=excluded.terms_original,
            terms_zh=excluded.terms_zh,
            impact_ai=excluded.impact_ai,
            title_zh_ai=excluded.title_zh_ai,
            summary_zh_ai=excluded.summary_zh_ai,
            updated_utc=excluded.updated_utc
        """,
        (
            int(event_id),
            payload.get("terms_original", ""),
            payload.get("terms_zh", ""),
            payload.get("impact_ai", ""),
            payload.get("title_zh_ai", ""),
            payload.get("summary_zh_ai", ""),
            now,
        ),
    )
    conn.commit()


@dataclass
class SourceSpec:
    name: str
    country: str
    channel_type: str
    language: str
    timezone: str
    rss_urls: List[str]
    political_spectrum: str = "N/A"  # New field: Left, Right, Center, State-run, etc.
    independence: str = "N/A"        # New field: Independent, State-funded, Corporate, etc.


SOURCES: List[SourceSpec] = [
    # --- 中国 ---
    SourceSpec(
        name="新华社",
        country="中国",
        channel_type="official_media",
        language="zh",
        timezone="Asia/Shanghai",
        rss_urls=["https://www.xinhuanet.com/english/rss/worldrss.xml"],
        political_spectrum="State-run",
        independence="State-owned",
    ),
    SourceSpec(
        name="Global Times",
        country="中国",
        channel_type="official_media",
        language="en",
        timezone="Asia/Shanghai",
        rss_urls=["https://www.globaltimes.cn/rss/world.xml"],
        political_spectrum="State-run",
        independence="State-owned",
    ),
    SourceSpec(
        name="CGTN",
        country="中国",
        channel_type="official_media",
        language="en",
        timezone="Asia/Shanghai",
        rss_urls=["https://www.cgtn.com/rss/world.xml"],
        political_spectrum="State-run",
        independence="State-owned",
    ),

    # --- 英国 ---
    SourceSpec(
        name="BBC",
        country="英国",
        channel_type="major_media",
        language="en",
        timezone="Europe/London",
        rss_urls=["http://feeds.bbci.co.uk/news/world/rss.xml"],
        political_spectrum="Center",
        independence="Publicly Funded (Independent Editorial)",
    ),
    SourceSpec(
        name="The Guardian",
        country="英国",
        channel_type="major_media",
        language="en",
        timezone="Europe/London",
        rss_urls=["https://www.theguardian.com/world/rss"],
        political_spectrum="Left-leaning",
        independence="Independent (Trust-owned)",
    ),
    SourceSpec(
        name="Sky News",
        country="英国",
        channel_type="major_media",
        language="en",
        timezone="Europe/London",
        rss_urls=["https://feeds.skynews.com/feeds/rss/world.xml"],
        political_spectrum="Center-Right",
        independence="Corporate (Comcast)",
    ),
    SourceSpec(
        name="英国政府",
        country="英国",
        channel_type="government",
        language="en",
        timezone="Europe/London",
        rss_urls=["https://www.gov.uk/government/announcements.atom"],
        political_spectrum="Government",
        independence="Government",
    ),

    # --- 美国 ---
    SourceSpec(
        name="CNN",
        country="美国",
        channel_type="major_media",
        language="en",
        timezone="America/New_York",
        rss_urls=["http://rss.cnn.com/rss/cnn_world.rss"],
        political_spectrum="Left-leaning/Liberal",
        independence="Corporate (Warner Bros)",
    ),
    SourceSpec(
        name="Fox News",
        country="美国",
        channel_type="major_media",
        language="en",
        timezone="America/New_York",
        rss_urls=["http://feeds.foxnews.com/foxnews/world"],
        political_spectrum="Right-leaning/Conservative",
        independence="Corporate (Fox Corp)",
    ),
    SourceSpec(
        name="AP",
        country="美国",
        channel_type="major_media",
        language="en",
        timezone="America/New_York",
        rss_urls=["https://apnews.com/hub/world-news?output=1"],
        political_spectrum="Center",
        independence="Independent (Non-profit cooperative)",
    ),
    SourceSpec(
        name="VOA (Voice of America)",
        country="美国",
        channel_type="government_funded",
        language="en",
        timezone="America/New_York",
        rss_urls=["https://www.voanews.com/api/zg$omevviq"],
        political_spectrum="Government-funded",
        independence="State-funded (Editorial Firewall)",
    ),
    SourceSpec(
        name="美国国防部",
        country="美国",
        channel_type="government",
        language="en",
        timezone="America/New_York",
        rss_urls=["https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?max=20&portalid=1&moduleid=1143"],
        political_spectrum="Government",
        independence="Government",
    ),

    # --- 法国 ---
    SourceSpec(
        name="France 24",
        country="法国",
        channel_type="major_media",
        language="en",
        timezone="Europe/Paris",
        rss_urls=["https://www.france24.com/en/rss"],
        political_spectrum="Center",
        independence="State-owned (France Médias Monde)",
    ),
    SourceSpec(
        name="RFI",
        country="法国",
        channel_type="major_media",
        language="en",
        timezone="Europe/Paris",
        rss_urls=["https://www.rfi.fr/en/rss"],
        political_spectrum="Center",
        independence="State-owned",
    ),
    SourceSpec(
        name="Le Monde",
        country="法国",
        channel_type="major_media",
        language="en",
        timezone="Europe/Paris",
        rss_urls=["https://www.lemonde.fr/en/rss/tag/international/index.xml"],
        political_spectrum="Center-Left",
        independence="Independent",
    ),

    # --- 德国 ---
    SourceSpec(
        name="DW",
        country="德国",
        channel_type="major_media",
        language="en",
        timezone="Europe/Berlin",
        rss_urls=["https://rss.dw.com/rdf/rss-en-world"],
        political_spectrum="Center",
        independence="State-funded (Public Law)",
    ),
    SourceSpec(
        name="Spiegel International",
        country="德国",
        channel_type="major_media",
        language="en",
        timezone="Europe/Berlin",
        rss_urls=["https://www.spiegel.de/international/index.rss"],
        political_spectrum="Center-Left",
        independence="Independent",
    ),

    # --- 俄罗斯 ---
    SourceSpec(
        name="TASS",
        country="俄罗斯",
        channel_type="official_media",
        language="en",
        timezone="Europe/Moscow",
        rss_urls=["http://tass.com/rss/v2.xml"],
        political_spectrum="State-run",
        independence="State-owned",
    ),
    SourceSpec(
        name="RT",
        country="俄罗斯",
        channel_type="state_media",
        language="en",
        timezone="Europe/Moscow",
        rss_urls=["https://www.rt.com/rss/news/"],
        political_spectrum="State-run",
        independence="State-owned",
    ),
    SourceSpec(
        name="俄罗斯外交部",
        country="俄罗斯",
        channel_type="government",
        language="en",
        timezone="Europe/Moscow",
        rss_urls=["https://mid.ru/en/rss/"],
        political_spectrum="Government",
        independence="Government",
    ),

    # --- 乌克兰 ---
    SourceSpec(
        name="Ukrinform",
        country="乌克兰",
        channel_type="official_media",
        language="en",
        timezone="Europe/Kyiv",
        rss_urls=["https://www.ukrinform.net/rss/block/lastnews"],
        political_spectrum="State-run",
        independence="State-owned",
    ),
    SourceSpec(
        name="Kyiv Independent",
        country="乌克兰",
        channel_type="independent_media",
        language="en",
        timezone="Europe/Kyiv",
        rss_urls=["https://kyivindependent.com/rss/"],
        political_spectrum="Pro-Ukraine/Western-aligned",
        independence="Independent (Crowdfunded)",
    ),
    SourceSpec(
        name="乌克兰国防部",
        country="乌克兰",
        channel_type="government",
        language="en",
        timezone="Europe/Kyiv",
        rss_urls=["https://www.mil.gov.ua/en/rss.xml"],
        political_spectrum="Government",
        independence="Government",
    ),

    # --- 伊朗 ---
    SourceSpec(
        name="Press TV",
        country="伊朗",
        channel_type="state_media",
        language="en",
        timezone="Asia/Tehran",
        rss_urls=["https://www.presstv.ir/RSS/Feed/Main"],
        political_spectrum="State-run",
        independence="State-owned",
    ),
    SourceSpec(
        name="Mehr News",
        country="伊朗",
        channel_type="semi_official",
        language="en",
        timezone="Asia/Tehran",
        rss_urls=["https://en.mehrnews.com/rss"],
        political_spectrum="Conservative/State-aligned",
        independence="Semi-official (IIDO)",
    ),
    SourceSpec(
        name="Tasnim News",
        country="伊朗",
        channel_type="semi_official",
        language="en",
        timezone="Asia/Tehran",
        rss_urls=["https://www.tasnimnews.com/en/rss/feed/0/7/0/1"],
        political_spectrum="Hardline/IRGC-aligned",
        independence="Semi-official",
    ),

    # --- 以色列 ---
    SourceSpec(
        name="Times of Israel",
        country="以色列",
        channel_type="major_media",
        language="en",
        timezone="Asia/Jerusalem",
        rss_urls=["https://www.timesofisrael.com/feed/"],
        political_spectrum="Center",
        independence="Independent",
    ),
    SourceSpec(
        name="Jerusalem Post",
        country="以色列",
        channel_type="major_media",
        language="en",
        timezone="Asia/Jerusalem",
        rss_urls=["https://www.jpost.com/rss/rssfeedsfrontpage.aspx"],
        political_spectrum="Center-Right",
        independence="Corporate",
    ),
    SourceSpec(
        name="Haaretz",
        country="以色列",
        channel_type="major_media",
        language="en",
        timezone="Asia/Jerusalem",
        rss_urls=["https://www.haaretz.com/cmlink/1.4605174"],
        political_spectrum="Left-leaning/Liberal",
        independence="Independent",
    ),

    # --- 国际/其他 ---
    SourceSpec(
        name="半岛电视台",
        country="卡塔尔",
        channel_type="major_media",
        language="en",
        timezone="Asia/Qatar",
        rss_urls=["https://www.aljazeera.com/xml/rss/all.xml"],
        political_spectrum="Pan-Arab/Critical of Israel",
        independence="State-funded (Qatar)",
    ),
    SourceSpec(
        name="UN News",
        country="国际组织",
        channel_type="official",
        language="en",
        timezone="America/New_York",
        rss_urls=["https://news.un.org/feed/subscribe/en/news/all/rss.xml"],
        political_spectrum="International",
        independence="International Organization",
    ),
]


CONFLICT_RULES = {
    "俄乌战争": {
        "keywords": [
            "ukraine",
            "russia",
            "kyiv",
            "kiev",
            "moscow",
            "donbas",
            "crimea",
            "kursk",
            "zaporizhzhia",
            "kherson",
            "kharkiv",
            "sumy",
            "dnipro",
            "avdiivka",
            "bakhmut",
            "pokrovsk",
            "vuhledar",
            "chornobyl",
            "belgorod",
            "black sea",
            "grain deal",
            "zelensky",
            "putin",
            "lavrov",
            "kuleba",
            "乌克兰",
            "俄罗斯",
            "基辅",
            "莫斯科",
            "顿巴斯",
            "克里米亚",
            "哈尔科夫",
            "苏梅",
            "第聂伯",
            "阿夫杰耶夫卡",
            "巴赫穆特",
            "波克罗夫斯克",
            "乌格列达尔",
            "别尔哥罗德",
            "黑海",
            "粮食协议",
            "泽连斯基",
            "普京",
            "拉夫罗夫",
        ],
        "sides": ["乌克兰", "俄罗斯", "美国/欧洲", "国际组织"],
    },
    "新一轮巴以冲突": {
        "keywords": [
            "israel",
            "gaza",
            "palestin",
            "hamas",
            "west bank",
            "rafah",
            "khan younis",
            "jabalia",
            "nuseirat",
            "jenin",
            "tulkarm",
            "idf",
            "hezbollah",
            "houthi",
            "red sea",
            "lebanon",
            "iran",
            "tehran",
            "qatar",
            "saudi",
            "u.s.",
            "us ",
            "israeli",
            "iranian",
            "netanyahu",
            "khamenei",
            "biden",
            "trump",
            "伊朗",
            "以色列",
            "加沙",
            "巴勒斯坦",
            "哈马斯",
            "约旦河西岸",
            "拉法",
            "汗尤尼斯",
            "贾巴利亚",
            "努塞拉特",
            "杰宁",
            "图勒凯尔姆",
            "真主党",
            "胡塞",
            "红海",
            "黎巴嫩",
            "卡塔尔",
            "沙特",
            "内塔尼亚胡",
            "哈梅内伊",
            "拜登",
            "特朗普",
        ],
        "sides": ["以色列", "巴勒斯坦/哈马斯", "伊朗及代理", "美国/盟友", "阿拉伯国家", "国际组织"],
    },
    "多冲突/外溢": {
        "keywords": [],
        "sides": ["多方"],
    },
}


PRIORITY_RULES = {
    "critical": (95, ["nuclear", "核", "invasion", "地面进攻", "capital", "首都", "massive", "大规模", "missile", "导弹齐射", "assassination", "刺杀", "war declaration", "宣战"]),
    "high": (75, ["airstrike", "空袭", "strike", "打击", "casualties", "伤亡", "sanction", "制裁", "ceasefire", "停火", "president", "minister", "speech", "vow", "pledge", "warn", "总统", "部长", "讲话", "誓言", "警告"]),
    "medium": (50, ["meeting", "会谈", "statement", "声明", "drone", "无人机", "aid", "援助", "visit", "访问"]),
}

def detect_event_type(text: str, score: int) -> str:
    lower = text.lower()

    if any(k in lower for k in ["president", "minister", "speech", "statement", "warn", "vow", "总统", "部长", "讲话", "声明", "警告", "誓言"]):
        return "重要表态"
            
    if score >= 90:
        return "重大升级"
    elif score >= 70:
        return "高优先级军事/外交事件"
    elif score >= 45:
        return "中等优先级动态"
    return "动态更新"


def score_priority(text: str) -> Dict[str, object]:
    lower = text.lower()
    score = 30
    level = "low"

    for level_name, (base_score, words) in PRIORITY_RULES.items():
        if any(word in lower for word in words):
            score = max(score, base_score)
            level = level_name

    event_type = detect_event_type(text, score)
    return {"score": score, "level": level, "event_type": event_type}


SEED_EVENTS = [
    ("俄乌战争", "2014-02-22 12:00:00+00:00", "俄罗斯与乌克兰局势升级（克里米亚危机）", "Russia-Ukraine escalation during Crimea crisis", "历史大事件", 92),
    ("俄乌战争", "2022-02-24 03:30:00+00:00", "俄乌全面战争爆发", "Full-scale Russia-Ukraine war begins", "历史大事件", 100),
    ("新一轮巴以冲突", "2023-10-07 00:00:00+00:00", "新一轮巴以冲突爆发（以色列-加沙战争）", "New Israel-Gaza war begins (post-7 Oct 2023)", "历史大事件", 100),
]

TIMELINE_MILESTONES = [
    (
        "俄乌战争",
        "2022-03-29 00:00:00+00:00",
        "俄军从基辅周边撤出，战场重心转向东部",
        "Russian forces withdraw from Kyiv area; focus shifts to eastern front",
        "战场转折",
        80,
        "【解释】北线撤出后，战线重组，东部成为主要消耗战方向；国际上对战争性质与持续时间预期上调。",
    ),
    (
        "俄乌战争",
        "2022-09-08 00:00:00+00:00",
        "乌军在哈尔科夫方向快速反攻，引发国际关注",
        "Ukrainian Kharkiv counteroffensive draws global attention",
        "战场转折",
        88,
        "【解释】快速推进改变外界对战场态势的判断，影响援助节奏与谈判预期。",
    ),
    (
        "俄乌战争",
        "2022-11-11 00:00:00+00:00",
        "赫尔松撤退与城市控制权变更",
        "Kherson withdrawal and change of control",
        "战场转折",
        85,
        "【解释】战略要地控制变化影响南线态势与补给安全，牵动黑海与河道相关风险评估。",
    ),
    (
        "俄乌战争",
        "2023-05-20 00:00:00+00:00",
        "巴赫穆特战役阶段性结局与国际舆论发酵",
        "Bakhmut battle reaches a phase outcome amid international debate",
        "战场消耗",
        78,
        "【解释】高消耗战役对兵力与弹药消耗预期产生示范效应，外部对补给/工业能力讨论升温。",
    ),
    (
        "俄乌战争",
        "2024-02-17 00:00:00+00:00",
        "阿夫杰耶夫卡方向态势变化引发关注",
        "Avdiivka situation shifts and triggers international reaction",
        "战场转折",
        80,
        "【解释】关键节点变化会影响顿巴斯方向火力投送与防线稳定预期，相关援助与舆论节奏常随之调整。",
    ),
    (
        "新一轮巴以冲突",
        "2023-10-27 00:00:00+00:00",
        "加沙地面行动扩大，国际人道与停火呼声上升",
        "Ground operations expand in Gaza; humanitarian and ceasefire calls rise",
        "战场升级",
        86,
        "【解释】地面行动扩大通常伴随伤亡与人道压力上升，国际斡旋与停火讨论进入高频阶段。",
    ),
    (
        "新一轮巴以冲突",
        "2023-12-01 00:00:00+00:00",
        "阶段性停火与人质/交换机制进入焦点",
        "Temporary pause and hostage/exchange mechanisms become focal",
        "外交窗口",
        82,
        "【解释】停火/暂停与交换机制往往决定后续行动强度与国际舆论走向，需跟踪执行条款与监督机制。",
    ),
    (
        "新一轮巴以冲突",
        "2024-02-01 00:00:00+00:00",
        "红海航运风险上升，能源与运价预期被重新定价",
        "Red Sea shipping risk rises; energy and freight expectations reprice",
        "外溢影响",
        84,
        "【解释】航运绕行与保险成本上升会传导至能源与部分粮食贸易成本，形成短期价格波动。",
    ),
    (
        "新一轮巴以冲突",
        "2024-05-01 00:00:00+00:00",
        "拉法相关行动与国际反应升温",
        "Rafah-related operations and international reaction intensify",
        "战场升级",
        85,
        "【解释】人口密集区域行动更易触发国际关注与政策压力，影响军援、制裁与谈判空间。",
    ),
]


MILESTONE_SEEDS = [
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2022-02-24T03:30:00+00:00",
        "date_precision": "day",
        "title_zh": "俄乌全面战争爆发",
        "summary_zh": "- 俄军自多方向进入乌克兰，战争进入全面阶段\n- 国际社会启动大规模制裁与军事/经济援助框架\n- 欧洲安全秩序与能源、粮食、金融预期快速重定价",
        "category": "开战",
        "phase": "2022：全面入侵与战线重组",
        "tags": "开战|制裁|援助|多方向",
        "source_urls": "",
        "importance": 100,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2022-03-29T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "俄军从基辅周边撤出，战场重心转向东部",
        "summary_zh": "- 北线作战阶段性结束，双方调整兵力部署\n- 东部与南部成为长期消耗战主轴\n- 战争长期化预期上升，外部援助与军工供给讨论升温",
        "category": "战场转折",
        "phase": "2022：全面入侵与战线重组",
        "tags": "基辅|战线重组|东部",
        "source_urls": "",
        "importance": 90,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2022-04-14T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "黑海“莫斯科”号巡洋舰沉没引发关注",
        "summary_zh": "- 黑海方向海空防护与制海能力议题升温\n- 沿岸目标防护与航运风险评估随之调整\n- 军事象征事件往往强化舆论与外部支援讨论",
        "category": "战场/海上",
        "phase": "2022：全面入侵与战线重组",
        "tags": "黑海|海军|制海",
        "source_urls": "",
        "importance": 78,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2022-05-16T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "马里乌波尔守军撤离亚速钢铁厂（战役阶段结束）",
        "summary_zh": "- 城市攻防与围困战进入阶段性尾声\n- 人道、战俘与信息战议题升温\n- 南线港口与海岸线控制对后续补给与外溢风险有长期影响",
        "category": "战役",
        "phase": "2022：全面入侵与战线重组",
        "tags": "马里乌波尔|围困|战俘",
        "source_urls": "",
        "importance": 82,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2022-06-23T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "乌克兰获得欧盟候选国地位（政治路径重要节点）",
        "summary_zh": "- 欧盟一体化路径被制度化推进\n- 战争与政治整合进程深度捆绑\n- 牵动长期经济援助、改革条件与安全安排讨论",
        "category": "政治/外交",
        "phase": "2022：全面入侵与战线重组",
        "tags": "欧盟|候选国|援助",
        "source_urls": "",
        "importance": 80,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2022-07-22T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "黑海粮食外运机制达成（“粮食协议”）",
        "summary_zh": "- 粮食与航运风险阶段性缓解，市场波动重新定价\n- 协议执行与续期成为后续博弈焦点\n- 相关港口、安全通道与保险成本持续牵动外溢影响",
        "category": "外交/经济",
        "phase": "2022：全面入侵与战线重组",
        "tags": "黑海|粮食|航运",
        "source_urls": "",
        "importance": 82,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2022-09-08T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "哈尔科夫方向快速反攻引发国际关注",
        "summary_zh": "- 前线态势快速变化，改变外界对战场强弱与节奏判断\n- 军援与训练计划更强调机动与合成作战能力\n- 促和/谈判窗口的舆论预期随战场变化而波动",
        "category": "战场转折",
        "phase": "2022：全面入侵与战线重组",
        "tags": "哈尔科夫|反攻|机动",
        "source_urls": "",
        "importance": 92,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2022-09-30T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "俄方宣布吞并乌克兰部分地区（国际争议升级）",
        "summary_zh": "- 领土议题与法律地位争议加剧，谈判难度上升\n- 制裁与外交对抗进一步强化\n- 战争目标与“不可谈判点”更清晰化，冲突更趋长期",
        "category": "政治/外交",
        "phase": "2022：全面入侵与战线重组",
        "tags": "吞并|制裁|谈判",
        "source_urls": "",
        "importance": 90,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2022-10-08T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "克里米亚大桥事件引发交通与补给安全关注",
        "summary_zh": "- 交通线与后勤枢纽脆弱性被放大\n- 战区补给与防护体系调整成为后续焦点\n- 对沿岸与跨区运输的风险溢价产生影响",
        "category": "战场/后勤",
        "phase": "2022：全面入侵与战线重组",
        "tags": "克里米亚|后勤|交通线",
        "source_urls": "",
        "importance": 83,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2022-10-10T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "大规模导弹/无人机打击引发能源基础设施压力",
        "summary_zh": "- 电力与供暖等民生基础设施成为高频目标\n- 冬季能源与防空需求上升\n- 外部援助更集中于防空、维修与能源系统支撑",
        "category": "基础设施/人道",
        "phase": "2022：全面入侵与战线重组",
        "tags": "导弹|无人机|电力",
        "source_urls": "",
        "importance": 86,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2022-12-21T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "泽连斯基访美并在美方寻求更多援助（象征性节点）",
        "summary_zh": "- 高层外交强化持续援助与国内政治动员\n- 援助结构更强调防空、弹药与长期财政支持\n- 外交行程常与战场态势与援助谈判节奏联动",
        "category": "重要表态",
        "phase": "2022：全面入侵与战线重组",
        "tags": "访美|援助|外交",
        "source_urls": "",
        "importance": 78,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2023-03-17T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "国际刑事司法机制对俄方高层发出逮捕令（争议与外溢）",
        "summary_zh": "- 国际法律与责任追究议题升温\n- 外交空间与第三方协调复杂度上升\n- 对冲突叙事与长期政治风险预期产生影响",
        "category": "国际机制",
        "phase": "2023：消耗战与反攻尝试",
        "tags": "国际法|责任追究|外交",
        "source_urls": "",
        "importance": 75,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2023-04-04T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "芬兰加入北约（安全格局变化）",
        "summary_zh": "- 北约与俄边界安全议题升温\n- 欧洲防务协作与军费结构调整加速\n- 冲突对地区安全架构的长期影响进一步固化",
        "category": "政治/外交",
        "phase": "2023：消耗战与反攻尝试",
        "tags": "北约|芬兰|安全格局",
        "source_urls": "",
        "importance": 80,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2023-07-17T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "黑海粮食机制中断后航运与粮价风险再度上升",
        "summary_zh": "- 航运保险与绕行成本上升，影响部分粮贸与能源物流预期\n- 港口与航道安全成为外溢风险核心\n- 各方围绕替代通道与护航方案的讨论升温",
        "category": "外交/经济",
        "phase": "2023：消耗战与反攻尝试",
        "tags": "黑海|粮食|航运",
        "source_urls": "",
        "importance": 82,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2024-03-07T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "瑞典加入北约（欧洲安全架构继续变化）",
        "summary_zh": "- 北欧与波罗的海安全协作进一步制度化\n- 欧洲防务产业与采购协同更紧密\n- 冲突外溢对联盟扩展与威慑策略影响持续",
        "category": "政治/外交",
        "phase": "2024：攻防转换与外溢风险",
        "tags": "北约|瑞典|安全架构",
        "source_urls": "",
        "importance": 78,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2022-11-11T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "赫尔松态势变化与控制权调整",
        "summary_zh": "- 南线关键节点变化影响补给与防御纵深\n- 第聂伯河相关渡河/火力控制成为后续焦点\n- 牵动黑海沿岸安全与外溢风险评估",
        "category": "战场转折",
        "phase": "2022：全面入侵与战线重组",
        "tags": "赫尔松|南线|第聂伯河",
        "source_urls": "",
        "importance": 88,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2023-05-20T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "巴赫穆特战役进入阶段性结局（高消耗战典型）",
        "summary_zh": "- 长期巷战凸显兵力、火力与补给消耗的结构性问题\n- 外界对弹药产能、动员与轮换体系的关注上升\n- 战场叙事与国内政治承受力成为舆论焦点",
        "category": "战役",
        "phase": "2023：消耗战与反攻尝试",
        "tags": "巴赫穆特|消耗战|弹药",
        "source_urls": "",
        "importance": 85,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2023-06-06T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "卡霍夫卡水坝被毁引发人道与基础设施冲击",
        "summary_zh": "- 洪水与基础设施损毁导致人道与生态风险上升\n- 前线地形与渡河条件发生变化，影响南线作战\n- 国际关注聚焦战争对关键基础设施的系统性破坏",
        "category": "外溢/人道",
        "phase": "2023：消耗战与反攻尝试",
        "tags": "水坝|基础设施|人道",
        "source_urls": "",
        "importance": 86,
    },
    {
        "conflict": "俄乌战争",
        "occurred_utc": "2024-02-17T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "阿夫杰耶夫卡方向态势变化引发关注",
        "summary_zh": "- 顿巴斯关键节点变化影响前线火力投送与防线稳定\n- 外界对援助节奏、弹药缺口与防空压力讨论升温\n- 战场进展对谈判预期与舆论环境产生联动影响",
        "category": "战役",
        "phase": "2024：攻防转换与外溢风险",
        "tags": "阿夫杰耶夫卡|顿巴斯|防线",
        "source_urls": "",
        "importance": 84,
    },

    {
        "conflict": "新一轮巴以冲突",
        "occurred_utc": "2023-10-07T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "新一轮巴以冲突爆发",
        "summary_zh": "- 以色列与加沙方向冲突升级，地区安全风险迅速外溢\n- 人质、停火与人道通道成为国际斡旋核心议题\n- 西岸、黎以边境与红海航运风险随局势演变而上升",
        "category": "开战",
        "phase": "2023：冲突爆发与地面行动",
        "tags": "开战|人质|人道",
        "source_urls": "",
        "importance": 100,
    },
    {
        "conflict": "新一轮巴以冲突",
        "occurred_utc": "2023-10-27T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "加沙地面行动扩大，国际停火与人道呼声上升",
        "summary_zh": "- 地面行动扩大通常伴随伤亡与人道压力显著增加\n- 国际斡旋聚焦“暂停/停火—人质交换—人道准入”组合\n- 舆论与政策压力影响各方行动强度与谈判窗口",
        "category": "战场升级",
        "phase": "2023：冲突爆发与地面行动",
        "tags": "地面行动|停火|人道",
        "source_urls": "",
        "importance": 88,
    },
    {
        "conflict": "新一轮巴以冲突",
        "occurred_utc": "2023-11-15T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "安理会通过首份人道相关决议（强调人道暂停等）",
        "summary_zh": "- 多边机制推动人道准入与保护平民议题制度化\n- 对“暂停/停火/准入”叙事与谈判框架产生影响\n- 但执行与监督通常仍受战场与政治拉扯制约",
        "category": "国际机制",
        "phase": "2023：冲突爆发与地面行动",
        "tags": "安理会|人道|决议",
        "source_urls": "",
        "importance": 76,
    },
    {
        "conflict": "新一轮巴以冲突",
        "occurred_utc": "2023-11-24T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "阶段性停火/暂停窗口与人质交换进入核心进程",
        "summary_zh": "- “暂停—交换—人道准入”成为高频斡旋组合\n- 停火窗口对后续行动强度、国际舆论与政策压力有直接影响\n- 条款执行、延长与破裂风险决定局势走向",
        "category": "停火/谈判",
        "phase": "2023：冲突爆发与地面行动",
        "tags": "停火|人质|斡旋",
        "source_urls": "",
        "importance": 90,
    },
    {
        "conflict": "新一轮巴以冲突",
        "occurred_utc": "2023-12-01T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "暂停窗口结束后，战事与人道压力再度上升",
        "summary_zh": "- 人道供给、医疗系统与民众迁徙压力加剧\n- 外交斡旋转向争取新一轮暂停或更长期停火\n- 外溢战线（黎以边境、红海等）风险评估持续波动",
        "category": "人道/外溢",
        "phase": "2023：冲突爆发与地面行动",
        "tags": "人道|外溢|暂停结束",
        "source_urls": "",
        "importance": 78,
    },
    {
        "conflict": "新一轮巴以冲突",
        "occurred_utc": "2024-03-25T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "安理会推动停火相关决议成为国际讨论焦点",
        "summary_zh": "- 多边框架对停火与人道准入的政治压力增强\n- 各方围绕决议措辞、执行与监督存在拉扯\n- 决议往往影响后续谈判节奏与国际援助协调",
        "category": "国际机制",
        "phase": "2024：外溢加深与政治拉扯",
        "tags": "安理会|停火|人道",
        "source_urls": "",
        "importance": 76,
    },
    {
        "conflict": "新一轮巴以冲突",
        "occurred_utc": "2024-04-01T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "人道援助安全与机制信任问题引发国际震动",
        "summary_zh": "- 人道机构与援助通道安全成为国际舆论焦点\n- 影响援助投送节奏、路线与协调机制\n- 进一步强化对“保护平民/准入规则/责任追究”的讨论",
        "category": "人道/外溢",
        "phase": "2024：外溢加深与政治拉扯",
        "tags": "人道援助|准入|责任",
        "source_urls": "",
        "importance": 80,
    },
    {
        "conflict": "新一轮巴以冲突",
        "occurred_utc": "2024-04-14T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "地区外溢升级：伊朗—以色列直接交锋引发关注",
        "summary_zh": "- 冲突从加沙主线向地区对抗外溢的风险上升\n- 空域安全、航运与能源预期被重新定价\n- 大国与地区国家密集斡旋以防止进一步升级",
        "category": "外溢升级",
        "phase": "2024：外溢加深与政治拉扯",
        "tags": "外溢|伊朗|以色列",
        "source_urls": "",
        "importance": 86,
    },
    {
        "conflict": "新一轮巴以冲突",
        "occurred_utc": "2024-05-10T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "联大关于巴勒斯坦地位的讨论与表决引发关注",
        "summary_zh": "- 巴勒斯坦地位、建国路径与国际承认议题升温\n- 多边场域的投票与表态影响外交压力与联盟结构\n- 与停火、人道准入和战后治理讨论形成联动",
        "category": "国际机制",
        "phase": "2024：外溢加深与政治拉扯",
        "tags": "联大|巴勒斯坦|地位",
        "source_urls": "",
        "importance": 74,
    },
    {
        "conflict": "新一轮巴以冲突",
        "occurred_utc": "2023-11-21T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "中方在金砖巴以特别视频峰会阐述停火与“两国方案”主张",
        "summary_zh": "- 提出当务之急：立即停火止战、保障人道救援、安全通道\n- 强调根本出路：落实“两国方案”并推动更具权威性的国际和会\n- 该类高层表态常影响国际协调节奏与安理会讨论焦点",
        "category": "重要表态",
        "phase": "2023：冲突爆发与地面行动",
        "tags": "表态|两国方案|金砖",
        "source_urls": "https://www.fmprc.gov.cn/zyxw/202311/t20231121_11184740.shtml",
        "importance": 80,
    },
    {
        "conflict": "新一轮巴以冲突",
        "occurred_utc": "2024-01-26T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "国际司法进程与人道议题进一步进入焦点",
        "summary_zh": "- 国际法律与人道框架在舆论与外交场域的重要性上升\n- “停火/准入/保护平民”与战后治理讨论更频繁\n- 相关进程往往对援助、制裁与外交压力形成叠加效应",
        "category": "国际机制",
        "phase": "2024：外溢加深与政治拉扯",
        "tags": "国际法|人道|斡旋",
        "source_urls": "",
        "importance": 78,
    },
    {
        "conflict": "新一轮巴以冲突",
        "occurred_utc": "2024-05-30T00:00:00+00:00",
        "date_precision": "day",
        "title_zh": "中国—阿拉伯国家发布巴勒斯坦问题联合声明（全文）",
        "summary_zh": "- 强调推动尽快停火止战与落实联合国相关决议\n- 重申“两国方案”与政治解决路径的重要性\n- 联合声明类文件常用于凝聚区域与多边协调共识",
        "category": "联合声明",
        "phase": "2024：外溢加深与政治拉扯",
        "tags": "联合声明|两国方案|阿盟",
        "source_urls": "https://www.mfa.gov.cn/ziliao_674904/1179_674909/202405/t20240531_11366712.shtml",
        "importance": 80,
    },
]

IMPACT_PATTERNS = [
    {
        "name": "核风险",
        "triggers": ["nuclear", "核", "strategic forces", "nuclear deterr", "核威慑"],
        "zh": "核相关表述增加，市场与外交层面可能出现更强风险溢价与危机管控信号。",
        "en": "Elevated nuclear-related rhetoric may increase risk premium and crisis-management signaling.",
    },
    {
        "name": "战场升级",
        "triggers": ["invasion", "ground operation", "offensive", "assault", "地面进攻", "大规模", "missile", "导弹", "airstrike", "空袭"],
        "zh": "军事行动强度上升的迹象，可能推动短期局势升级并影响人道与外交节奏。",
        "en": "Signs of heightened military activity may drive short-term escalation and affect humanitarian/diplomatic tempo.",
    },
    {
        "name": "外交窗口",
        "triggers": ["ceasefire", "停火", "talks", "meeting", "会谈", "statement", "声明", "envoy", "特使"],
        "zh": "出现外交沟通或停火相关线索，可能打开（或关闭）阶段性缓和窗口，需关注执行与各方条件。",
        "en": "Diplomatic or ceasefire signals may open (or close) a de-escalation window; watch terms and enforcement.",
    },
    {
        "name": "经济制裁",
        "triggers": ["sanction", "制裁", "export control", "资产冻结", "oil", "energy", "shipping"],
        "zh": "制裁/反制裁与能源运输相关信息可能影响供应链与能源价格预期，并产生外溢经济效应。",
        "en": "Sanctions/retaliation and energy-shipping signals may shift supply-chain and energy price expectations.",
    },
    {
        "name": "领导人表态",
        "triggers": ["president", "minister", "leader", "speech", "warn", "vow", "总统", "部长", "讲话", "警告", "誓言"],
        "zh": "高层表态可能改变谈判筹码与国内动员强度，但需结合后续政策与行动验证。",
        "en": "Leadership statements can shift bargaining positions and mobilization, but require policy/action follow-through.",
    },
]


def analyze_impact(text: str, priority_level: str) -> str:
    lower = text.lower()
    matched = []
    for p in IMPACT_PATTERNS:
        if any(t.lower() in lower for t in p["triggers"]):
            matched.append(p)

    base_conf = {"critical": 0.75, "high": 0.6, "medium": 0.45, "low": 0.35}.get(priority_level, 0.35)
    conf = min(0.9, base_conf + 0.05 * max(0, len(matched) - 1))

    if not matched:
        zh = f"影响分析（初步，置信度{conf:.2f}）：未命中明确主题关键词，暂按常规态势跟踪，等待更多信息确认。"
        en = f"Impact (preliminary, confidence {conf:.2f}): No strong topical triggers detected; track as routine pending confirmation."
        return zh + "\n" + en

    zh_items = "；".join([p["zh"] for p in matched[:3]])
    en_items = "; ".join([p["en"] for p in matched[:3]])
    zh = f"影响分析（初步，置信度{conf:.2f}）：{zh_items}（仅基于标题/摘要关键词推断，非事实断言）"
    en = f"Impact (preliminary, confidence {conf:.2f}): {en_items} (keyword-based inference from headline/summary; not a factual assertion)."
    return zh + "\n" + en


def _execute_with_retry(conn: sqlite3.Connection, sql: str, params: Tuple[object, ...] = ()) -> None:
    for attempt in range(10):
        try:
            conn.execute(sql, params)
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(0.25 * (attempt + 1))
                continue
            raise
    conn.execute(sql, params)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conflict TEXT NOT NULL,
            title_zh TEXT NOT NULL,
            title_original TEXT NOT NULL,
            summary_zh TEXT NOT NULL,
            summary_original TEXT NOT NULL,
            source_name TEXT NOT NULL,
            source_country TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_url TEXT NOT NULL UNIQUE,
            side_tag TEXT NOT NULL,
            event_type TEXT NOT NULL,
            priority_score INTEGER NOT NULL,
            priority_level TEXT NOT NULL,
            impact_analysis TEXT DEFAULT '',
            political_spectrum TEXT DEFAULT 'N/A',
            independence TEXT DEFAULT 'N/A',
            published_utc TEXT NOT NULL,
            published_local TEXT NOT NULL,
            published_bj TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    # Column migrations (retry on lock)
    try:
        conn.execute("SELECT impact_analysis FROM events LIMIT 1")
    except sqlite3.OperationalError:
        _execute_with_retry(conn, "ALTER TABLE events ADD COLUMN impact_analysis TEXT DEFAULT ''")

    try:
        conn.execute("SELECT political_spectrum FROM events LIMIT 1")
    except sqlite3.OperationalError:
        _execute_with_retry(conn, "ALTER TABLE events ADD COLUMN political_spectrum TEXT DEFAULT 'N/A'")
        _execute_with_retry(conn, "ALTER TABLE events ADD COLUMN independence TEXT DEFAULT 'N/A'")

    _execute_with_retry(
        conn,
        """
        CREATE TABLE IF NOT EXISTS stance_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            country TEXT NOT NULL,
            issuer TEXT NOT NULL,
            doc_type TEXT DEFAULT 'statement',
            title_original TEXT NOT NULL,
            language TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            published_utc TEXT,
            published_bj TEXT,
            summary_original TEXT DEFAULT '',
            translation_zh TEXT DEFAULT '',
            summary_zh TEXT DEFAULT '',
            updated_utc TEXT DEFAULT ''
        )
        """,
    )

    try:
        conn.execute("SELECT doc_type FROM stance_docs LIMIT 1")
    except sqlite3.OperationalError:
        _execute_with_retry(conn, "ALTER TABLE stance_docs ADD COLUMN doc_type TEXT DEFAULT 'statement'")

    for col, ddl in [
        ("summary_zh", "ALTER TABLE stance_docs ADD COLUMN summary_zh TEXT DEFAULT ''"),
        ("updated_utc", "ALTER TABLE stance_docs ADD COLUMN updated_utc TEXT DEFAULT ''"),
    ]:
        try:
            conn.execute(f"SELECT {col} FROM stance_docs LIMIT 1")
        except sqlite3.OperationalError:
            _execute_with_retry(conn, ddl)

    _execute_with_retry(
        conn,
        """
        CREATE TABLE IF NOT EXISTS event_ai (
            event_id INTEGER PRIMARY KEY,
            terms_original TEXT DEFAULT '',
            terms_zh TEXT DEFAULT '',
            impact_ai TEXT DEFAULT '',
            title_zh_ai TEXT DEFAULT '',
            summary_zh_ai TEXT DEFAULT '',
            updated_utc TEXT DEFAULT ''
        )
        """,
    )

    _execute_with_retry(
        conn,
        """
        CREATE TABLE IF NOT EXISTS milestones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conflict TEXT NOT NULL,
            occurred_utc TEXT NOT NULL,
            occurred_bj TEXT NOT NULL,
            date_precision TEXT DEFAULT 'day',
            title_zh TEXT NOT NULL,
            summary_zh TEXT NOT NULL,
            category TEXT DEFAULT '',
            phase TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            source_urls TEXT DEFAULT '',
            importance INTEGER DEFAULT 50,
            updated_utc TEXT DEFAULT '',
            UNIQUE(conflict, occurred_utc, title_zh)
        )
        """,
    )

    try:
        conn.execute("SELECT updated_utc FROM milestones LIMIT 1")
    except sqlite3.OperationalError:
        _execute_with_retry(conn, "ALTER TABLE milestones ADD COLUMN updated_utc TEXT DEFAULT ''")

    for col, ddl in [
        ("title_zh_ai", "ALTER TABLE event_ai ADD COLUMN title_zh_ai TEXT DEFAULT ''"),
        ("summary_zh_ai", "ALTER TABLE event_ai ADD COLUMN summary_zh_ai TEXT DEFAULT ''"),
    ]:
        try:
            conn.execute(f"SELECT {col} FROM event_ai LIMIT 1")
        except sqlite3.OperationalError:
            _execute_with_retry(conn, ddl)

    row = conn.execute("SELECT value FROM meta WHERE key = 'remove_2026_seed_v1'").fetchone()
    if not row:
        _execute_with_retry(conn, "BEGIN IMMEDIATE")
        try:
            _execute_with_retry(conn, "DELETE FROM events WHERE source_type IN ('timeline_seed','timeline_milestone') AND published_utc LIKE '2026-%'")
            _execute_with_retry(conn, "DELETE FROM events WHERE title_zh LIKE '2026%'")
            _execute_with_retry(conn, "INSERT OR REPLACE INTO meta(key, value) VALUES('remove_2026_seed_v1', 'done')")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    row = conn.execute("SELECT value FROM meta WHERE key = 'fix_event_type_leader_v1'").fetchone()
    if not row:
        _execute_with_retry(conn, "BEGIN IMMEDIATE")
        try:
            _execute_with_retry(conn, "UPDATE events SET event_type='重要表态' WHERE event_type LIKE '领导人表态 (%'")
            _execute_with_retry(conn, "INSERT OR REPLACE INTO meta(key, value) VALUES('fix_event_type_leader_v1', 'done')")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # Normalize naming once (avoid doing UPDATE on every Streamlit rerun)
    row = conn.execute("SELECT value FROM meta WHERE key = 'migration_conflict_names_v1'").fetchone()
    if not row:
        _execute_with_retry(conn, "BEGIN IMMEDIATE")
        try:
            _execute_with_retry(conn, "UPDATE events SET conflict = '俄乌战争' WHERE conflict = '俄乌冲突'")
            _execute_with_retry(conn, "UPDATE events SET conflict = '新一轮巴以冲突' WHERE conflict = '美伊战争'")
            _execute_with_retry(conn, "INSERT OR REPLACE INTO meta(key, value) VALUES('migration_conflict_names_v1', 'done')")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    conn.commit()
    return conn


def seed_major_events(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for conflict, utc_time, zh_text, original_text, event_type, score in SEED_EVENTS:
        cur.execute(
            """
            INSERT OR IGNORE INTO events (
                conflict, title_zh, title_original, summary_zh, summary_original,
                source_name, source_country, source_type, source_url, side_tag, event_type,
                priority_score, priority_level, published_utc, published_local, published_bj,
                impact_analysis, political_spectrum, independence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conflict,
                zh_text,
                original_text,
                zh_text,
                original_text,
                "系统基线",
                "国际",
                "timeline_seed",
                f"seed://{conflict}/{utc_time}",
                "国际组织",
                event_type,
                score,
                "critical" if score >= 90 else "medium",
                utc_time,
                utc_time,
                to_bj(utc_time),
                "【历史基线】该事件为冲突重要转折点。\n[Historical Baseline] Major turning point in conflict.",
                "N/A",
                "N/A",
            ),
        )

    for conflict, utc_time, zh_text, original_text, event_type, score, note_zh in TIMELINE_MILESTONES:
        cur.execute(
            """
            INSERT OR IGNORE INTO events (
                conflict, title_zh, title_original, summary_zh, summary_original,
                source_name, source_country, source_type, source_url, side_tag, event_type,
                priority_score, priority_level, published_utc, published_local, published_bj,
                impact_analysis, political_spectrum, independence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conflict,
                zh_text,
                original_text,
                note_zh,
                original_text,
                "系统里程碑",
                "国际",
                "timeline_milestone",
                f"milestone://{conflict}/{utc_time}",
                "国际组织",
                event_type,
                score,
                "high" if score >= 75 else "medium",
                utc_time,
                utc_time,
                to_bj(utc_time),
                f"{note_zh}\n[Milestone] {original_text}",
                "N/A",
                "N/A",
            ),
        )
    conn.commit()


def seed_milestones(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for m in MILESTONE_SEEDS:
        utc = str(m["occurred_utc"])
        try:
            bj = to_bj(utc)
        except Exception:
            bj = utc
        cur.execute(
            """
            INSERT OR IGNORE INTO milestones(
                conflict, occurred_utc, occurred_bj, date_precision, title_zh, summary_zh,
                category, phase, tags, source_urls, importance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                m.get("conflict", ""),
                utc,
                bj,
                m.get("date_precision", "day"),
                m.get("title_zh", ""),
                m.get("summary_zh", ""),
                m.get("category", ""),
                m.get("phase", ""),
                m.get("tags", ""),
                m.get("source_urls", ""),
                int(m.get("importance", 50)),
            ),
        )
    conn.commit()


def insert_milestones(conn: sqlite3.Connection, items: List[Dict[str, object]]) -> int:
    if not items:
        return 0
    cur = conn.cursor()
    inserted = 0
    for m in items:
        utc = str(m.get("occurred_utc", "")).strip()
        if not utc:
            continue
        try:
            bj = to_bj(utc)
        except Exception:
            bj = utc
        title = str(m.get("title_zh", "")).strip()
        summary = str(m.get("summary_zh", "")).strip()
        if not title or not summary:
            continue
        conflict = str(m.get("conflict", "")).strip()
        cur.execute(
            """
            INSERT OR IGNORE INTO milestones(
                conflict, occurred_utc, occurred_bj, date_precision, title_zh, summary_zh,
                category, phase, tags, source_urls, importance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conflict,
                utc,
                bj,
                str(m.get("date_precision", "day") or "day"),
                title,
                summary,
                str(m.get("category", "") or ""),
                str(m.get("phase", "") or ""),
                str(m.get("tags", "") or ""),
                str(m.get("source_urls", "") or ""),
                int(m.get("importance", 70) or 70),
            ),
        )
        if cur.rowcount:
            inserted += 1
    conn.commit()
    return inserted


def milestone_year_counts(df_ms: pd.DataFrame) -> Dict[int, int]:
    if df_ms.empty:
        return {}
    dt = pd.to_datetime(df_ms["occurred_utc"], errors="coerce", utc=True)
    years = dt.dt.year.dropna().astype(int)
    out: Dict[int, int] = {}
    for y, cnt in years.value_counts().items():
        out[int(y)] = int(cnt)
    return out


def read_milestones(conn: sqlite3.Connection, conflict: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT * FROM milestones
        WHERE conflict = ?
        ORDER BY datetime(occurred_utc) DESC, importance DESC
        """,
        conn,
        params=(conflict,),
    )


def strip_html(text: str) -> str:
    t = text or ""
    t = re.sub(r"<\s*script[^>]*>[\s\S]*?<\s*/\s*script\s*>", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"<\s*style[^>]*>[\s\S]*?<\s*/\s*style\s*>", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"<\s*img[^>]*>", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"!\[[^\]]*\]\([^\)]+\)", " ", t)
    t = re.sub(r"\[[^\]]+\]\([^\)]+\)", " ", t)
    return t


def h(text: object) -> str:
    return html_lib.escape(str(text or ""), quote=True)


def has_zh(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def normalize_text(text: str) -> str:
    raw = strip_html(str(text or ""))
    cleaned = re.sub(r"\s+", " ", raw.strip())
    return cleaned[:800]


def as_text(val: object) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        return " ".join([str(x) for x in val])
    return str(val)


def detect_conflict(text: str) -> Optional[str]:
    lower = text.lower()
    scored: List[Tuple[str, int]] = []
    for conflict, cfg in CONFLICT_RULES.items():
        if conflict == "多冲突/外溢":
            continue
        hits = 0
        for keyword in cfg["keywords"]:
            if keyword and keyword in lower:
                hits += 1
        scored.append((conflict, hits))

    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored or scored[0][1] == 0:
        return None

    if len(scored) > 1 and scored[0][1] == scored[1][1] and scored[0][1] >= 2:
        return "多冲突/外溢"
    return scored[0][0]


def detect_side(conflict: str, text: str) -> str:
    lower = text.lower()
    if conflict == "俄乌冲突":
        conflict = "俄乌战争"
    if conflict == "美伊战争":
        conflict = "新一轮巴以冲突"

    if conflict == "俄乌战争":
        if "ukrain" in lower or "乌克兰" in lower or "kyiv" in lower:
            return "乌克兰"
        if "russia" in lower or "俄罗斯" in lower or "moscow" in lower:
            return "俄罗斯"
        if any(k in lower for k in ["eu", "nato", "美国", "u.s", "us ", "britain", "france", "germany"]):
            return "美国/欧洲"
        return "国际组织"

    if conflict == "新一轮巴以冲突":
        if any(k in lower for k in ["israel", "israeli", "idf", "以色列"]):
            return "以色列"
        if any(k in lower for k in ["palestin", "hamas", "gaza", "巴勒斯坦", "哈马斯", "加沙"]):
            return "巴勒斯坦/哈马斯"
        if any(k in lower for k in ["iran", "iranian", "tehran", "hezbollah", "houthi", "伊朗", "真主党", "胡塞"]):
            return "伊朗及代理"
        if any(k in lower for k in ["u.s", "us ", "united states", "美国", "uk ", "britain", "france", "germany", "eu", "nato"]):
            return "美国/盟友"
        if any(k in lower for k in ["qatar", "saudi", "uae", "egypt", "jordan", "turkey", "卡塔尔", "沙特", "埃及", "约旦", "土耳其"]):
            return "阿拉伯国家"
        return "国际组织"

    return "多方"
def parse_dt(entry: dict, source_tz: str) -> datetime:
    raw = entry.get("published") or entry.get("updated") or datetime.now(timezone.utc).isoformat()
    dt = date_parser.parse(raw)
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=ZoneInfo(source_tz))
    return dt.astimezone(timezone.utc)


def to_bj(utc_text: str) -> str:
    dt = date_parser.parse(utc_text).astimezone(BEIJING_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S%z")


def local_time(utc_dt: datetime, source_tz: str) -> str:
    return utc_dt.astimezone(ZoneInfo(source_tz)).strftime("%Y-%m-%d %H:%M:%S%z")


def fetch_all_sources(conn: sqlite3.Connection, days_back: int = 14, per_feed: int = 80) -> int:
    inserted = 0
    cur = conn.cursor()
    for source in SOURCES:
        for url in source.rss_urls:
            try:
                parsed = feedparser.parse(url)
                entries = parsed.entries[:per_feed]
            except Exception:
                continue
            for entry in entries:
                title = normalize_text(as_text(entry.get("title", "")))
                summary = normalize_text(as_text(entry.get("summary", "")))
                merged = f"{title} {summary}"
                conflict = detect_conflict(merged)
                if not conflict:
                    continue
                utc_dt = parse_dt(entry, source.timezone)
                now_utc = datetime.now(timezone.utc)
                if utc_dt < now_utc - timedelta(days=days_back):
                    continue
                priority = score_priority(merged)
                side_tag = detect_side(conflict, merged)
                impact_analysis_text = analyze_impact(merged, str(priority.get("level", "low")))
                link_val = as_text(entry.get("link", "")).strip()
                source_url = link_val or f"{url}#{title[:48]}"
                try:
                    cur.execute(
                        """
                        INSERT INTO events (
                            conflict, title_zh, title_original, summary_zh, summary_original,
                            source_name, source_country, source_type, source_url, side_tag,
                            event_type, priority_score, priority_level, impact_analysis,
                            political_spectrum, independence,
                            published_utc, published_local, published_bj
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            conflict,
                            title,
                            title,
                            summary or title,
                            summary or title,
                            source.name,
                            source.country,
                            source.channel_type,
                            source_url,
                            side_tag,
                            priority["event_type"],
                            priority["score"],
                            priority["level"],
                            impact_analysis_text,
                            source.political_spectrum,
                            source.independence,
                            utc_dt.isoformat(),
                            local_time(utc_dt, source.timezone),
                            utc_dt.astimezone(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S%z"),
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    continue
    conn.commit()
    return inserted


def ensure_daily_update(conn: sqlite3.Connection) -> int:
    today_bj = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d")
    cur = conn.cursor()
    row = cur.execute("SELECT value FROM meta WHERE key = 'last_update_bj'").fetchone()
    if row and row[0] == today_bj:
        return 0
    inserted = fetch_all_sources(conn, days_back=14, per_feed=60)
    cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('last_update_bj', ?)", (today_bj,))
    conn.commit()
    return inserted


def read_events(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT * FROM events
        ORDER BY datetime(substr(replace(published_bj, '+0800', ''), 1, 19)) DESC
        """,
        conn,
    )


def summary_by_date(df: pd.DataFrame, date_str: str, conflict: str) -> Dict[str, List[str]]:
    scoped = df[(df["conflict"] == conflict) & (df["published_bj"].str.startswith(date_str))]
    sides = CONFLICT_RULES[conflict]["sides"]
    zh_lines = [f"- 日期（北京时间）：{date_str}", f"- 主题：{conflict}"]
    en_lines = [f"- Date (Beijing): {date_str}", f"- Conflict: {conflict}"]
    for side in sides:
        side_rows = scoped[scoped["side_tag"] == side].sort_values("priority_score", ascending=False).head(2)
        if side_rows.empty:
            zh_lines.append(f"- {side}：今日未检索到满足白名单条件的高可信新增报道。")
            en_lines.append(f"- {side}: No new high-credibility whitelisted updates were captured today.")
            continue
        top = side_rows.iloc[0]
        origin_terms = ""
        try:
            conn = get_conn()
            ai_map = read_event_ai(conn, [int(top["id"])]) if pd.notna(top.get("id")) else {}
            if pd.notna(top.get("id")):
                origin_terms = (ai_map.get(int(top["id"]), {}) or {}).get("terms_original", "")
            conn.close()
        except Exception:
            origin_terms = ""
        if not origin_terms:
            origin_terms = ", ".join(heuristic_terms(str(top.get("title_original", ""))))
        impact = str(top.get("impact_analysis", "") or "")
        if pd.notna(top.get("id")):
            try:
                conn = get_conn()
                ai_map = read_event_ai(conn, [int(top["id"])])
                ai_imp = (ai_map.get(int(top["id"]), {}) or {}).get("impact_ai", "")
                if ai_imp:
                    impact = ai_imp
                conn.close()
            except Exception:
                pass
        impact_zh = impact.split("\n")[0] if impact else ""
        impact_en = impact.split("\n")[1] if impact and "\n" in impact else impact

        zh_lines.append(
            f"- {side}：{top['title_zh']}（优先级 {top['priority_level']}，来源 [{top['source_name']}]({top['source_url']})，北京时间 {top['published_bj']}，当地时间 {top['published_local']}，关键词：{origin_terms or 'N/A'}）\n  > **影响分析**：{impact_zh}"
        )
        en_lines.append(
            f"- {side}: {top['title_original']} (Priority {top['priority_level']}, Source [{top['source_name']}]({top['source_url']}), Beijing {top['published_bj']}, Local {top['published_local']})\n  > **Impact**: {impact_en}"
        )
    return {"zh": zh_lines, "original": en_lines}


def _has_openai_key() -> bool:
    return bool((st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")).strip())


def ensure_ai_enrichment(conn: sqlite3.Connection, df_in: pd.DataFrame, max_items: int) -> int:
    if max_items <= 0:
        return 0
    if df_in.empty:
        return 0
    if not _has_openai_key():
        return 0
    ids = [int(x) for x in df_in["id"].dropna().astype(int).tolist()]
    if not ids:
        return 0
    ai_map = read_event_ai(conn, ids)
    targets = []
    for rid in ids:
        cur = ai_map.get(rid, {})
        if not (cur.get("summary_zh_ai") or cur.get("impact_ai") or cur.get("terms_original")):
            targets.append(rid)
    targets = targets[:max_items]
    if not targets:
        return 0
    done = 0
    prog = st.progress(0)
    for i, rid in enumerate(targets, start=1):
        row = df_in[df_in["id"] == rid].iloc[0]
        title_o = str(row.get("title_original", "") or "")
        summary_o = normalize_text(str(row.get("summary_original", "") or ""))
        payload = ai_translate_and_expand(normalize_text(title_o), summary_o)
        if payload:
            upsert_event_ai_full(conn, rid, payload)
            done += 1
        prog.progress(i / len(targets))
    return done


def render_event_card(row: pd.Series, ai: dict) -> None:
    title = (ai.get("title_zh_ai") or "").strip() or str(row.get("title_zh", "") or "").strip()
    summary = (ai.get("summary_zh_ai") or "").strip() or str(row.get("summary_zh", "") or "").strip()
    terms = (ai.get("terms_zh") or "").strip() or ""
    terms_o = (ai.get("terms_original") or "").strip() or ""

    if title and not has_zh(title):
        title = f"（待翻译）{title}"
    if summary and not has_zh(summary):
        summary = "（待生成中文要点摘要，可在本页点击‘生成中文摘要’）"

    chips = []
    chips.append(f"<span class=\"ww-chip\">{h(row.get('side_tag',''))}</span>")
    chips.append(f"<span class=\"ww-chip\">{h(row.get('event_type',''))}</span>")
    chips.append(f"<span class=\"ww-chip\">重要性 {h(row.get('priority_score',''))}</span>")
    if str(row.get("priority_level", "")) in ["critical", "high"]:
        chips.append(f"<span class=\"ww-chip ww-chip-warn\">{h(row.get('priority_level',''))}</span>")
    chips_html = "".join(chips)

    meta = f"北京时间 {row.get('published_bj','')}｜来源 {row.get('source_name','')}"
    url = str(row.get("source_url", "") or "").strip()
    link = f"<div class=\"ww-link\"><a href=\"{h(url)}\" target=\"_blank\">打开原文</a></div>" if url else ""

    parts = []
    if terms:
        parts.append(f"关键词：{terms}\n")
    elif terms_o:
        parts.append(f"关键词（原文）：{terms_o}\n")
    if summary:
        parts.append(summary)
    body = "\n".join([p for p in parts if p.strip()])

    st.markdown(
        f"""
        <div class="ww-card">
          <div class="ww-title">{h(title)}</div>
          <div class="ww-meta">{h(meta)}</div>
          <div class="ww-chips">{chips_html}</div>
          <div class="ww-body">{h(body)}</div>
          {link}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_timeline_like_mfa(conn: sqlite3.Connection, df_in: pd.DataFrame, conflict: str) -> None:
    if df_in.empty:
        st.info("该筛选条件下暂无事件。")
        return
    df_in = df_in.sort_values("published_bj_dt", ascending=False).copy()
    ids = [int(x) for x in df_in["id"].dropna().astype(int).tolist()]
    ai_map = read_event_ai(conn, ids)

    df_in["day"] = df_in["published_bj_dt"].dt.date.astype(str)
    for day, g in df_in.groupby("day", sort=False):
        st.markdown(f"<div class=\"ww-day\">{day}</div>", unsafe_allow_html=True)
        for _, r in g.iterrows():
            ai = ai_map.get(int(r["id"]), {}) if pd.notna(r.get("id")) else {}
            render_event_card(r, ai)


def render_milestone_card(row: pd.Series) -> None:
    title = str(row.get("title_zh", "") or "").strip()
    summary = str(row.get("summary_zh", "") or "").strip()
    category = str(row.get("category", "") or "").strip()
    phase = str(row.get("phase", "") or "").strip()
    tags = str(row.get("tags", "") or "").strip()
    occurred_bj = str(row.get("occurred_bj", "") or "").strip()
    urls = str(row.get("source_urls", "") or "").strip()

    chips = []
    if phase:
        chips.append(f"<span class=\"ww-chip\">{h(phase)}</span>")
    if category:
        chips.append(f"<span class=\"ww-chip\">{h(category)}</span>")
    if tags:
        for t in [x.strip() for x in tags.split("|") if x.strip()][:6]:
            chips.append(f"<span class=\"ww-chip\">{h(t)}</span>")
    chips_html = "".join(chips)

    links = ""
    if urls:
        items = [u.strip() for u in urls.split("\n") if u.strip()]
        if items:
            link_lines = []
            for u in items[:4]:
                link_lines.append(f"<a href=\"{h(u)}\" target=\"_blank\">来源链接</a>")
            links = "<div class=\"ww-link\">" + " | ".join(link_lines) + "</div>"

    st.markdown(
        f"""
        <div class="ww-card">
          <div class="ww-title">{h(title)}</div>
          <div class="ww-meta">{h('北京时间 ' + occurred_bj)}</div>
          <div class="ww-chips">{chips_html}</div>
          <div class="ww-body">{h(summary)}</div>
          {links}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_milestone_timeline(conn: sqlite3.Connection, conflict: str, df_ms: pd.DataFrame) -> None:
    if df_ms.empty:
        st.info("该冲突尚未录入里程碑。")
        return
    df_ms = df_ms.copy()
    df_ms["occurred_bj_dt"] = pd.to_datetime(df_ms["occurred_bj"], format="%Y-%m-%d %H:%M:%S%z", errors="coerce")
    df_ms = df_ms.sort_values("occurred_bj_dt", ascending=False)
    df_ms["day"] = df_ms["occurred_bj_dt"].dt.date.astype(str)

    for day, g in df_ms.groupby("day", sort=False):
        st.markdown(f"<div class=\"ww-day\">{day}</div>", unsafe_allow_html=True)
        for _, r in g.iterrows():
            render_milestone_card(r)


def render_dashboard(df: pd.DataFrame) -> None:
    st.title("战争进展采集分析平台")
    st.caption("本平台包含三部分：每日要闻（中文要点）、里程碑时间轴（梳理战争全过程）、官方立场文件库（正式文件入口与要点摘要）。")

    events_df = df.copy() if (df is not None and not df.empty) else pd.DataFrame()
    if not events_df.empty and "published_bj" in events_df.columns:
        events_df["published_bj_dt"] = pd.to_datetime(events_df["published_bj"], format="%Y-%m-%d %H:%M:%S%z", errors="coerce")

    tab_overview, tab_timeline, tab_docs, tab_ingest = st.tabs(["每日要闻", "里程碑时间轴", "官方立场文件库", "实时新闻采集（白名单）"])

    with tab_overview:
        st.subheader("每日要闻（中文）")
        st.caption("说明：每日要闻来自白名单媒体的实时采集；可点击按钮生成更详细的中文要点摘要（不编造不可核实细节）。")

        if events_df.empty:
            st.info("暂无实时采集事件。可在侧边栏开启‘自动更新实时采集’，或点击‘立即更新数据’获取最新条目。")
        else:
            available_conflicts = sorted([c for c in events_df["conflict"].dropna().unique().tolist()])
            default_conflicts = [c for c in ["俄乌战争", "新一轮巴以冲突"] if c in available_conflicts] or available_conflicts
            conflict_one = st.selectbox("选择议题", default_conflicts, index=0, key="daily_conflict")
            scoped = events_df[events_df["conflict"] == conflict_one].copy()
            scoped = scoped.dropna(subset=["published_bj_dt"])

            if scoped.empty:
                st.info("该议题暂无采集事件。")
            else:
                min_date = scoped["published_bj_dt"].dt.date.min()
                max_date = scoped["published_bj_dt"].dt.date.max()
                col_a, col_b, col_c = st.columns([2, 2, 2])
                with col_a:
                    selected_date = st.date_input("选择日期（北京时间）", value=max_date, min_value=min_date, max_value=max_date)
                with col_b:
                    importance_level = st.selectbox("筛选强度", ["只看重大", "较重要", "全部"], index=1)
                with col_c:
                    max_items = st.slider("最多展示条数", min_value=10, max_value=60, value=25, step=5)

                thresh = 90 if importance_level.startswith("只看重大") else (75 if importance_level.startswith("较") else 30)
                day_str = selected_date.strftime("%Y-%m-%d")
                day_df = scoped[scoped["published_bj"].astype(str).str.startswith(day_str)].copy()
                day_df = day_df[day_df["priority_score"] >= thresh].sort_values("priority_score", ascending=False).head(int(max_items))

                if day_df.empty:
                    st.info("该日期在当前筛选条件下暂无要闻。")
                else:
                    conn = get_conn()
                    if st.button("生成/补全中文要点摘要（AI）", key="ai_daily"):
                        if not _has_openai_key():
                            st.warning("请先在侧边栏填写 OpenAI API Key。")
                        else:
                            done = ensure_ai_enrichment(conn, day_df, int(min(20, len(day_df))))
                            st.success(f"已生成/补全 {done} 条。")
                    render_timeline_like_mfa(conn, day_df, conflict_one)
                    conn.close()

    with tab_timeline:
        st.subheader("里程碑时间轴（用于梳理战争阶段与进展）")
        st.caption("说明：里程碑用于系统性梳理战争全过程，不等同于新闻滚动。建议2022-2025每年至少40条；2026按重要进展实时补充。")

        conn = get_conn()
        conflict_one = st.selectbox("选择冲突", ["俄乌战争", "新一轮巴以冲突"], key="ms_conflict")
        seed_milestones(conn)
        ms = read_milestones(conn, conflict_one)

        target_years = [2022, 2023, 2024, 2025, 2026]
        counts = milestone_year_counts(ms)
        count_rows = []
        for y in target_years:
            count_rows.append({"年份": y, "当前条目数": counts.get(y, 0), "建议目标": (40 if y != 2026 else 10)})
        st.dataframe(pd.DataFrame(count_rows), use_container_width=True, hide_index=True)

        with st.expander("自动补齐年度里程碑（AI）", expanded=False):
            colx, coly = st.columns([2, 2])
            with colx:
                gen_years = st.multiselect("生成年份", target_years, default=target_years)
            with coly:
                per_year_target = st.slider("2022-2025每年目标条数", min_value=20, max_value=60, value=40, step=5)

            if st.button("按缺口自动补齐", key="ms_gen"):
                if not _has_openai_key():
                    st.warning("请先在侧边栏填写 OpenAI API Key。")
                else:
                    total_inserted = 0
                    prog = st.progress(0)
                    years_to_run = [int(y) for y in gen_years]
                    for i, y in enumerate(years_to_run, start=1):
                        counts_now = milestone_year_counts(read_milestones(conn, conflict_one))
                        have = counts_now.get(int(y), 0)
                        target = 10 if int(y) == 2026 else int(per_year_target)
                        need = max(0, target - have)
                        if need > 0:
                            items = ai_generate_milestones(conflict_one, int(y), int(min(need, 45)))
                            total_inserted += insert_milestones(conn, items)
                        prog.progress(i / max(1, len(years_to_run)))
                    st.success(f"已新增里程碑 {total_inserted} 条。")

        with st.expander("扩写现有里程碑阐释（AI）", expanded=False):
            st.caption("用于把过短的要点扩写为更完整的阐释（10-16条），便于从头到尾读懂战争进展。")
            max_expand = st.slider("本次最多扩写条数", min_value=5, max_value=50, value=20, step=5)
            if st.button("扩写当前筛选范围内的短条目", key="ms_expand"):
                if not _has_openai_key():
                    st.warning("请先在侧边栏填写 OpenAI API Key。")
                else:
                    ms_now = read_milestones(conn, conflict_one)
                    short_ids = []
                    for r in ms_now.itertuples(index=False):
                        s = str(r.summary_zh or "")
                        if s.count("\n") < 6:
                            short_ids.append(int(r.id))
                    short_ids = short_ids[: int(max_expand)]
                    if not short_ids:
                        st.info("当前没有需要扩写的短条目。")
                    else:
                        updated = expand_milestones(conn, short_ids)
                        st.success(f"已扩写 {updated} 条。")

        ms = read_milestones(conn, conflict_one)
        if ms.empty:
            st.info("该冲突尚未录入里程碑。可使用上方‘自动补齐年度里程碑（AI）’生成。")
            conn.close()
        else:
            years = sorted(list({int(y) for y in milestone_year_counts(ms).keys()}))
            year_sel = st.multiselect("展示年份", years, default=years)
            phases = sorted([p for p in ms["phase"].fillna("").unique().tolist() if p])
            cats = sorted([p for p in ms["category"].fillna("").unique().tolist() if p])
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                phase_sel = st.multiselect("阶段筛选", phases, default=phases)
            with col2:
                cat_sel = st.multiselect("类型筛选", cats, default=cats)
            with col3:
                q = st.text_input("关键词检索", value="")

            filtered = ms.copy()
            if year_sel:
                dt = pd.to_datetime(filtered["occurred_utc"], errors="coerce", utc=True)
                filtered = filtered[dt.dt.year.isin([int(y) for y in year_sel])]
            if phase_sel:
                filtered = filtered[filtered["phase"].isin(phase_sel)]
            if cat_sel:
                filtered = filtered[filtered["category"].isin(cat_sel)]
            if q.strip():
                ql = q.strip().lower()
                mask = (
                    filtered["title_zh"].fillna("").astype(str).str.lower().str.contains(ql)
                    | filtered["summary_zh"].fillna("").astype(str).str.lower().str.contains(ql)
                    | filtered["tags"].fillna("").astype(str).str.lower().str.contains(ql)
                )
                filtered = filtered[mask]
            render_milestone_timeline(conn, conflict_one, filtered)
            conn.close()

    with tab_docs:
        st.subheader("官方立场文件库")
        st.caption("说明：尽量收录各国外交部门户发布的正式立场文件/联合声明/政策汇总页。支持翻译与要点摘要生成。")

        conn = get_conn()
        seed_stance_docs(conn)
        docs_df = read_stance_docs(conn)

        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            country_filter = st.multiselect("国家/机构", sorted(docs_df["country"].unique().tolist()), default=sorted(docs_df["country"].unique().tolist()))
        with c2:
            topic_filter = st.multiselect("议题", sorted(docs_df["topic"].unique().tolist()), default=sorted(docs_df["topic"].unique().tolist()))
        with c3:
            doc_type_filter = st.multiselect("文件类型", sorted(docs_df["doc_type"].fillna("statement").unique().tolist()), default=sorted(docs_df["doc_type"].fillna("statement").unique().tolist()))

        view_df = docs_df[
            docs_df["country"].isin(country_filter)
            & docs_df["topic"].isin(topic_filter)
            & docs_df["doc_type"].fillna("statement").isin(doc_type_filter)
        ].copy()
        view_df = view_df.sort_values(["country", "topic", "id"], ascending=[True, True, False])

        col_a, col_b = st.columns([1, 2])
        with col_a:
            if st.button("生成要点摘要（AI，最多8条）", key="doc_sum"):
                if not _has_openai_key():
                    st.warning("请先在侧边栏填写 OpenAI API Key。")
                else:
                    todo = view_df[view_df["summary_zh"].fillna("") == ""].head(8)
                    if todo.empty:
                        st.info("当前筛选范围内没有需要生成摘要的条目。")
                    else:
                        prog = st.progress(0)
                        for i, r in enumerate(todo.itertuples(index=False), start=1):
                            try:
                                summarize_doc_row(conn, int(r.id))
                            except Exception:
                                pass
                            prog.progress(i / len(todo))
                        st.success("已生成要点摘要。")
        with col_b:
            if st.button("翻译未翻译条目（AI，最多8条）", key="doc_tr"):
                if not _has_openai_key():
                    st.warning("请先在侧边栏填写 OpenAI API Key。")
                else:
                    todo = view_df[(view_df["translation_zh"].fillna("") == "") & (~view_df["language"].fillna("").str.startswith("zh"))].head(8)
                    if todo.empty:
                        st.info("当前筛选范围内没有需要翻译的条目。")
                    else:
                        prog = st.progress(0)
                        for i, r in enumerate(todo.itertuples(index=False), start=1):
                            try:
                                translate_doc_row(conn, int(r.id))
                            except Exception:
                                pass
                            prog.progress(i / len(todo))
                        st.success("已完成翻译。")

        for r in view_df.head(80).itertuples(index=False):
            title = str(r.translation_zh or "").strip() or str(r.title_original or "").strip()
            meta = f"{r.country}｜{r.issuer}｜{r.topic}｜{r.doc_type}"
            chips = f"<span class=\"ww-chip\">{h(r.country)}</span><span class=\"ww-chip\">{h(r.doc_type)}</span>"
            summary = str(r.summary_zh or "").strip() or "（尚未生成要点摘要）"
            url = str(r.url or "").strip()
            link = f"<div class=\"ww-link\"><a href=\"{h(url)}\" target=\"_blank\">打开原文</a></div>" if url else ""
            st.markdown(
                f"""
                <div class="ww-card">
                  <div class="ww-title">{h(title)}</div>
                  <div class="ww-meta">{h(meta)}</div>
                  <div class="ww-chips">{chips}</div>
                  <div class="ww-body">{h(summary)}</div>
                  {link}
                </div>
                """,
                unsafe_allow_html=True,
            )
        conn.close()

    with tab_ingest:
        st.subheader("实时新闻采集（白名单）")
        st.caption("说明：这是系统从白名单媒体抓取的新闻条目，用于补充近况；不等同于里程碑梳理。")

        if events_df.empty:
            st.info("暂无采集事件。")
        else:
            available_conflicts = sorted([c for c in events_df["conflict"].dropna().unique().tolist()])
            default_conflicts = [c for c in ["俄乌战争", "新一轮巴以冲突"] if c in available_conflicts] or available_conflicts
            conflict_one = st.selectbox("选择议题", default_conflicts, index=0, key="ing_conflict")
            scoped = events_df[events_df["conflict"] == conflict_one].copy()
            scoped = scoped.dropna(subset=["published_bj_dt"])
            if scoped.empty:
                st.info("该议题暂无采集事件。")
            else:
                col_a, col_b, col_c = st.columns([2, 2, 2])
                with col_a:
                    view_days = st.slider("回看天数", min_value=7, max_value=365, value=30, step=7)
                with col_b:
                    level = st.selectbox("筛选强度", ["只看重大", "较重要", "全部"], index=0)
                with col_c:
                    max_items = st.slider("最多展示条数", min_value=20, max_value=400, value=120, step=20)

                min_score = 90 if level.startswith("只看重大") else (75 if level.startswith("较") else 30)
                cutoff = scoped["published_bj_dt"].max() - pd.Timedelta(days=view_days)
                view_df = scoped[(scoped["published_bj_dt"] >= cutoff) & (scoped["priority_score"] >= min_score)].copy()
                view_df = view_df.sort_values("published_bj_dt", ascending=False).head(int(max_items))
                conn = get_conn()
                if st.button("生成/补全中文要点摘要（AI，最多20条）", key="ai_ing"):
                    if not _has_openai_key():
                        st.warning("请先在侧边栏填写 OpenAI API Key。")
                    else:
                        done = ensure_ai_enrichment(conn, view_df, 20)
                        st.success(f"已生成/补全 {done} 条。")
                render_timeline_like_mfa(conn, view_df, conflict_one)
                conn.close()


def main() -> None:
    st.set_page_config(page_title="战争进展平台", layout="wide")
    inject_style()
    conn = get_conn()
    seed_major_events(conn)
    seed_milestones(conn)
    seed_stance_docs(conn)
    auto_update = st.sidebar.toggle("自动更新实时采集（可选）", value=False)
    auto_inserted = ensure_daily_update(conn) if auto_update else 0
    st.sidebar.metric("今日新增采集条目", auto_inserted)
    st.sidebar.markdown("---")
    st.sidebar.subheader("AI")
    st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key")
    st.sidebar.caption("仅保存在本次会话内，用于中文翻译与关键词提炼。")
    st.sidebar.subheader("实时采集")
    days_back = st.sidebar.slider("手动更新：回溯天数", min_value=7, max_value=365, value=30, step=7)
    per_feed = st.sidebar.slider("每个来源最多抓取", min_value=20, max_value=200, value=80, step=10)
    if st.sidebar.button("立即抓取更新"):
        inserted = fetch_all_sources(conn, days_back=int(days_back), per_feed=int(per_feed))
        st.sidebar.success(f"抓取完成，新增 {inserted} 条。")
    df = read_events(conn)
    render_dashboard(df)
    conn.close()


if __name__ == "__main__":
    main()
