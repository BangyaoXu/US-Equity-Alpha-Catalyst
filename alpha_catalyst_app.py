from __future__ import annotations

import re
import glob
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import feedparser
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from dateutil.parser import parse as dt_parse
from textblob import TextBlob

# =========================
# APP CONFIG
# =========================
st.set_page_config(layout="wide", page_title="US Equity Alpha & Catalyst Dashboard")

UNIVERSE_GLOB = "selected_universe_*.csv"
DEFAULT_PRICE_PERIOD = "1y"
NEWS_WINDOW_DAYS = 30
NEWS_RECENT_DAYS = 7

CACHE_TTL_PRICES = 60 * 60
CACHE_TTL_META = 60 * 60
CACHE_TTL_NEWS = 30 * 60
CACHE_TTL_WEBPAGE = 30 * 60

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"

# =========================
# UTILITIES
# =========================
def parse_universe_date_from_filename(fn: str) -> Optional[str]:
    m = re.search(r"selected_universe_(\d{4}-\d{2}-\d{2})\.csv$", fn)
    return m.group(1) if m else None

def pick_universe_files() -> List[Tuple[str, str]]:
    files = sorted(glob.glob(UNIVERSE_GLOB))
    out = []
    for f in files:
        d = parse_universe_date_from_filename(Path(f).name)
        if d:
            out.append((d, f))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def sentiment_polarity(text: str) -> float:
    try:
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def kdj(high: pd.Series, low: pd.Series, close: pd.Series, n=9, k=3, d=3):
    ln = low.rolling(n).min()
    hn = high.rolling(n).max()
    rsv = (close - ln) / (hn - ln) * 100
    rsv = rsv.replace([np.inf, -np.inf], np.nan).ffill()
    k_line = rsv.ewm(span=k, adjust=False).mean()
    d_line = k_line.ewm(span=d, adjust=False).mean()
    j_line = 3 * k_line - 2 * d_line
    return k_line, d_line, j_line

# RSI series (not just last point)
def rsi_series(close: pd.Series, window=14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def normalize_universe_keep_original(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    lower_map = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in lower_map:
                return lower_map[n.lower()]
        return None

    company_col = pick("Company Name", "Company", "Name", "company_name")
    ticker_col = pick("Ticker", "ticker", "Symbol", "symbol")
    sector_col = pick("Sector", "sector")
    industry_col = pick("Industry", "industry")

    if ticker_col is None:
        raise ValueError("Universe CSV must have a ticker column (Ticker/ticker/Symbol/symbol).")

    df["ticker_norm"] = df[ticker_col].astype(str).str.upper().str.strip()
    df["company_norm"] = df[company_col].astype(str) if company_col else df["ticker_norm"]
    df["sector_norm"] = df[sector_col].astype(str) if sector_col else "Unknown"
    df["industry_norm"] = df[industry_col].astype(str) if industry_col else ""

    want = ["Company Name", "Ticker", "Industry", "Sector", "PE1", "PE2", "EG1", "EG2", "PEG1", "PEG2"]
    present = []
    for w in want:
        c = pick(w)
        if c and c not in present:
            present.append(c)

    return df, present

# =========================
# DATA FETCH
# =========================
@st.cache_data(ttl=CACHE_TTL_PRICES)
def fetch_prices_panel(tickers: List[str], period: str = DEFAULT_PRICE_PERIOD) -> pd.DataFrame:
    px = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    return px

@st.cache_data(ttl=CACHE_TTL_META)
def fetch_info_one(ticker: str) -> Dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=CACHE_TTL_NEWS)
def fetch_google_news_rss(ticker: str, days: int = NEWS_WINDOW_DAYS) -> pd.DataFrame:
    q = f"{ticker} stock"
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        rows = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        for e in feed.entries:
            pub = None
            if "published" in e:
                try:
                    pub = dt_parse(e.published)
                    if pub.tzinfo is None:
                        pub = pub.replace(tzinfo=timezone.utc)
                    else:
                        pub = pub.astimezone(timezone.utc)
                except Exception:
                    pub = None
            if pub and pub < cutoff:
                continue

            title = getattr(e, "title", "")
            link = getattr(e, "link", "")
            source = ""
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                if len(parts) == 2:
                    title, source = parts[0], parts[1]
            rows.append({"time": pub, "title": title, "link": link, "source": source})
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.dropna(subset=["title"]).sort_values("time", ascending=False)
        return df
    except Exception:
        return pd.DataFrame()

# =========================
# WEBPAGE FETCH + SUMMARIZE (best-effort, free)
# =========================
@st.cache_data(ttl=CACHE_TTL_WEBPAGE)
def fetch_html(url: str, timeout=10) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        ct = r.headers.get("Content-Type", "")
        if 200 <= r.status_code < 300 and "text/html" in ct:
            return r.text
        return None
    except Exception:
        return None

def extract_main_text(html: str, max_chars: int = 20000) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = "\n".join(lines)
    return text[:max_chars]

def summarize_text_simple(text: str, max_sentences: int = 5) -> str:
    if not text or len(text) < 200:
        return ""
    sents = re.split(r"(?<=[\.\!\?])\s+", text)
    sents = [s.strip() for s in sents if len(s.strip()) > 40]
    if not sents:
        return ""
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    if not words:
        return ""
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    scores = []
    for i, s in enumerate(sents[:200]):
        ws = re.findall(r"[A-Za-z]{3,}", s.lower())
        score = sum(freq.get(w, 0) for w in ws) / (1 + len(ws))
        scores.append((score, i, s))
    scores.sort(reverse=True, key=lambda x: x[0])
    chosen = sorted(scores[:max_sentences], key=lambda x: x[1])
    return " ".join([c[2] for c in chosen])

def summarize_url(url: str) -> str:
    html = fetch_html(url)
    if not html:
        return ""
    text = extract_main_text(html)
    return summarize_text_simple(text, max_sentences=5)

# =========================
# IR DISCOVERY (heuristic)
# =========================
def normalize_base_url(website: str) -> Optional[str]:
    if not website or not isinstance(website, str):
        return None
    website = website.strip()
    if not website:
        return None
    if not website.startswith("http"):
        website = "https://" + website
    return website.rstrip("/")

def try_urls_exist(urls: List[str]) -> Optional[str]:
    for u in urls:
        html = fetch_html(u)
        if html and len(html) > 2000:
            return u
    return None

def find_investor_relations_url(website: str) -> Optional[str]:
    base = normalize_base_url(website)
    if not base:
        return None
    candidates = [
        f"{base}/investors",
        f"{base}/investor-relations",
        f"{base}/investors-relations",
        f"{base}/investor",
        f"{base}/news",
        f"{base}/press-releases",
        f"{base}/press",
    ]
    return try_urls_exist(candidates)

def extract_ir_news_links(ir_url: str, max_links=8) -> List[Tuple[str, str]]:
    html = fetch_html(ir_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        txt = (a.get_text(" ", strip=True) or "").strip()
        if not txt or len(txt) < 8:
            continue
        key = (txt + " " + href).lower()
        if any(k in key for k in ["press", "release", "news", "investor", "results", "earnings", "quarter"]):
            if href.startswith("/"):
                base = normalize_base_url(ir_url)
                href = base + href if base else href
            if href.startswith("http"):
                links.append((txt, href))
    seen = set()
    out = []
    for t, h in links:
        if h in seen:
            continue
        seen.add(h)
        out.append((t, h))
        if len(out) >= max_links:
            break
    return out

# =========================
# UI
# =========================
st.title("US Equity Alpha & Catalyst Dashboard")

files = pick_universe_files()
if not files:
    st.error(f"No files found matching {UNIVERSE_GLOB} in this folder.")
    st.stop()

date_options = [d for d, _ in files]
file_map = {d: f for d, f in files}

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    date_sel = st.selectbox("Universe date", date_options, index=0)
with c2:
    max_stocks = st.number_input("Max tickers to load", min_value=10, max_value=3000, value=300, step=10)
with c3:
    refresh = st.button("Refresh caches")

if refresh:
    st.cache_data.clear()
    st.success("Caches cleared. Re-run will refetch.")

universe_path = file_map[date_sel]
st.caption(f"Using: `{Path(universe_path).name}`")

uni_raw = pd.read_csv(universe_path)
uni, key_cols_present = normalize_universe_keep_original(uni_raw)

# Sidebar filters
st.sidebar.header("Filters")
sector_list = sorted(uni["sector_norm"].dropna().unique().tolist())
sector_sel = st.sidebar.multiselect("Sector", sector_list, default=sector_list)
uni_f = uni[uni["sector_norm"].isin(sector_sel)].copy()

tickers = uni_f["ticker_norm"].dropna().unique().tolist()[: int(max_stocks)]

# =========================
# 1) Universe key columns table
# =========================
st.subheader("Universe (Your CSV Key Columns)")
if key_cols_present:
    st.dataframe(uni_f[key_cols_present].copy(), use_container_width=True, height=320)
else:
    st.info("Preferred key columns not found. Showing the first 12 columns instead.")
    st.dataframe(uni_f.iloc[:, :12], use_container_width=True, height=320)

# =========================
# 2) Deep dive: indicators + catalysts ONLY
# =========================
st.subheader("Deep Dive: Buy/Sell Indicators & Catalysts")

ticker_sel = st.selectbox(
    "Select ticker",
    sorted(tickers),
    index=0 if tickers else None
)

if not ticker_sel:
    st.stop()

with st.spinner("Fetching price history…"):
    px_panel = fetch_prices_panel([ticker_sel], period=DEFAULT_PRICE_PERIOD)

info = fetch_info_one(ticker_sel)
df_t = px_panel[ticker_sel].copy() if isinstance(px_panel.columns, pd.MultiIndex) and ticker_sel in px_panel.columns.levels[0] else pd.DataFrame()

if df_t.empty or "Close" not in df_t.columns:
    st.warning("No price data for selected ticker.")
    st.stop()

df_t = df_t.dropna(subset=["Close"])
close = df_t["Close"]
high = df_t["High"] if "High" in df_t.columns else close
low = df_t["Low"] if "Low" in df_t.columns else close

# Technicals
ema12 = ema(close, 12)
ema26 = ema(close, 26)
ema50 = ema(close, 50)
ema200 = ema(close, 200)

rsi14 = rsi_series(close, 14)
macd_line, signal_line, hist = macd(close, 12, 26, 9)
k_line, d_line, j_line = kdj(high, low, close, n=9, k=3, d=3)

# --- Price + EMAs
price_df = pd.DataFrame({
    "Date": close.index,
    "Close": close.values,
    "EMA12": ema12.values,
    "EMA26": ema26.values,
    "EMA50": ema50.values,
    "EMA200": ema200.values,
}).dropna()

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df["Close"], name="Close"))
fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df["EMA12"], name="EMA12"))
fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df["EMA26"], name="EMA26"))
fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df["EMA50"], name="EMA50"))
fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df["EMA200"], name="EMA200"))
fig_price.update_layout(title=f"{ticker_sel} — Price & EMAs", height=420, legend=dict(orientation="h"))
st.plotly_chart(fig_price, use_container_width=True)

# --- RSI
rsi_df = pd.DataFrame({"Date": rsi14.index, "RSI14": rsi14.values}).dropna()
fig_rsi = px.line(rsi_df, x="Date", y="RSI14", title="RSI(14)")
fig_rsi.update_layout(height=260)
st.plotly_chart(fig_rsi, use_container_width=True)

# --- MACD
macd_df = pd.DataFrame({
    "Date": close.index,
    "MACD": macd_line.values,
    "Signal": signal_line.values,
    "Hist": hist.values,
}).dropna()

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=macd_df["Date"], y=macd_df["MACD"], name="MACD"))
fig_macd.add_trace(go.Scatter(x=macd_df["Date"], y=macd_df["Signal"], name="Signal"))
fig_macd.add_trace(go.Bar(x=macd_df["Date"], y=macd_df["Hist"], name="Hist"))
fig_macd.update_layout(title="MACD(12,26,9)", height=320, legend=dict(orientation="h"))
st.plotly_chart(fig_macd, use_container_width=True)

# --- KDJ
kdj_df = pd.DataFrame({
    "Date": close.index,
    "K": k_line.values,
    "D": d_line.values,
    "J": j_line.values,
}).dropna()

fig_kdj = px.line(kdj_df, x="Date", y=["K", "D", "J"], title="KDJ(9,3,3)")
fig_kdj.update_layout(height=320)
st.plotly_chart(fig_kdj, use_container_width=True)

# =========================
# Catalysts: News + summaries
# =========================
st.markdown("### Catalysts: Recent News")

news_df = fetch_google_news_rss(ticker_sel, days=NEWS_WINDOW_DAYS)
summarize = st.checkbox("Try summarizing linked articles (best-effort; can be slow / blocked)", value=False)

if news_df is None or news_df.empty:
    st.write("No recent RSS news found.")
else:
    news_df["time"] = pd.to_datetime(news_df["time"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=NEWS_RECENT_DAYS)
    recent = news_df[news_df["time"] >= cutoff].copy()
    if recent.empty:
        recent = news_df.head(12).copy()
    else:
        recent = recent.head(12)

    for _, n in recent.iterrows():
        t = n.get("time")
        t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
        title = str(n.get("title", ""))
        link = str(n.get("link", ""))
        src = str(n.get("source", ""))
        pol = sentiment_polarity(title)

        st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_ | headline polarity: `{pol:+.2f}`")

        if summarize and link.startswith("http"):
            with st.spinner("Fetching & summarizing…"):
                summ = summarize_url(link)
            if summ:
                st.caption(summ)
            else:
                st.caption("Summary unavailable (blocked / non-HTML / parsing failed).")

# =========================
# Catalysts: Investor Relations
# =========================
st.markdown("### Catalysts: Investor Relations (Official Site)")

website = info.get("website", "")
base = normalize_base_url(website)
if not base:
    st.write("Official website not available via yfinance for this ticker.")
else:
    st.write(f"Official website (from yfinance): {base}")

    ir_url = find_investor_relations_url(base)
    if not ir_url:
        st.write("Could not auto-locate an Investor Relations page (heuristic).")
        st.write("Try manually: usually `/investors` or `/investor-relations` on the official domain.")
    else:
        st.write(f"Detected IR / News page: {ir_url}")

        links = extract_ir_news_links(ir_url, max_links=8)
        if not links:
            st.write("No obvious IR/news links found on that page (site may be JS-rendered).")
        else:
            st.markdown("**Latest IR / Press / News links (best-effort extraction):**")
            summarize_ir = st.checkbox("Try summarizing IR links too (best-effort)", value=False)
            for txt, href in links:
                st.markdown(f"- [{txt}]({href})")
                if summarize_ir:
                    with st.spinner("Fetching & summarizing…"):
                        summ = summarize_url(href)
                    if summ:
                        st.caption(summ)
                    else:
                        st.caption("Summary unavailable (blocked / JS-rendered / parsing failed).")
