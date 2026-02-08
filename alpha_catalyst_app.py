from __future__ import annotations

import re
import glob
import math
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
st.set_page_config(layout="wide", page_title="US Equity Performance & Catalyst Dashboard")

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

def _safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).replace(",", "").strip()
        return float(s)
    except Exception:
        return np.nan

def sentiment_polarity(text: str) -> float:
    try:
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0

def compute_rsi(close: pd.Series, window=14) -> float:
    close = close.dropna()
    if len(close) < window + 5:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def kdj(high: pd.Series, low: pd.Series, close: pd.Series, n=9, k=3, d=3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    KDJ from stochastic RSV:
      RSV = (C - Ln) / (Hn - Ln) * 100
      K = EMA(RSV, k) (commonly SMA; EMA works and is stable)
      D = EMA(K, d)
      J = 3K - 2D
    """
    ln = low.rolling(n).min()
    hn = high.rolling(n).max()
    rsv = (close - ln) / (hn - ln) * 100
    rsv = rsv.replace([np.inf, -np.inf], np.nan).fillna(method="ffill")
    k_line = rsv.ewm(span=k, adjust=False).mean()
    d_line = k_line.ewm(span=d, adjust=False).mean()
    j_line = 3 * k_line - 2 * d_line
    return k_line, d_line, j_line

def compute_returns(close: pd.Series) -> Dict[str, float]:
    close = close.dropna()
    out = {"ret_1d": np.nan, "ret_5d": np.nan, "ret_1m": np.nan, "ret_3m": np.nan, "ret_6m": np.nan}
    if len(close) < 10:
        return out
    out["ret_1d"] = close.iloc[-1] / close.iloc[-2] - 1
    if len(close) >= 6:
        out["ret_5d"] = close.iloc[-1] / close.iloc[-6] - 1
    if len(close) >= 21:
        out["ret_1m"] = close.iloc[-1] / close.iloc[-21] - 1
    if len(close) >= 63:
        out["ret_3m"] = close.iloc[-1] / close.iloc[-63] - 1
    if len(close) >= 126:
        out["ret_6m"] = close.iloc[-1] / close.iloc[-126] - 1
    return out

def realized_vol(close: pd.Series, window=20) -> float:
    close = close.dropna()
    if len(close) < window + 5:
        return np.nan
    r = close.pct_change()
    return float(r.rolling(window).std().iloc[-1] * np.sqrt(252))

def volume_surge(vol: pd.Series, window=20) -> float:
    vol = vol.dropna()
    if len(vol) < window + 5:
        return np.nan
    v_now = vol.iloc[-1]
    v_avg = vol.rolling(window).mean().iloc[-1]
    if v_avg == 0 or np.isnan(v_avg):
        return np.nan
    return float(v_now / v_avg)

def normalize_universe_keep_original(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Keep original columns; only add normalized helper columns:
      - ticker_norm
      - company_norm
      - sector_norm
      - industry_norm
    Also returns a list of 'preferred original key columns' present.
    """

    df = df.copy()
    lower_map = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in lower_map:
                return lower_map[n.lower()]
        return None

    # Your preferred schema
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

    # key columns list (only those present)
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
        if r.status_code >= 200 and r.status_code < 300 and "text/html" in r.headers.get("Content-Type", ""):
            return r.text
        return None
    except Exception:
        return None

def extract_main_text(html: str, max_chars: int = 20000) -> str:
    soup = BeautifulSoup(html, "lxml")
    # remove junk
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = "\n".join(lines)
    return text[:max_chars]

def summarize_text_simple(text: str, max_sentences: int = 5) -> str:
    """
    Lightweight extractive summarizer (no paid NLP):
    - split into sentences
    - score sentences by word frequency excluding very short tokens
    """
    if not text or len(text) < 200:
        return ""

    # crude sentence split
    sents = re.split(r"(?<=[\.\!\?])\s+", text)
    sents = [s.strip() for s in sents if len(s.strip()) > 40]
    if not sents:
        return ""

    # word freq
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    if not words:
        return ""
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # score each sentence
    scores = []
    for i, s in enumerate(sents[:200]):  # limit
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
        f"{base}/investors-and-media",
        f"{base}/news",
        f"{base}/press-releases",
        f"{base}/press",
    ]
    found = try_urls_exist(candidates)
    return found

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
        # filter to likely news items
        key = (txt + " " + href).lower()
        if any(k in key for k in ["press", "release", "news", "investor", "results", "earnings", "quarter"]):
            # make absolute
            if href.startswith("/"):
                base = normalize_base_url(ir_url)
                href = base + href if base else href
            if href.startswith("http"):
                links.append((txt, href))
    # de-dup
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
# BUILD RAW PERFORMANCE TABLE
# =========================
def build_raw_performance(universe: pd.DataFrame, prices_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, u in universe.iterrows():
        ticker = u["ticker_norm"]
        if isinstance(prices_panel.columns, pd.MultiIndex) and ticker in prices_panel.columns.levels[0]:
            df = prices_panel[ticker]
        else:
            continue

        close = df["Close"].dropna() if "Close" in df.columns else pd.Series(dtype=float)
        if close.empty:
            continue

        vol = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)
        high = df["High"].dropna() if "High" in df.columns else pd.Series(dtype=float)
        low = df["Low"].dropna() if "Low" in df.columns else pd.Series(dtype=float)

        rets = compute_returns(close)
        v20 = realized_vol(close, 20)
        v60 = realized_vol(close, 60)
        vsurge = volume_surge(vol, 20) if not vol.empty else np.nan
        rsi14 = compute_rsi(close, 14)

        ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
        dist50 = (close.iloc[-1] / ma50 - 1) if not pd.isna(ma50) else np.nan
        dist200 = (close.iloc[-1] / ma200 - 1) if not pd.isna(ma200) else np.nan

        # minimal fundamentals from yfinance
        info = fetch_info_one(ticker)
        mkt_cap = _safe_float(info.get("marketCap"))
        pe = _safe_float(info.get("trailingPE"))
        fwd_pe = _safe_float(info.get("forwardPE"))
        beta = _safe_float(info.get("beta"))

        rows.append({
            "Ticker": ticker,
            "Company": u.get("company_norm", ticker),
            "Sector": u.get("sector_norm", ""),
            "Industry": u.get("industry_norm", ""),
            "Price": float(close.iloc[-1]),
            "Ret 1D": rets["ret_1d"],
            "Ret 1W": rets["ret_5d"],
            "Ret 1M": rets["ret_1m"],
            "Ret 3M": rets["ret_3m"],
            "Ret 6M": rets["ret_6m"],
            "Vol 20D": v20,
            "Vol 60D": v60,
            "RSI 14": rsi14,
            "VolSurge 20D": vsurge,
            "Dist 50DMA": dist50,
            "Dist 200DMA": dist200,
            "MktCap": mkt_cap,
            "PE (yf)": pe,
            "Fwd PE (yf)": fwd_pe,
            "Beta (yf)": beta,
        })
    return pd.DataFrame(rows)

# =========================
# UI
# =========================
st.title("📊 Stock Performance, Technicals, News & IR Monitor (Raw)")

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

# Sidebar filters (Zacks sector = your Sector column)
st.sidebar.header("Filters")
sector_list = sorted(uni["sector_norm"].dropna().unique().tolist())
sector_sel = st.sidebar.multiselect("Sector", sector_list, default=sector_list)
uni_f = uni[uni["sector_norm"].isin(sector_sel)].copy()

# Ticker cap
tickers = uni_f["ticker_norm"].dropna().unique().tolist()[: int(max_stocks)]

# -------------------------
# (1) Your original key columns table
# -------------------------
st.subheader("Universe (Original CSV Key Columns)")
if key_cols_present:
    st.dataframe(uni_f[key_cols_present].copy(), use_container_width=True, height=320)
else:
    st.info("None of the preferred key columns were found. Showing the first 12 columns instead.")
    st.dataframe(uni_f.iloc[:, :12], use_container_width=True, height=320)

# -------------------------
# Prices download + raw perf table
# -------------------------
with st.spinner("Fetching prices…"):
    px_panel = fetch_prices_panel(tickers, period=DEFAULT_PRICE_PERIOD)

with st.spinner("Building raw performance table…"):
    perf = build_raw_performance(uni_f[uni_f["ticker_norm"].isin(tickers)], px_panel)

st.subheader("Performance & Basic Metrics (Raw, sortable)")
if perf.empty:
    st.warning("No usable tickers found (check tickers are valid for yfinance).")
    st.stop()

# Let user choose sort column
sort_col = st.selectbox("Sort by", perf.columns.tolist(), index=perf.columns.get_loc("Ret 1M") if "Ret 1M" in perf.columns else 0)
sort_asc = st.checkbox("Ascending", value=False)
perf_view = perf.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

st.dataframe(
    perf_view.style.format({
        "Price": "{:.2f}",
        "Ret 1D": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "",
        "Ret 1W": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "",
        "Ret 1M": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "",
        "Ret 3M": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "",
        "Ret 6M": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "",
        "Vol 20D": "{:.2f}",
        "Vol 60D": "{:.2f}",
        "RSI 14": "{:.1f}",
        "VolSurge 20D": "{:.2f}",
        "Dist 50DMA": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "",
        "Dist 200DMA": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "",
        "MktCap": lambda x: f"{x/1e9:.2f}B" if pd.notna(x) else "",
        "PE (yf)": "{:.2f}",
        "Fwd PE (yf)": "{:.2f}",
        "Beta (yf)": "{:.2f}",
    }),
    use_container_width=True,
    height=520
)

# -------------------------
# Sector analytics WITHOUT SPDR ETFs
# -------------------------
st.subheader("Sector Snapshot (within your universe)")
sector_agg = (
    perf.groupby("Sector")
    .agg(
        n=("Ticker", "count"),
        avg_1m=("Ret 1M", "mean"),
        avg_3m=("Ret 3M", "mean"),
        avg_vol60=("Vol 60D", "mean"),
        median_rsi=("RSI 14", "median"),
    )
    .reset_index()
    .sort_values("avg_1m", ascending=False)
)

s1, s2 = st.columns(2)
with s1:
    fig = px.bar(sector_agg, x="Sector", y="avg_1m", title="Average 1M Return by Sector (Universe)")
    st.plotly_chart(fig, use_container_width=True)
with s2:
    fig2 = px.bar(sector_agg, x="Sector", y="avg_vol60", title="Average 60D Vol by Sector (Universe)")
    st.plotly_chart(fig2, use_container_width=True)

st.dataframe(
    sector_agg.style.format({
        "avg_1m": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "",
        "avg_3m": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "",
        "avg_vol60": "{:.2f}",
        "median_rsi": "{:.1f}",
    }),
    use_container_width=True
)

# =========================
# DEEP DIVE: TECHNICALS + NEWS + IR
# =========================
st.subheader("Stock Deep Dive (Technicals + News + Investor Relations)")

ticker_sel = st.selectbox("Select ticker", perf.sort_values("Ret 1M", ascending=False)["Ticker"].tolist())
info = fetch_info_one(ticker_sel)

# Extract OHLCV
df_t = None
if isinstance(px_panel.columns, pd.MultiIndex) and ticker_sel in px_panel.columns.levels[0]:
    df_t = px_panel[ticker_sel].copy()
if df_t is None or df_t.empty:
    st.warning("No price data for selected ticker.")
    st.stop()

df_t = df_t.dropna(subset=["Close"])
close = df_t["Close"]
high = df_t["High"] if "High" in df_t.columns else close
low = df_t["Low"] if "Low" in df_t.columns else close
vol = df_t["Volume"] if "Volume" in df_t.columns else pd.Series(index=df_t.index, dtype=float)

# Compute technical indicators
ema12 = ema(close, 12)
ema26 = ema(close, 26)
ema50 = ema(close, 50)
ema200 = ema(close, 200)

rsi14 = close.rolling(1).apply(lambda _: np.nan)  # placeholder
rsi14 = pd.Series(index=close.index, dtype=float)
# compute RSI as series
delta = close.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
rsi14 = 100 - (100 / (1 + rs))

macd_line, signal_line, hist = macd(close, 12, 26, 9)
k_line, d_line, j_line = kdj(high, low, close, n=9, k=3, d=3)

# --- Price chart with EMAs
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
fig_price.update_layout(title=f"{ticker_sel} Price + EMAs", height=420, legend=dict(orientation="h"))
st.plotly_chart(fig_price, use_container_width=True)

# --- RSI chart
rsi_df = pd.DataFrame({"Date": rsi14.index, "RSI14": rsi14.values}).dropna()
fig_rsi = px.line(rsi_df, x="Date", y="RSI14", title="RSI(14)")
fig_rsi.update_layout(height=260)
st.plotly_chart(fig_rsi, use_container_width=True)

# --- MACD chart
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

# --- KDJ chart
kdj_df = pd.DataFrame({
    "Date": close.index,
    "K": k_line.values,
    "D": d_line.values,
    "J": j_line.values,
}).dropna()

fig_kdj = px.line(kdj_df, x="Date", y=["K", "D", "J"], title="KDJ(9,3,3)")
fig_kdj.update_layout(height=320)
st.plotly_chart(fig_kdj, use_container_width=True)

# --- Technical snapshot table (latest values)
snap = {
    "Close": float(close.iloc[-1]),
    "EMA12": float(ema12.iloc[-1]),
    "EMA26": float(ema26.iloc[-1]),
    "EMA50": float(ema50.iloc[-1]) if not pd.isna(ema50.iloc[-1]) else np.nan,
    "EMA200": float(ema200.iloc[-1]) if not pd.isna(ema200.iloc[-1]) else np.nan,
    "RSI14": float(rsi14.iloc[-1]) if not pd.isna(rsi14.iloc[-1]) else np.nan,
    "MACD": float(macd_line.iloc[-1]),
    "MACD Signal": float(signal_line.iloc[-1]),
    "MACD Hist": float(hist.iloc[-1]),
    "K": float(k_line.iloc[-1]) if not pd.isna(k_line.iloc[-1]) else np.nan,
    "D": float(d_line.iloc[-1]) if not pd.isna(d_line.iloc[-1]) else np.nan,
    "J": float(j_line.iloc[-1]) if not pd.isna(j_line.iloc[-1]) else np.nan,
    "Vol 60D": realized_vol(close, 60),
    "VolSurge 20D": volume_surge(vol, 20) if vol is not None and not vol.empty else np.nan,
}
st.markdown("### Latest Technical Snapshot")
st.dataframe(pd.DataFrame([snap]).T.rename(columns={0: "Value"}), use_container_width=True)

# =========================
# Recent news (RSS) + best-effort content summary
# =========================
st.markdown("### Recent Company News (Google News RSS)")

news_df = fetch_google_news_rss(ticker_sel, days=NEWS_WINDOW_DAYS)
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

    summarize = st.checkbox("Try summarizing linked articles (best-effort, can be slow / blocked)", value=False)

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
                st.caption("Summary unavailable (site blocked / non-HTML / parsing failed).")

# =========================
# Investor Relations discovery + latest updates
# =========================
st.markdown("### Investor Relations (Official Site)")

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

        # Try to surface latest IR/press links
        links = extract_ir_news_links(ir_url, max_links=8)
        if not links:
            st.write("No obvious IR/news links found on that page (site structure may be JS-rendered).")
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

# =========================
# Export
# =========================
st.subheader("Export")
csv_out = perf_view.to_csv(index=False).encode("utf-8")
st.download_button("Download performance table (CSV)", data=csv_out, file_name=f"performance_{date_sel}.csv", mime="text/csv")
