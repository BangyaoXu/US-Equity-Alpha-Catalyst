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

def rsi_series(close: pd.Series, window=14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def bollinger(close: pd.Series, window=20, num_std=2.0):
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    return mid, upper, lower

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr

def adx(high: pd.Series, low: pd.Series, close: pd.Series, n=14):
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1 / n, adjust=False).mean() / atr

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx_line = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx_line, plus_di, minus_di

def clamp_score(x: float, lo=-1.0, hi=1.0) -> float:
    if pd.isna(x):
        return 0.0
    return float(max(lo, min(hi, x)))

def score_to_label(score: float) -> str:
    if score >= 0.5:
        return "Buy"
    if score <= -0.5:
        return "Sell"
    return "Neutral"

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
def fetch_google_news_rss(ticker: str, days: int = 30) -> pd.DataFrame:
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
# IR DISCOVERY (heuristic)
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

# NOTE: removed the left sidebar filter entirely
uni_f = uni.copy()

tickers = uni_f["ticker_norm"].dropna().unique().tolist()[: int(max_stocks)]

# =========================
# 1) Opportunity Set
# =========================
st.subheader("Opportunity Set")
if key_cols_present:
    st.dataframe(uni_f[key_cols_present].copy(), use_container_width=True, height=320)
else:
    st.info("Preferred key columns not found. Showing the first 12 columns instead.")
    st.dataframe(uni_f.iloc[:, :12], use_container_width=True, height=320)

# =========================
# 2) Technical Analysis
# =========================
st.subheader("Technical Analysis")

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

EMA_LIST = [5, 10, 20, 60, 120, 250]
emas = {n: ema(close, n) for n in EMA_LIST}

rsi14 = rsi_series(close, 14)
macd_line, signal_line, hist = macd(close, 12, 26, 9)
k_line, d_line, j_line = kdj(high, low, close, n=9, k=3, d=3)
bb_mid, bb_up, bb_low = bollinger(close, window=20, num_std=2.0)
adx_line, plus_di, minus_di = adx(high, low, close, n=14)

# --- Price + EMAs
price_df = pd.DataFrame({"Date": close.index, "Close": close.values})
for n in EMA_LIST:
    price_df[f"EMA{n}"] = emas[n].values
price_df = price_df.dropna()

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df["Close"], name="Close"))
for n in EMA_LIST:
    fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df[f"EMA{n}"], name=f"EMA{n}"))
fig_price.update_layout(
    title=f"{ticker_sel} — Price & EMAs (5/10/20/60/120/250)",
    height=440,
    legend=dict(orientation="h")
)
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

# --- Bollinger Bands
bb_df = pd.DataFrame({
    "Date": close.index,
    "Close": close.values,
    "BB_Mid": bb_mid.values,
    "BB_Up": bb_up.values,
    "BB_Low": bb_low.values,
}).dropna()

fig_bb = go.Figure()
fig_bb.add_trace(go.Scatter(x=bb_df["Date"], y=bb_df["Close"], name="Close"))
fig_bb.add_trace(go.Scatter(x=bb_df["Date"], y=bb_df["BB_Mid"], name="BB Mid"))
fig_bb.add_trace(go.Scatter(x=bb_df["Date"], y=bb_df["BB_Up"], name="BB Upper"))
fig_bb.add_trace(go.Scatter(x=bb_df["Date"], y=bb_df["BB_Low"], name="BB Lower"))
fig_bb.update_layout(title="Bollinger Bands (20, 2)", height=320, legend=dict(orientation="h"))
st.plotly_chart(fig_bb, use_container_width=True)

# --- ADX + DI
adx_df = pd.DataFrame({
    "Date": close.index,
    "ADX": adx_line.values,
    "DI+": plus_di.values,
    "DI-": minus_di.values,
}).dropna()

fig_adx = px.line(adx_df, x="Date", y=["ADX", "DI+", "DI-"], title="ADX(14) + DI+/DI-")
fig_adx.update_layout(height=320)
st.plotly_chart(fig_adx, use_container_width=True)

# =========================
# Indicator Scoring Table (below charts)
# =========================
def _last(s: pd.Series) -> float:
    s2 = s.dropna()
    return float(s2.iloc[-1]) if len(s2) else np.nan

last_close = float(close.iloc[-1])
ema_vals = {n: _last(emas[n]) for n in EMA_LIST}

stack_ok = (all(pd.notna(ema_vals[n]) for n in EMA_LIST) and
            (ema_vals[5] > ema_vals[10] > ema_vals[20] > ema_vals[60] > ema_vals[120] > ema_vals[250]))
stack_bad = (all(pd.notna(ema_vals[n]) for n in EMA_LIST) and
             (ema_vals[5] < ema_vals[10] < ema_vals[20] < ema_vals[60] < ema_vals[120] < ema_vals[250]))
ema_stack_score = 1.0 if stack_ok else (-1.0 if stack_bad else 0.0)

ema_cross_score = 0.0
if pd.notna(ema_vals[5]) and pd.notna(ema_vals[20]):
    ema_cross_score = 1.0 if ema_vals[5] > ema_vals[20] else -1.0

rsi_last = _last(rsi14)
rsi_score = 0.0
if pd.notna(rsi_last):
    if rsi_last <= 30:
        rsi_score = 1.0
    elif rsi_last >= 70:
        rsi_score = -1.0
    else:
        rsi_score = clamp_score((50 - rsi_last) / 20.0)

macd_last = _last(macd_line)
sig_last = _last(signal_line)
hist_last = _last(hist)
macd_score = 0.0
if pd.notna(macd_last) and pd.notna(sig_last):
    direction = 1.0 if macd_last > sig_last else -1.0
    denom = (abs(macd_last) + 1e-9)
    macd_score = clamp_score(direction * min(1.0, abs(hist_last) / denom))

k_last, d_last = _last(k_line), _last(d_line)
kdj_score = 0.0
if pd.notna(k_last) and pd.notna(d_last):
    base = 1.0 if k_last > d_last else -1.0
    if k_last < 20 and base > 0:
        kdj_score = 1.0
    elif k_last > 80 and base < 0:
        kdj_score = -1.0
    else:
        kdj_score = 0.5 * base

bb_up_last, bb_low_last = _last(bb_up), _last(bb_low)
bb_score = 0.0
if pd.notna(bb_up_last) and pd.notna(bb_low_last):
    if last_close < bb_low_last:
        bb_score = 1.0
    elif last_close > bb_up_last:
        bb_score = -1.0
    else:
        bb_score = 0.0

adx_last = _last(adx_line)
pdi_last, mdi_last = _last(plus_di), _last(minus_di)
adx_score = 0.0
if pd.notna(adx_last) and pd.notna(pdi_last) and pd.notna(mdi_last):
    if adx_last >= 25:
        adx_score = 1.0 if pdi_last > mdi_last else -1.0
    else:
        adx_score = 0.0

scores = [
    ("EMA Stack (5>10>20>60>120>250)", ema_stack_score),
    ("EMA Cross (5 vs 20)", ema_cross_score),
    ("RSI(14)", rsi_score),
    ("MACD(12,26,9)", macd_score),
    ("KDJ(9,3,3)", kdj_score),
    ("Bollinger(20,2)", bb_score),
    ("ADX(14) + DI", adx_score),
]
composite = float(np.mean([s for _, s in scores])) if scores else 0.0

st.markdown("### Indicator Scores (Buy/Sell)")
score_table = pd.DataFrame(
    [{"Indicator": name, "Score (-1..+1)": float(sc), "Signal": score_to_label(float(sc))} for name, sc in scores]
)
score_table = pd.concat(
    [score_table, pd.DataFrame([{
        "Indicator": "Composite (mean of above)",
        "Score (-1..+1)": composite,
        "Signal": score_to_label(composite),
    }])],
    ignore_index=True
)

st.dataframe(
    score_table.style.format({"Score (-1..+1)": "{:.2f}"}),
    use_container_width=True
)

# =========================
# News Sentiment (no summarization)
# =========================
st.markdown("### News Sentiment")

window_label = st.selectbox("News window", ["1w", "2w", "1m", "3m"], index=0)
window_days = {"1w": 7, "2w": 14, "1m": 30, "3m": 90}[window_label]

news_df = fetch_google_news_rss(ticker_sel, days=window_days)

if news_df is None or news_df.empty:
    st.write("No recent RSS news found.")
else:
    news_df["time"] = pd.to_datetime(news_df["time"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=window_days)
    recent = news_df[news_df["time"] >= cutoff].copy().head(30)

    pols = [sentiment_polarity(str(t)) for t in recent["title"].tolist()]
    avg_pol = float(np.mean(pols)) if pols else 0.0

    n1, n2 = st.columns(2)
    n1.metric("Headline count", f"{len(recent)}")
    n2.metric("Avg headline polarity (TextBlob)", f"{avg_pol:+.2f}")

    for _, n in recent.iterrows():
        t = n.get("time")
        t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
        title = str(n.get("title", ""))
        link = str(n.get("link", ""))
        src = str(n.get("source", ""))
        pol = sentiment_polarity(title)
        st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_ | polarity: `{pol:+.2f}`")

# =========================
# Investor Relations (no summarization)
# =========================
st.markdown("### Investor Relations")

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
            for txt, href in links:
                st.markdown(f"- [{txt}]({href})")
