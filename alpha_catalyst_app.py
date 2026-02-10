# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import glob
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import StringIO

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
DEFAULT_PRICE_PERIOD = "2y"   # for tech indicators like EMA250
DEFAULT_PRICE_INTERVAL = "1d"

CACHE_TTL_PRICES = 60 * 60
CACHE_TTL_META = 60 * 60
CACHE_TTL_NEWS = 30 * 60
CACHE_TTL_INDICATORS = 60 * 60

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"

# =========================
# UTILITIES
# =========================
def parse_universe_date_from_filename(fn: str) -> Optional[str]:
    m = re.search(r"selected_universe_(\d{4}-\d{2}-\d{2})\.csv$", fn)
    return m.group(1) if m else None

def pick_universe_files() -> List[Tuple[str, str]]:
    files = sorted(glob.glob(UNIVERSE_GLOB))
    out: List[Tuple[str, str]] = []
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
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def safe_last(x: pd.Series) -> float:
    try:
        v = x.dropna().iloc[-1]
        return float(v)
    except Exception:
        return float("nan")

def get_ohlc_from_panel(px_panel: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    yfinance.download with group_by='ticker' returns MultiIndex columns:
      first level ticker, second level OHLCV.
    For single ticker, sometimes it returns single-index OHLCV.
    """
    if px_panel is None or getattr(px_panel, "empty", True):
        return pd.DataFrame()

    if isinstance(px_panel.columns, pd.MultiIndex):
        # exact match
        if ticker in px_panel.columns.levels[0]:
            df = px_panel[ticker].copy()
            df.columns = [str(c) for c in df.columns]
            return df
        # case-insensitive fallback
        for t in px_panel.columns.levels[0]:
            if str(t).upper() == str(ticker).upper():
                df = px_panel[t].copy()
                df.columns = [str(c) for c in df.columns]
                return df
        return pd.DataFrame()

    # single ticker frame
    df = px_panel.copy()
    df.columns = [str(c) for c in df.columns]
    return df

def normalize_universe_keep_original(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
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

    canonical = {
        "company": company_col or "",
        "ticker": ticker_col,
        "sector": sector_col or "",
        "industry": industry_col or "",
    }
    return df, present, canonical

# =========================
# FREE SECTOR-INDICATOR ENGINE (best-effort)
# =========================
def _fred_csv_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={requests.utils.quote(series_id)}"

@st.cache_data(ttl=CACHE_TTL_INDICATORS)
def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Free, no-key: FRED CSV endpoint. Defensive: never raises; returns empty on HTTP errors."""
    url = _fred_csv_url(series_id)
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/csv,*/*",
                "Referer": "https://fred.stlouisfed.org/",
            },
            timeout=20,
        )
        if not (200 <= r.status_code < 300):
            return pd.DataFrame(columns=["date", "value"])

        df = pd.read_csv(StringIO(r.text))
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "value"])

        date_col = "DATE" if "DATE" in df.columns else df.columns[0]
        val_col = series_id if series_id in df.columns else df.columns[1]

        out = df[[date_col, val_col]].copy()
        out.columns = ["date", "value"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date")
        return out
    except Exception:
        return pd.DataFrame(columns=["date", "value"])

@st.cache_data(ttl=CACHE_TTL_INDICATORS)
def fetch_yf_series(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    yfinance (best-effort). Returns columns: ['date','value'].
    If yfinance is blocked/rate-limited on Streamlit Cloud, caller should fallback (Stooq).
    """
    try:
        px = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,  # more stable on Streamlit Cloud
        )
    except Exception:
        return pd.DataFrame(columns=["date", "value"])

    if px is None or px.empty:
        return pd.DataFrame(columns=["date", "value"])

    px.columns = [str(c) for c in px.columns]
    if "Close" not in px.columns:
        for alt in ["Adj Close", "AdjClose", "close", "CLOSE"]:
            if alt in px.columns:
                px = px.rename(columns={alt: "Close"})
                break
    if "Close" not in px.columns:
        return pd.DataFrame(columns=["date", "value"])

    px = px.dropna(subset=["Close"])
    if px.empty:
        return pd.DataFrame(columns=["date", "value"])

    out = px[["Close"]].copy().reset_index()
    date_col = "Date" if "Date" in out.columns else out.columns[0]
    out = out.rename(columns={date_col: "date", "Close": "value"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date")

@st.cache_data(ttl=CACHE_TTL_INDICATORS)
def fetch_stooq_series(symbol: str) -> pd.DataFrame:
    """
    Free, no-key fallback via Stooq daily CSV.
    Stooq symbols look like: 'spy.us', 'qqq.us', 'soxx.us', 'vixy.us', 'uup.us', 'gld.us'
    Returns columns: ['date','value'].
    """
    try:
        url = f"https://stooq.com/q/d/l/?s={requests.utils.quote(symbol.lower())}&i=d"
        r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"}, timeout=20)
        if not (200 <= r.status_code < 300):
            return pd.DataFrame(columns=["date", "value"])
        df = pd.read_csv(StringIO(r.text))
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "value"])
        cols = {c.lower(): c for c in df.columns}
        date_c = cols.get("date", df.columns[0])
        close_c = cols.get("close", None)
        if close_c is None:
            return pd.DataFrame(columns=["date", "value"])
        out = df[[date_c, close_c]].copy()
        out.columns = ["date", "value"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date")
        return out
    except Exception:
        return pd.DataFrame(columns=["date", "value"])

def infer_sector_bucket(zacks_sector: str) -> str:
    s = (zacks_sector or "").strip().lower()
    if not s:
        return "General"

    rules = [
        ("health", "Healthcare"),
        ("medical", "Healthcare"),
        ("bio", "Healthcare"),

        ("tech", "Technology"),
        ("computer", "Technology"),
        ("software", "Technology"),
        ("semiconductor", "Technology"),

        ("energy", "Energy"),
        ("oil", "Energy"),
        ("gas", "Energy"),

        ("basic", "Basic Materials"),
        ("material", "Basic Materials"),
        ("metal", "Basic Materials"),
        ("mining", "Basic Materials"),
        ("chemical", "Basic Materials"),

        ("finance", "Financials"),
        ("bank", "Financials"),
        ("insurance", "Financials"),

        ("utility", "Utilities"),

        ("real estate", "Real Estate"),
        ("reit", "Real Estate"),

        ("consumer staples", "Consumer Staples"),
        ("staples", "Consumer Staples"),

        ("consumer discretionary", "Consumer Discretionary"),
        ("retail", "Consumer Discretionary"),
        ("wholesale", "Consumer Discretionary"),
        ("auto", "Consumer Discretionary"),

        ("industrial", "Industrials"),
        ("transport", "Industrials"),
        ("construction", "Industrials"),
        ("aerospace", "Industrials"),

        ("telecom", "Communication Services"),
        ("communication", "Communication Services"),
        ("media", "Communication Services"),
        ("entertain", "Communication Services"),
    ]
    for key, bucket in rules:
        if key in s:
            return bucket
    return "General"

# Indicator catalog:
# - source: "fred" or "yfinance" or "computed"
# - id: series id / symbol
# - stooq: stooq symbol for fallback where possible
# - bullish: "higher" or "lower"
INDICATORS: Dict[str, Dict] = {
    # General / cross-sector (avoid caret indices; use ETF proxies + Stooq fallback)
    "SP500": {"name": "S&P 500 (SPY)", "source": "yfinance", "id": "SPY", "stooq": "spy.us", "bullish": "higher"},
    "VIX": {"name": "VIX (VIXY proxy)", "source": "yfinance", "id": "VIXY", "stooq": "vixy.us", "bullish": "lower"},
    "DXY": {"name": "Broad USD Index (FRED)", "source": "fred", "id": "DTWEXBGS", "bullish": "lower"},
    "UST10Y": {"name": "10Y Treasury Yield", "source": "fred", "id": "DGS10", "bullish": "lower"},
    "UST2Y": {"name": "2Y Treasury Yield", "source": "fred", "id": "DGS2", "bullish": "lower"},
    "YC_10_2": {"name": "Yield Curve (10Y-2Y, bp)", "source": "computed", "id": "DGS10-DGS2", "bullish": "higher"},

    # Industrials
    "ISM_PMI": {"name": "ISM Manufacturing PMI", "source": "fred", "id": "NAPM", "bullish": "higher"},
    "INDPRO": {"name": "Industrial Production Index", "source": "fred", "id": "INDPRO", "bullish": "higher"},
    "DGORDER": {"name": "Durable Goods Orders", "source": "fred", "id": "DGORDER", "bullish": "higher"},

    # Consumer
    "UMCSENT": {"name": "U. Michigan Consumer Sentiment", "source": "fred", "id": "UMCSENT", "bullish": "higher"},
    "RSAFS": {"name": "Retail Sales (Total)", "source": "fred", "id": "RSAFS", "bullish": "higher"},
    "CPIAUCSL": {"name": "CPI (Inflation)", "source": "fred", "id": "CPIAUCSL", "bullish": "lower"},

    # Real estate
    "MORTGAGE30US": {"name": "30Y Mortgage Rate", "source": "fred", "id": "MORTGAGE30US", "bullish": "lower"},
    "HOUST": {"name": "Housing Starts", "source": "fred", "id": "HOUST", "bullish": "higher"},
    "CSUSHPINSA": {"name": "Case-Shiller Home Price Index", "source": "fred", "id": "CSUSHPINSA", "bullish": "higher"},

    # Energy / Materials (FRED commodities)
    "WTI": {"name": "WTI Spot ($/bbl) (FRED)", "source": "fred", "id": "DCOILWTICO", "bullish": "higher"},
    "BRENT": {"name": "Brent Spot ($/bbl) (FRED)", "source": "fred", "id": "DCOILBRENTEU", "bullish": "higher"},
    "NATGAS": {"name": "Nat Gas Henry Hub ($/MMBtu) (FRED)", "source": "fred", "id": "DHHNGSP", "bullish": "higher"},
    "COPPER": {"name": "Copper Global ($/mt) (FRED)", "source": "fred", "id": "PCOPPUSDM", "bullish": "higher"},
    "ALUMINUM": {"name": "Aluminum Global ($/mt) (FRED)", "source": "fred", "id": "PALUMUSDM", "bullish": "higher"},
    "GOLD": {"name": "Gold (GLD)", "source": "yfinance", "id": "GLD", "stooq": "gld.us", "bullish": "higher"},

    # Financials
    "CREDIT_SPREAD": {"name": "BBB Corporate Spread (proxy)", "source": "fred", "id": "BAA10Y", "bullish": "lower"},
    "UNRATE": {"name": "Unemployment Rate", "source": "fred", "id": "UNRATE", "bullish": "lower"},
    "FEDFUNDS": {"name": "Fed Funds Rate", "source": "fred", "id": "FEDFUNDS", "bullish": "lower"},

    # Tech / chip cycle proxies (ETF proxies instead of ^NDX/^SOX)
    "SOX": {"name": "Semis (SOXX)", "source": "yfinance", "id": "SOXX", "stooq": "soxx.us", "bullish": "higher"},
    "NDX": {"name": "Nasdaq-100 (QQQ)", "source": "yfinance", "id": "QQQ", "stooq": "qqq.us", "bullish": "higher"},
}

SECTOR_TO_INDICATORS: Dict[str, List[str]] = {
    "General": ["SP500", "VIX", "DXY", "UST10Y", "YC_10_2"],
    "Healthcare": ["SP500", "VIX", "UST10Y"],
    "Industrials": ["ISM_PMI", "INDPRO", "DGORDER", "UST10Y"],
    "Financials": ["UST10Y", "YC_10_2", "FEDFUNDS", "UNRATE", "CREDIT_SPREAD", "VIX"],
    "Consumer Discretionary": ["UMCSENT", "RSAFS", "UST10Y", "CPIAUCSL"],
    "Consumer Staples": ["CPIAUCSL", "UMCSENT", "RSAFS"],
    "Technology": ["NDX", "SOX", "UST10Y", "VIX"],
    "Basic Materials": ["COPPER", "GOLD", "WTI", "DXY", "ISM_PMI"],
    "Energy": ["WTI", "BRENT", "NATGAS", "DXY"],
    "Utilities": ["UST10Y", "YC_10_2", "CPIAUCSL"],
    "Real Estate": ["MORTGAGE30US", "HOUST", "CSUSHPINSA", "UST10Y"],
    "Communication Services": ["NDX", "VIX", "UST10Y"],
}

def compute_curve_10_2() -> pd.DataFrame:
    d10 = fetch_fred_series("DGS10")
    d2 = fetch_fred_series("DGS2")
    if d10.empty or d2.empty:
        return pd.DataFrame()
    df = pd.merge(d10, d2, on="date", how="inner", suffixes=("_10", "_2"))
    df["value"] = (df["value_10"] - df["value_2"]) * 100.0  # bp
    return df[["date", "value"]]

@st.cache_data(ttl=CACHE_TTL_INDICATORS)
def fetch_indicator_series(ind_key: str) -> pd.DataFrame:
    cfg = INDICATORS.get(ind_key, {})
    src = cfg.get("source")

    if src == "fred":
        return fetch_fred_series(cfg["id"])

    if src == "computed" and ind_key == "YC_10_2":
        return compute_curve_10_2()

    if src == "yfinance":
        ser = fetch_yf_series(cfg.get("id", ""))
        if ser is not None and not ser.empty:
            return ser
        stooq_id = cfg.get("stooq", "")
        if stooq_id:
            return fetch_stooq_series(stooq_id)
        return pd.DataFrame()

    return pd.DataFrame()

def indicator_snapshot(df: pd.DataFrame) -> Dict[str, float]:
    """latest, 1w change, 1m change, zscore(≈1y)"""
    if df is None or df.empty:
        return {"latest": np.nan, "chg_1w": np.nan, "chg_1m": np.nan, "z_1y": np.nan}

    s = df.dropna(subset=["date", "value"]).copy()
    s = s.sort_values("date")
    if s.empty:
        return {"latest": np.nan, "chg_1w": np.nan, "chg_1m": np.nan, "z_1y": np.nan}

    latest = float(s["value"].iloc[-1])
    v = s["value"].values
    chg_1w = latest - float(v[-6]) if len(v) >= 6 else np.nan
    chg_1m = latest - float(v[-22]) if len(v) >= 22 else np.nan

    tail = s.tail(260)
    mu = tail["value"].mean()
    sd = tail["value"].std(ddof=0)
    z = (latest - mu) / sd if sd and sd > 0 else np.nan
    return {"latest": latest, "chg_1w": chg_1w, "chg_1m": chg_1m, "z_1y": z}

def indicator_signal(cfg: Dict, snap: Dict[str, float]) -> Tuple[str, float]:
    bullish = cfg.get("bullish", "higher")
    chg = snap.get("chg_1m", np.nan)
    if not np.isfinite(chg):
        return ("N/A", 0.0)
    sign = 1.0 if chg > 0 else (-1.0 if chg < 0 else 0.0)
    score = sign if bullish == "higher" else -sign
    label = "Bullish" if score > 0 else ("Bearish" if score < 0 else "Neutral")
    return (label, float(score))

# =========================
# DATA FETCH
# =========================
@st.cache_data(ttl=CACHE_TTL_PRICES)
def fetch_prices_panel(tickers: List[str], period: str = DEFAULT_PRICE_PERIOD, interval: str = DEFAULT_PRICE_INTERVAL) -> pd.DataFrame:
    try:
        px_panel = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            group_by="ticker",
            threads=False,  # Cloud stability
            progress=False,
        )
        return px_panel
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL_META)
def fetch_info_one(ticker: str) -> Dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=CACHE_TTL_NEWS)
def fetch_google_news_rss(ticker: str, days: int) -> pd.DataFrame:
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
def normalize_base_url(website: str) -> Optional[str]:
    if not website or not isinstance(website, str):
        return None
    website = website.strip()
    if not website:
        return None
    if not website.startswith("http"):
        website = "https://" + website
    return website.rstrip("/")

@st.cache_data(ttl=CACHE_TTL_META)
def fetch_html(url: str, timeout=10) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        ct = r.headers.get("Content-Type", "")
        if 200 <= r.status_code < 300 and "text/html" in ct:
            return r.text
        return None
    except Exception:
        return None

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

def extract_ir_news_links(ir_url: str, max_links=10) -> List[Tuple[str, str]]:
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
        if any(k in key for k in ["press", "release", "news", "investor", "results", "earnings", "quarter", "sec"]):
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
# SEC EDGAR (latest filings) + EARNINGS CALENDAR (best-effort)
# =========================
SEC_BASE = "https://data.sec.gov"
SEC_TICKER_MAP_URLS = [
    "https://www.sec.gov/files/company_tickers.json",
    "https://www.sec.gov/files/company_tickers_exchange.json",
]

def _sec_headers() -> Dict[str, str]:
    contact = ""
    try:
        contact = st.secrets.get("SEC_CONTACT", "")
    except Exception:
        contact = ""
    ua = USER_AGENT
    if contact and contact not in ua:
        ua = f"{USER_AGENT} ({contact})"
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json,text/html,*/*",
    }

@st.cache_data(ttl=24 * 60 * 60)
def fetch_sec_ticker_map() -> pd.DataFrame:
    for url in SEC_TICKER_MAP_URLS:
        try:
            r = requests.get(url, headers=_sec_headers(), timeout=20)
            if not (200 <= r.status_code < 300):
                continue
            data = r.json()
            rows = []
            if isinstance(data, dict):
                for _, obj in data.items():
                    t = str(obj.get("ticker", "")).upper().strip()
                    cik = obj.get("cik_str", obj.get("cik", None))
                    title = obj.get("title", obj.get("name", ""))
                    if t and cik is not None:
                        rows.append({"ticker": t, "cik": int(cik), "title": title})
            elif isinstance(data, list):
                for obj in data:
                    t = str(obj.get("ticker", "")).upper().strip()
                    cik = obj.get("cik_str", obj.get("cik", None))
                    title = obj.get("title", obj.get("name", ""))
                    if t and cik is not None:
                        rows.append({"ticker": t, "cik": int(cik), "title": title})
            df = pd.DataFrame(rows).drop_duplicates(subset=["ticker"])
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame(columns=["ticker", "cik", "title"])

def cik10(cik_int: int) -> str:
    return str(int(cik_int)).zfill(10)

@st.cache_data(ttl=CACHE_TTL_META)
def fetch_sec_submissions(cik_int: int) -> Dict:
    url = f"{SEC_BASE}/submissions/CIK{cik10(cik_int)}.json"
    try:
        r = requests.get(url, headers=_sec_headers(), timeout=20)
        if not (200 <= r.status_code < 300):
            return {}
        return r.json() or {}
    except Exception:
        return {}

def _filing_doc_url(cik_int: int, accession: str, primary_doc: str) -> str:
    acc_nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik_int)}/{acc_nodash}/{primary_doc}"

def _ixviewer_url(doc_url: str) -> str:
    return f"https://www.sec.gov/ixviewer/documents/?doc={requests.utils.quote(doc_url)}"

def get_latest_filings_for_ticker(ticker: str, forms: Optional[List[str]] = None, limit: int = 8) -> pd.DataFrame:
    m = fetch_sec_ticker_map()
    if m.empty:
        return pd.DataFrame()
    hit = m[m["ticker"] == ticker.upper()].head(1)
    if hit.empty:
        return pd.DataFrame()
    cik_int = int(hit["cik"].iloc[0])
    sub = fetch_sec_submissions(cik_int)
    recent = (((sub or {}).get("filings") or {}).get("recent") or {})
    if not recent:
        return pd.DataFrame()

    df = pd.DataFrame(recent)
    if df.empty:
        return pd.DataFrame()

    keep = [c for c in ["filingDate", "reportDate", "acceptanceDateTime", "form", "accessionNumber", "primaryDocument"] if c in df.columns]
    df = df[keep].copy()
    df["filingDate"] = pd.to_datetime(df.get("filingDate"), errors="coerce")
    if forms:
        df = df[df["form"].isin(forms)]
    df = df.sort_values("filingDate", ascending=False).head(limit)

    def make_link(row):
        doc = str(row.get("primaryDocument", "") or "")
        acc = str(row.get("accessionNumber", "") or "")
        if not doc or not acc:
            return ""
        doc_url = _filing_doc_url(cik_int, acc, doc)
        return _ixviewer_url(doc_url)

    df["link"] = df.apply(make_link, axis=1)
    df.insert(0, "ticker", ticker.upper())
    df.insert(1, "cik", cik_int)
    return df

@st.cache_data(ttl=CACHE_TTL_META)
def fetch_earnings_calendar_yf(ticker: str, limit: int = 8) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        df = t.get_earnings_dates(limit=limit)
        if df is None or isinstance(df, dict) or df.empty:
            return pd.DataFrame()
        out = df.reset_index()
        if "Earnings Date" in out.columns:
            out = out.rename(columns={"Earnings Date": "earnings_date"})
        elif out.columns[0] not in ("earnings_date",):
            out = out.rename(columns={out.columns[0]: "earnings_date"})
        out["earnings_date"] = pd.to_datetime(out["earnings_date"], errors="coerce", utc=True)
        return out.sort_values("earnings_date", ascending=False)
    except Exception:
        return pd.DataFrame()

def nearest_earnings_catalyst(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df is None or df.empty or "earnings_date" not in df.columns:
        return None
    dts = pd.to_datetime(df["earnings_date"], errors="coerce", utc=True).dropna()
    if dts.empty:
        return None
    # compare as UTC-naive
    dts = dts.dt.tz_convert(None)
    now = pd.Timestamp.utcnow()  # tz-naive UTC
    future = dts[dts >= now]
    if not future.empty:
        return future.min()
    return dts.max()

# =========================
# TECHNICAL SIGNALS (scored)
# =========================
def score_ema_trend(close: pd.Series, ema_s: pd.Series) -> Tuple[str, float]:
    c = safe_last(close)
    e = safe_last(ema_s)
    if not np.isfinite(c) or not np.isfinite(e):
        return ("N/A", 0.0)
    slope = safe_last(ema_s.diff())
    if c > e and slope >= 0:
        return ("Buy (above rising EMA)", 1.0)
    if c < e and slope <= 0:
        return ("Sell (below falling EMA)", -1.0)
    return ("Neutral", 0.0)

def score_rsi(rsi: pd.Series) -> Tuple[str, float]:
    v = safe_last(rsi)
    if not np.isfinite(v):
        return ("N/A", 0.0)
    if v <= 30:
        return ("Buy (oversold)", 1.0)
    if v >= 70:
        return ("Sell (overbought)", -1.0)
    dv = safe_last(rsi.diff())
    if np.isfinite(dv) and dv > 0:
        return ("Neutral (RSI rising)", 0.25)
    if np.isfinite(dv) and dv < 0:
        return ("Neutral (RSI falling)", -0.25)
    return ("Neutral", 0.0)

def score_macd(macd_line: pd.Series, signal_line: pd.Series, hist: pd.Series) -> Tuple[str, float]:
    m = safe_last(macd_line)
    s = safe_last(signal_line)
    h = safe_last(hist)
    if not np.isfinite(m) or not np.isfinite(s) or not np.isfinite(h):
        return ("N/A", 0.0)
    cross = 1.0 if m > s else (-1.0 if m < s else 0.0)
    accel = safe_last(hist.diff())
    bonus = 0.25 if np.isfinite(accel) and accel > 0 else (-0.25 if np.isfinite(accel) and accel < 0 else 0.0)
    score = float(np.clip(cross + bonus, -1.0, 1.0))
    label = "Buy (MACD>Signal)" if score > 0 else ("Sell (MACD<Signal)" if score < 0 else "Neutral")
    return (label, score)

def score_kdj(k: pd.Series, d: pd.Series, j: pd.Series) -> Tuple[str, float]:
    kv, dv, jv = safe_last(k), safe_last(d), safe_last(j)
    if not np.isfinite(kv) or not np.isfinite(dv) or not np.isfinite(jv):
        return ("N/A", 0.0)
    dk = safe_last((k - d))
    dk_prev = safe_last((k - d).shift(1))
    cross_up = np.isfinite(dk_prev) and dk_prev <= 0 and dk > 0
    cross_dn = np.isfinite(dk_prev) and dk_prev >= 0 and dk < 0
    if cross_up and kv < 20:
        return ("Buy (K/D cross up, oversold)", 1.0)
    if cross_dn and kv > 80:
        return ("Sell (K/D cross down, overbought)", -1.0)
    if kv > dv:
        return ("Neutral (K above D)", 0.25)
    if kv < dv:
        return ("Neutral (K below D)", -0.25)
    return ("Neutral", 0.0)

# ---- additional indicators (best-effort)
def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return mid, upper, lower

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return true_range(high, low, close).rolling(window).mean()

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()

def score_momentum(close: pd.Series, lookback: int) -> Tuple[str, float]:
    if close is None or close.dropna().shape[0] < lookback + 2:
        return ("N/A", 0.0)
    ret = float(close.iloc[-1] / close.iloc[-(lookback + 1)] - 1.0)
    if ret > 0.03:
        return (f"Buy ({lookback}d momentum)", 1.0)
    if ret < -0.03:
        return (f"Sell ({lookback}d momentum)", -1.0)
    return (f"Neutral ({lookback}d momentum)", 0.0)

def score_bollinger(close: pd.Series) -> Tuple[str, float]:
    mid, up, lo = bollinger_bands(close, 20, 2.0)
    c = safe_last(close)
    u = safe_last(up)
    l = safe_last(lo)
    m = safe_last(mid)
    if not np.isfinite(c) or not np.isfinite(u) or not np.isfinite(l) or not np.isfinite(m):
        return ("N/A", 0.0)
    denom = (u - l)
    if not np.isfinite(denom) or denom == 0:
        return ("N/A", 0.0)
    pos = (c - l) / denom
    if pos <= 0.1:
        return ("Buy (near lower band)", 0.75)
    if pos >= 0.9:
        return ("Sell (near upper band)", -0.75)
    return ("Neutral (within bands)", 0.25 if c >= m else -0.25)

def score_atr_pct(high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[str, float]:
    a = atr(high, low, close, 14)
    c = safe_last(close)
    ap = safe_last(a) / c if np.isfinite(c) and c != 0 else np.nan
    if not np.isfinite(ap):
        return ("N/A", 0.0)
    if ap > 0.06:
        return ("Neutral (high volatility)", -0.25)
    return ("Neutral (vol normal)", 0.0)

def score_obv_trend(close: pd.Series, volume: Optional[pd.Series]) -> Tuple[str, float]:
    if volume is None or volume.dropna().empty:
        return ("N/A", 0.0)
    o = obv(close, volume)
    slope = safe_last(o.diff().rolling(5).mean())
    if not np.isfinite(slope):
        return ("N/A", 0.0)
    if slope > 0:
        return ("Neutral (OBV rising)", 0.25)
    if slope < 0:
        return ("Neutral (OBV falling)", -0.25)
    return ("Neutral", 0.0)

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

c1, c2 = st.columns([1.2, 1])
with c1:
    date_sel = st.selectbox("Universe date", date_options, index=0)
with c2:
    refresh = st.button("Refresh caches")

if refresh:
    st.cache_data.clear()
    st.success("Caches cleared. Re-run will refetch.")

universe_path = file_map[date_sel]
st.caption(f"Using: `{Path(universe_path).name}`")

uni_raw = pd.read_csv(universe_path)
uni, key_cols_present, canonical_cols = normalize_universe_keep_original(uni_raw)

# =========================
# Opportunity Set
# =========================
st.subheader("Opportunity Set")

if key_cols_present:
    st.dataframe(uni[key_cols_present].copy(), use_container_width=True, height=320)
else:
    st.info("Preferred key columns not found. Showing the first 12 columns instead.")
    st.dataframe(uni.iloc[:, :12], use_container_width=True, height=320)

tickers_all = uni["ticker_norm"].dropna().unique().tolist()
tickers_all = sorted([t for t in tickers_all if t])

ticker_sel = st.selectbox("Select ticker", tickers_all, index=0 if tickers_all else None)
if not ticker_sel:
    st.stop()

row = uni.loc[uni["ticker_norm"] == ticker_sel].head(1)
sector_zacks = str(row["sector_norm"].iloc[0]) if not row.empty else "Unknown"
industry = str(row["industry_norm"].iloc[0]) if not row.empty else ""
sector_bucket = infer_sector_bucket(sector_zacks)

st.caption(f"Selected: **{ticker_sel}** | Sector (Zacks): **{sector_zacks}** | Bucket: **{sector_bucket}** | Industry: **{industry}**")

# =========================
# Sector-Specific Indicators
# =========================
st.subheader("Sector-Specific Indicators")

inds = SECTOR_TO_INDICATORS.get(sector_bucket, SECTOR_TO_INDICATORS["General"])
if not inds:
    st.info("No indicators configured for this sector bucket.")
else:
    rows = []
    for k in inds:
        cfg = INDICATORS.get(k, {})
        ser = fetch_indicator_series(k)
        snap = indicator_snapshot(ser)
        label, score = indicator_signal(cfg, snap)
        rows.append({
            "Indicator": cfg.get("name", k),
            "Key": k,
            "Source": cfg.get("source", ""),
            "Series/Symbol": cfg.get("id", ""),
            "Latest": snap["latest"],
            "Δ 1w": snap["chg_1w"],
            "Δ 1m": snap["chg_1m"],
            "Z(≈1y)": snap["z_1y"],
            "Signal": label,
            "Score": score,
        })
    snap_df = pd.DataFrame(rows)
    st.dataframe(
        snap_df,
        use_container_width=True,
        height=260,
        column_config={
            "Latest": st.column_config.NumberColumn(format="%.4f"),
            "Δ 1w": st.column_config.NumberColumn(format="%.4f"),
            "Δ 1m": st.column_config.NumberColumn(format="%.4f"),
            "Z(≈1y)": st.column_config.NumberColumn(format="%.2f"),
            "Score": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    cols = st.columns(2)
    for i, k in enumerate(inds):
        cfg = INDICATORS.get(k, {})
        ser = fetch_indicator_series(k)
        with cols[i % 2]:
            st.write(f"**{cfg.get('name', k)}**")
            if ser is None or ser.empty:
                st.info("No data (endpoint may be unavailable).")
            else:
                ser2 = ser.dropna().tail(3000).copy()
                fig = px.line(ser2, x="date", y="value", title=None)
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)

# =========================
# Technical Analysis
# =========================
st.subheader("Technical Analysis")

with st.spinner("Fetching price history…"):
    px_panel = fetch_prices_panel([ticker_sel], period=DEFAULT_PRICE_PERIOD, interval=DEFAULT_PRICE_INTERVAL)

info = fetch_info_one(ticker_sel)
df_t = get_ohlc_from_panel(px_panel, ticker_sel)

if df_t.empty or "Close" not in df_t.columns:
    st.warning("No price data for selected ticker.")
    st.stop()

df_t = df_t.dropna(subset=["Close"])
close = df_t["Close"].astype(float)
high = df_t["High"].astype(float) if "High" in df_t.columns else close
low = df_t["Low"].astype(float) if "Low" in df_t.columns else close
volume = df_t["Volume"].astype(float) if "Volume" in df_t.columns else None

ema_spans = [5, 10, 20, 60, 120, 250]
emas = {f"EMA{n}": ema(close, n) for n in ema_spans}

rsi14 = rsi_series(close, 14)
macd_line, signal_line, hist = macd(close, 12, 26, 9)
k_line, d_line, j_line = kdj(high, low, close, n=9, k=3, d=3)

# --- Price + EMAs
price_df = pd.DataFrame({"Date": close.index, "Close": close.values})
for n in ema_spans:
    price_df[f"EMA{n}"] = emas[f"EMA{n}"].values
price_df = price_df.dropna()

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df["Close"], name="Close"))
for n in ema_spans:
    fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df[f"EMA{n}"], name=f"EMA{n}"))
fig_price.update_layout(title=f"{ticker_sel} — Price & EMAs", height=420, legend=dict(orientation="h"))
st.plotly_chart(fig_price, use_container_width=True)

# --- RSI
rsi_df = pd.DataFrame({"Date": rsi14.index, "RSI14": rsi14.values}).dropna()
fig_rsi = px.line(rsi_df, x="Date", y="RSI14", title="RSI(14)")
fig_rsi.update_layout(height=260)
st.plotly_chart(fig_rsi, use_container_width=True)

# --- MACD (legend fix: smaller and above chart)
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
fig_macd.update_layout(
    title="MACD(12,26,9)",
    height=320,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0.0,
        font=dict(size=10),
    ),
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_macd, use_container_width=True)

# --- KDJ
kdj_df = pd.DataFrame({"Date": close.index, "K": k_line.values, "D": d_line.values, "J": j_line.values}).dropna()
fig_kdj = px.line(kdj_df, x="Date", y=["K", "D", "J"], title="KDJ(9,3,3)")
fig_kdj.update_layout(height=320)
st.plotly_chart(fig_kdj, use_container_width=True)

# --- Signals table + composite score (EMA consolidated)
sig_rows = []

# EMA cluster
ema_scores = []
ema_details = []
for n in ema_spans:
    label_n, sc_n = score_ema_trend(close, emas[f"EMA{n}"])
    ema_scores.append(sc_n)
    ema_details.append(f"{n}:{sc_n:+.0f}")
ema_cluster = float(np.nanmean(ema_scores)) if len(ema_scores) else 0.0
sig_rows.append({
    "Indicator": "EMA Cluster (5/10/20/60/120/250)",
    "Value": ema_cluster,
    "Signal": " / ".join(ema_details[:6]),
    "Score": float(np.clip(ema_cluster, -1.0, 1.0)),
})

label, sc = score_rsi(rsi14)
sig_rows.append({"Indicator": "RSI(14)", "Value": safe_last(rsi14), "Signal": label, "Score": sc})

label, sc = score_macd(macd_line, signal_line, hist)
sig_rows.append({"Indicator": "MACD(12,26,9)", "Value": safe_last(hist), "Signal": label, "Score": sc})

label, sc = score_kdj(k_line, d_line, j_line)
sig_rows.append({"Indicator": "KDJ(9,3,3)", "Value": safe_last(k_line), "Signal": label, "Score": sc})

# Additional indicators
label, sc = score_momentum(close, 20)
sig_rows.append({"Indicator": "Momentum 20d", "Value": np.nan, "Signal": label, "Score": sc})

label, sc = score_momentum(close, 60)
sig_rows.append({"Indicator": "Momentum 60d", "Value": np.nan, "Signal": label, "Score": sc})

label, sc = score_bollinger(close)
sig_rows.append({"Indicator": "Bollinger(20,2)", "Value": np.nan, "Signal": label, "Score": sc})

label, sc = score_atr_pct(high, low, close)
sig_rows.append({"Indicator": "ATR% (14)", "Value": np.nan, "Signal": label, "Score": sc})

label, sc = score_obv_trend(close, volume)
sig_rows.append({"Indicator": "OBV Trend", "Value": np.nan, "Signal": label, "Score": sc})

sig_df = pd.DataFrame(sig_rows)
composite = float(sig_df["Score"].mean()) if not sig_df.empty else 0.0
st.markdown(f"#### Indicator Signals (Composite Score: **{composite:+.2f}**)")

st.dataframe(
    sig_df,
    use_container_width=True,
    height=300,
    column_config={
        "Value": st.column_config.NumberColumn(format="%.4f"),
        "Score": st.column_config.NumberColumn(format="%.2f"),
    },
)

# =========================
# News Sentiment
# =========================
st.subheader("News Sentiment")

news_window_label = st.selectbox("News window", ["1w", "2w", "1m", "2m", "3m"], index=2)
days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
news_days = days_map.get(news_window_label, 30)

news_df = fetch_google_news_rss(ticker_sel, days=news_days)

if news_df is None or news_df.empty:
    st.write("No recent RSS news found.")
else:
    news_df["time"] = pd.to_datetime(news_df["time"], utc=True, errors="coerce")
    news_df = news_df.dropna(subset=["title"]).head(25)

    pols = [sentiment_polarity(str(t)) for t in news_df["title"].tolist()]
    sent_df = pd.DataFrame({"polarity": pols}).replace([np.inf, -np.inf], np.nan).dropna()
    if not sent_df.empty:
        fig = px.histogram(sent_df, x="polarity", nbins=20, title="Headline Sentiment Polarity (TextBlob)")
        fig.update_layout(height=240)
        st.plotly_chart(fig, use_container_width=True)

    for _, n in news_df.iterrows():
        t = n.get("time")
        t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
        title = str(n.get("title", ""))
        link = str(n.get("link", ""))
        src = str(n.get("source", ""))
        pol = sentiment_polarity(title)
        st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_ | headline polarity: `{pol:+.2f}`")

# =========================
# SEC Filings & Earnings Catalysts
# =========================
st.subheader("SEC Filings & Earnings Catalysts")

c_sec1, c_sec2 = st.columns([1.3, 1])
with c_sec1:
    forms_sel = st.multiselect("Filings to show", ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"], default=["10-Q", "10-K", "8-K"])
with c_sec2:
    filings_limit = st.slider("Max filings", min_value=3, max_value=20, value=8, step=1)

filings_df = get_latest_filings_for_ticker(ticker_sel, forms=forms_sel, limit=filings_limit)
if filings_df.empty:
    st.info("SEC filings unavailable (ticker->CIK mapping or submissions fetch failed).")
else:
    show_df = filings_df.copy()
    show_df["filingDate"] = pd.to_datetime(show_df["filingDate"], errors="coerce").dt.date
    st.dataframe(
        show_df[["ticker", "form", "filingDate", "reportDate", "accessionNumber", "link"]],
        use_container_width=True,
        height=260,
        column_config={
            "link": st.column_config.LinkColumn("Filing (SEC ixviewer)"),
        },
    )
    st.caption("Tip: open the latest 10-Q/10-K and jump to Items like 1A (Risk Factors), 2/7 (MD&A), 8 (Financials), or scan for 'guidance', 'outlook', 'restructuring', 'impairment', 'material weakness', etc.")

st.markdown("#### Earnings Calendar (best-effort)")
earn_df = fetch_earnings_calendar_yf(ticker_sel, limit=12)
if earn_df.empty:
    st.info("Earnings dates not available via yfinance on this host/IP (best-effort source).")
else:
    cols = [c for c in ["earnings_date", "EPS Estimate", "Reported EPS", "Surprise(%)", "Revenue Estimate", "Reported Revenue"] if c in earn_df.columns]
    show_cols = cols if cols else earn_df.columns[:6].tolist()
    st.dataframe(earn_df[show_cols].head(12), use_container_width=True, height=260)

    nxt = nearest_earnings_catalyst(earn_df)
    if nxt is not None:
        st.success(f"Nearest earnings catalyst (UTC): {nxt.strftime('%Y-%m-%d %H:%M')}")

# =========================
# Investor Relations
# =========================
st.subheader("Investor Relations")

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

        links = extract_ir_news_links(ir_url, max_links=10)
        if not links:
            st.write("No obvious IR/news links found on that page (site may be JS-rendered).")
        else:
            st.markdown("**IR / Press / News links (best-effort extraction):**")
            for txt, href in links:
                st.markdown(f"- [{txt}]({href})")
