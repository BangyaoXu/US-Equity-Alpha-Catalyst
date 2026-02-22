# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import glob
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import StringIO
from urllib.parse import quote

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
DEFAULT_PRICE_PERIOD = "2y"
DEFAULT_PRICE_INTERVAL = "1d"

CACHE_TTL_PRICES = 60 * 60
CACHE_TTL_META = 60 * 60
CACHE_TTL_NEWS = 30 * 60
CACHE_TTL_INDICATORS = 60 * 60
CACHE_TTL_EARNINGS = 15 * 60
CACHE_TTL_RETURNS = 30 * 60
CACHE_TTL_ANALYST = 60 * 60
CACHE_TTL_SECTOR = 60 * 60

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome Safari"
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

# =========================
# “LAST GOOD SERIES” memory + Stooq fallback (fixes blank/broken sector charts)
# =========================
def _last_good_store() -> Dict[str, pd.DataFrame]:
    if "LAST_GOOD_SERIES" not in st.session_state or not isinstance(st.session_state["LAST_GOOD_SERIES"], dict):
        st.session_state["LAST_GOOD_SERIES"] = {}
    return st.session_state["LAST_GOOD_SERIES"]

def _remember_last_good_series(key: str, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    store = _last_good_store()
    if df is not None and isinstance(df, pd.DataFrame) and (not df.empty):
        store[key] = df
        return df
    prev = store.get(key)
    return prev if isinstance(prev, pd.DataFrame) and (not prev.empty) else None

@st.cache_data(ttl=12 * 60 * 60)
def fetch_stooq_daily(stooq_symbol: str) -> pd.DataFrame:
    """
    Fetch daily OHLC from Stooq (free). Returns: date, value (Close)
    """
    s = (stooq_symbol or "").strip().lower()
    if not s:
        return pd.DataFrame()

    url = f"https://stooq.com/q/d/l/?s={quote(s)}&i=d"
    try:
        r = SESSION.get(url, timeout=20, headers={"Accept": "text/csv,*/*"})
        if r.status_code != 200 or not r.text or "Date,Open" not in r.text:
            return pd.DataFrame()

        df = pd.read_csv(StringIO(r.text))
        if df.empty or "Date" not in df.columns or "Close" not in df.columns:
            return pd.DataFrame()

        out = df[["Date", "Close"]].copy()
        out.columns = ["date", "value"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["date", "value"]).sort_values("date")
        return out
    except Exception:
        return pd.DataFrame()

# =========================
# UTILITIES
# =========================
def color_return(v):
    try:
        v = float(v)
    except Exception:
        return ""
    if not np.isfinite(v):
        return ""
    if v > 0:
        return "color: #16a34a; font-weight:600;"   # green
    if v < 0:
        return "color: #dc2626; font-weight:600;"   # red
    return ""

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
    if px_panel is None or getattr(px_panel, "empty", True):
        return pd.DataFrame()
    if isinstance(px_panel.columns, pd.MultiIndex):
        if ticker in px_panel.columns.levels[0]:
            df = px_panel[ticker].copy()
            df.columns = [str(c) for c in df.columns]
            return df
        for t in px_panel.columns.levels[0]:
            if str(t).upper() == str(ticker).upper():
                df = px_panel[t].copy()
                df.columns = [str(c) for c in df.columns]
                return df
        return pd.DataFrame()
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

def normalize_yf_ticker(t: str) -> str:
    return (t or "").upper().strip().replace(".", "-")

def strip_trailing_parens(name: str) -> str:
    s = str(name or "").strip()
    return re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()

# =========================
# HTTP helpers
# =========================
def _get_text(url: str, timeout: int = 25, retries: int = 3) -> Optional[str]:
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout, headers={"Accept": "text/csv,*/*"})
            if 200 <= r.status_code < 300 and r.text and len(r.text) > 20:
                return r.text
            if r.status_code in (403, 429, 500, 502, 503, 504):
                time.sleep(0.6 * (2 ** i))
                continue
            return None
        except Exception:
            time.sleep(0.6 * (2 ** i))
            continue
    return None

# =========================
# PRICE RETURNS SINCE UNIVERSE DATE
# =========================
@st.cache_data(ttl=CACHE_TTL_RETURNS)
def fetch_returns_since(universe_date_str: str, tickers: List[str]) -> Dict[str, float]:
    if not tickers:
        return {}
    try:
        start_dt = pd.to_datetime(universe_date_str, errors="coerce")
        if pd.isna(start_dt):
            return {}

        now_utc = pd.Timestamp.now(tz="UTC")
        end_dt = (now_utc + pd.Timedelta(days=1)).date()
        start_str = start_dt.date().isoformat()
        end_str = end_dt.isoformat()

        px = yf.download(
            tickers=[normalize_yf_ticker(t) for t in tickers],
            start=start_str,
            end=end_str,
            auto_adjust=True,
            group_by="ticker",
            threads=False,
            progress=False,
        )
        if px is None or px.empty:
            return {}

        ret_map: Dict[str, float] = {}
        for t in tickers:
            tt = normalize_yf_ticker(t)
            close = None
            if isinstance(px.columns, pd.MultiIndex):
                if tt in px.columns.levels[0] and ("Close" in px[tt].columns):
                    close = px[tt]["Close"].dropna()
                else:
                    for lvl in px.columns.levels[0]:
                        if str(lvl).upper() == tt:
                            if "Close" in px[lvl].columns:
                                close = px[lvl]["Close"].dropna()
                            break
            else:
                if "Close" in px.columns:
                    close = px["Close"].dropna()

            if close is None or close.empty or close.shape[0] < 2:
                ret_map[t.upper().strip()] = np.nan
                continue

            first_px = float(close.iloc[0])
            last_px = float(close.iloc[-1])
            if not np.isfinite(first_px) or first_px == 0 or not np.isfinite(last_px):
                ret_map[t.upper().strip()] = np.nan
            else:
                ret_map[t.upper().strip()] = (last_px / first_px) - 1.0

        return ret_map
    except Exception:
        return {}

# =========================
# DATA FETCH (prices/meta/news)
# =========================
@st.cache_data(ttl=CACHE_TTL_PRICES)
def fetch_prices_panel(tickers: List[str], period: str = DEFAULT_PRICE_PERIOD, interval: str = DEFAULT_PRICE_INTERVAL) -> pd.DataFrame:
    try:
        return yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            group_by="ticker",
            threads=False,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL_META)
def fetch_info_one(ticker: str) -> Dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

# =========================
# News RSS (generalized query)
# =========================
@st.cache_data(ttl=CACHE_TTL_NEWS)
def fetch_google_news_rss_query(query: str, days: int) -> pd.DataFrame:
    q = (query or "").strip()
    if not q:
        return pd.DataFrame()

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
                    pub = pub if pub.tzinfo else pub.replace(tzinfo=timezone.utc)
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

@st.cache_data(ttl=CACHE_TTL_NEWS)
def fetch_google_news_rss(ticker: str, days: int) -> pd.DataFrame:
    return fetch_google_news_rss_query(f"{ticker} stock", days)

# =========================
# FRED generic (for sector series like Finance)
# =========================
@st.cache_data(ttl=24 * 60 * 60)
def fetch_fred_series(series_id: str) -> pd.DataFrame:
    sid = (series_id or "").strip()
    if not sid:
        return pd.DataFrame()
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={requests.utils.quote(sid)}"
    txt = _get_text(url, timeout=25, retries=3)
    if not txt:
        return pd.DataFrame()
    try:
        df = pd.read_csv(StringIO(txt))
        if df.shape[1] < 2:
            return pd.DataFrame()
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"]).sort_values("date")
        return df
    except Exception:
        return pd.DataFrame()

# =========================
# EIA (best-effort scrapes) – weekly refinery utilization + monthly rig count
# =========================
@st.cache_data(ttl=24 * 60 * 60)
def fetch_eia_pet_weekly_series(series_code: str) -> pd.DataFrame:
    """
    Best-effort: EIA weekly petroleum series via LeafHandler.
    Example: WPULEUS3 = Weekly U.S. Percent Utilization of Refinery Operable Capacity.
    Returns df(date,value).
    """
    code = (series_code or "").strip().upper()
    if not code:
        return pd.DataFrame()

    url = f"https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?f=W&n=PET&s={requests.utils.quote(code)}"
    html = fetch_html(url, timeout=20)
    if not html:
        return pd.DataFrame()

    try:
        # EIA page usually has a simple two-column table: Date, Value
        tables = pd.read_html(html)
        if not tables:
            return pd.DataFrame()

        # pick the first table that looks like date/value
        best = None
        for tb in tables:
            if tb is None or tb.empty or tb.shape[1] < 2:
                continue
            c0 = str(tb.columns[0]).lower()
            if "date" in c0 or "week" in c0:
                best = tb
                break
        if best is None:
            best = tables[0]

        df = best.copy()
        df.columns = ["date", "value"] + [f"c{i}" for i in range(3, df.shape[1] + 1)]
        df = df[["date", "value"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"]).sort_values("date")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=24 * 60 * 60)
def fetch_eia_ng_monthly_rig_count() -> pd.DataFrame:
    """
    Best-effort: EIA monthly 'U.S. Crude Oil and Natural Gas Rotary Rigs in Operation'
    from EIA dnav table.
    Returns df(date,value) monthly.
    """
    url = "https://www.eia.gov/dnav/ng/hist/e_ertrr0_xr0_nus_cm.htm"
    html = fetch_html(url, timeout=20)
    if not html:
        return pd.DataFrame()

    try:
        tables = pd.read_html(html)
        if not tables:
            return pd.DataFrame()

        # The main table is usually the first large one: Year rows, Jan..Dec columns
        tb = max(tables, key=lambda t: (0 if t is None else t.size))
        if tb is None or tb.empty:
            return pd.DataFrame()

        df = tb.copy()
        # Expect first col "Year", then months
        year_col = df.columns[0]
        df = df.rename(columns={year_col: "Year"})
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year"])

        # melt months to rows
        month_cols = [c for c in df.columns if str(c).strip()[:3].lower() in
                      ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]]
        if not month_cols:
            return pd.DataFrame()

        m = df.melt(id_vars=["Year"], value_vars=month_cols, var_name="Month", value_name="value")
        m["value"] = pd.to_numeric(m["value"], errors="coerce")
        m = m.dropna(subset=["value"])

        # build date
        month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
        m["m"] = m["Month"].astype(str).str.strip().str[:3].str.lower().map(month_map)
        m = m.dropna(subset=["m"])
        m["date"] = pd.to_datetime(
            m["Year"].astype(int).astype(str) + "-" + m["m"].astype(int).astype(str) + "-01",
            errors="coerce"
        )
        out = m[["date", "value"]].dropna().sort_values("date")
        return out
    except Exception:
        return pd.DataFrame()

# =========================
# EARNINGS (forecasts/actual/surprises via yfinance)
# =========================
@st.cache_data(ttl=CACHE_TTL_EARNINGS)
def fetch_earnings_dates_yf(ticker: str, limit: int = 12) -> pd.DataFrame:
    try:
        t = yf.Ticker(normalize_yf_ticker(ticker))
        df = t.get_earnings_dates(limit=limit)
        if df is None or isinstance(df, dict) or df.empty:
            return pd.DataFrame()
        out = df.reset_index()
        if "Earnings Date" in out.columns:
            out = out.rename(columns={"Earnings Date": "earnings_date"})
        elif out.columns[0] != "earnings_date":
            out = out.rename(columns={out.columns[0]: "earnings_date"})
        out["earnings_date"] = pd.to_datetime(out["earnings_date"], errors="coerce", utc=True)
        return out.sort_values("earnings_date", ascending=False)
    except Exception:
        return pd.DataFrame()

def nearest_earnings_catalyst(earn_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if earn_df is None or earn_df.empty or "earnings_date" not in earn_df.columns:
        return None

    dts = pd.to_datetime(earn_df["earnings_date"], errors="coerce", utc=True).dropna()
    if dts.empty:
        return None

    dts_naive = dts.dt.tz_convert("UTC").dt.tz_localize(None)
    arr = dts_naive.to_numpy(dtype="datetime64[ns]")

    now = np.datetime64(pd.Timestamp.now(tz="UTC").tz_localize(None).to_datetime64())
    future = arr[arr >= now]

    if future.size > 0:
        return pd.Timestamp(future.min())
    return pd.Timestamp(arr.max())

@st.cache_data(ttl=CACHE_TTL_EARNINGS)
def fetch_earnings_surprise_history(ticker: str, limit: int = 24) -> pd.DataFrame:
    df = fetch_earnings_dates_yf(ticker, limit=limit)
    if df is None or df.empty:
        return pd.DataFrame()

    s_col = None
    for c in ["Surprise(%)", "Surprise (%)", "Surprise %", "Surprise"]:
        if c in df.columns:
            s_col = c
            break
    if s_col is None:
        return pd.DataFrame()

    out = df.copy()
    out["earnings_date"] = pd.to_datetime(out["earnings_date"], errors="coerce", utc=True)
    out["surprise_pct"] = pd.to_numeric(out[s_col], errors="coerce")
    out = out.dropna(subset=["earnings_date", "surprise_pct"]).sort_values("earnings_date")
    return out[["earnings_date", "surprise_pct"]]

# =========================
# Earnings estimates snapshot (Yahoo "trend" object via yfinance)
# =========================
@st.cache_data(ttl=CACHE_TTL_ANALYST)
def fetch_earnings_estimate_snapshot_yf(ticker: str) -> pd.DataFrame:
    sym = normalize_yf_ticker(ticker)
    try:
        t = yf.Ticker(sym)
        et = getattr(t, "earnings_trend", None)
        if et is None:
            return pd.DataFrame()
        if isinstance(et, pd.DataFrame) and not et.empty:
            df = et.copy().reset_index(drop=False)
            return df.head(12)
        return pd.DataFrame()
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
def fetch_html(url: str, timeout=12) -> Optional[str]:
    try:
        r = SESSION.get(url, timeout=timeout)
        if 200 <= r.status_code < 300 and r.text and "html" in (r.headers.get("Content-Type", "").lower()):
            return r.text
        if 200 <= r.status_code < 300 and r.text and len(r.text) > 2000:
            return r.text
        return None
    except Exception:
        return None

# =========================
# TradingEconomics Historical API (guest:guest)
# =========================
@st.cache_data(ttl=12 * 60 * 60)
def fetch_tradingeconomics_hist_via_api(country_slug: str, indicator_slug: str) -> pd.DataFrame:
    """
    Uses TradingEconomics historical endpoint (guest:guest) and returns df(date,value).
    country_slug: e.g. 'united-states', 'china'
    indicator_slug: e.g. 'manufacturing-pmi'
    """
    country = (country_slug or "").strip().lower()
    ind_slug = (indicator_slug or "").strip().lower()
    if not country or not ind_slug:
        return pd.DataFrame()

    # Convert slug -> indicator string used by TE API
    # e.g. 'manufacturing-pmi' -> 'manufacturing pmi'
    indicator = ind_slug.replace("-", " ").strip()

    # Historical endpoint supports guest:guest and CSV
    api_url = (
        "https://api.tradingeconomics.com/historical/"
        f"country/{requests.utils.quote(country)}/indicator/{requests.utils.quote(indicator)}"
        "?c=guest:guest&f=csv"
    )

    txt = _get_text(api_url, timeout=25, retries=3)
    if not txt:
        return pd.DataFrame()

    try:
        df = pd.read_csv(StringIO(txt))
        if df is None or df.empty:
            return pd.DataFrame()

        # TE CSV typically includes DateTime and Close (or Value)
        dt_col = None
        for c in df.columns:
            if str(c).strip().lower() == "datetime":
                dt_col = c
                break

        val_col = None
        for cand in ["close", "value"]:
            for c in df.columns:
                if str(c).strip().lower() == cand:
                    val_col = c
                    break
            if val_col:
                break

        if not dt_col or not val_col:
            return pd.DataFrame()

        out = df[[dt_col, val_col]].copy()
        out.columns = ["date", "value"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["date", "value"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        return out
    except Exception:
        return pd.DataFrame()

# =========================
# TradingEconomics (PMI) – best-effort HTML table scrape (used by Basic Materials only)
# =========================
@st.cache_data(ttl=12 * 60 * 60)
def fetch_tradingeconomics_hist_from_page(url: str) -> pd.DataFrame:
    html = fetch_html(url, timeout=16)
    if not html:
        return pd.DataFrame()

    try:
        tables = pd.read_html(html)
    except Exception:
        tables = []

    best = None
    for tb in tables:
        if tb is None or tb.empty:
            continue
        cols = [str(c).strip().lower() for c in tb.columns]
        if any("date" in c for c in cols) and (any("value" in c for c in cols) or any("last" in c for c in cols)):
            best = tb.copy()
            break

    if best is None or best.empty:
        return pd.DataFrame()

    c_date, c_val = None, None
    for c in best.columns:
        lc = str(c).strip().lower()
        if c_date is None and "date" in lc:
            c_date = c
        if c_val is None and ("value" in lc or "last" in lc or lc == "actual"):
            c_val = c

    if c_date is None or c_val is None:
        return pd.DataFrame()

    out = best[[c_date, c_val]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out

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
# Business Services: fundamentals time series + best-effort operational KPIs
# =========================
@st.cache_data(ttl=CACHE_TTL_SECTOR)
def fetch_business_services_fundamentals(sym: str) -> Dict[str, pd.DataFrame]:
    """
    Returns time series dataframes (date,value) for:
      - revenue_yoy_pct
      - operating_margin_pct
      - ebitda_margin_pct (best-effort, may be empty)
    Uses Yahoo quarterly_financials rows.
    """
    sym = normalize_yf_ticker(sym)
    inc = yf_quarterly_income(sym)

    def _series_to_df(s: Optional[pd.Series]) -> pd.DataFrame:
        if s is None or not isinstance(s, pd.Series):
            return pd.DataFrame()
        x = s.dropna().sort_index()
        if x.empty:
            return pd.DataFrame()
        out = pd.DataFrame({"date": pd.to_datetime(x.index, errors="coerce"), "value": pd.to_numeric(x.values, errors="coerce")})
        out = out.dropna(subset=["date", "value"]).sort_values("date")
        return out

    rev_s = _row_get(inc, ["total revenue", "totalrevenue", "revenue"])
    opinc_s = _row_get(inc, ["operating income", "operatingincome"])
    ebitda_s = _row_get(inc, ["ebitda"])  # often missing on Yahoo quarterly_financials

    # Revenue YoY (%) series: compare quarter t vs t-4
    rev_yoy = pd.Series(dtype=float)
    if rev_s is not None:
        r = pd.to_numeric(rev_s, errors="coerce").dropna().sort_index()
        if r.shape[0] >= 5:
            rev_yoy = (r / r.shift(4) - 1.0) * 100.0

    # Operating margin (%) series
    op_margin = pd.Series(dtype=float)
    if opinc_s is not None and rev_s is not None:
        a = pd.to_numeric(opinc_s, errors="coerce").dropna().sort_index()
        b = pd.to_numeric(rev_s, errors="coerce").dropna().sort_index()
        df = pd.concat([a, b], axis=1, join="inner")
        if not df.empty:
            op_margin = (df.iloc[:, 0] / df.iloc[:, 1].replace(0, np.nan)) * 100.0

    # EBITDA margin (%) series (best-effort)
    ebitda_margin = pd.Series(dtype=float)
    if ebitda_s is not None and rev_s is not None:
        a = pd.to_numeric(ebitda_s, errors="coerce").dropna().sort_index()
        b = pd.to_numeric(rev_s, errors="coerce").dropna().sort_index()
        df = pd.concat([a, b], axis=1, join="inner")
        if not df.empty:
            ebitda_margin = (df.iloc[:, 0] / df.iloc[:, 1].replace(0, np.nan)) * 100.0

    return {
        "revenue_yoy_pct": _series_to_df(rev_yoy),
        "operating_margin_pct": _series_to_df(op_margin),
        "ebitda_margin_pct": _series_to_df(ebitda_margin),
    }

@st.cache_data(ttl=24 * 60 * 60)
def fetch_business_services_operational_kpis_best_effort(ticker: str) -> Dict[str, float]:
    """
    Best-effort extraction from SEC XBRL companyfacts.
    Many of these are NON-GAAP / not standardized => often unavailable.

    Returns:
      - backlog_latest
      - backlog_yoy_pct
      - bookings_latest
      - bookings_yoy_pct
      - book_to_bill
      - retention_rate_pct
      - renewal_rate_pct
    """
    out = {
        "backlog_latest": np.nan,
        "backlog_yoy_pct": np.nan,
        "bookings_latest": np.nan,
        "bookings_yoy_pct": np.nan,
        "book_to_bill": np.nan,
        "retention_rate_pct": np.nan,
        "renewal_rate_pct": np.nan,
    }

    sym = normalize_yf_ticker(ticker)
    cik_int = _resolve_cik_for_ticker(sym)
    if cik_int is None:
        return out

    cf = fetch_sec_companyfacts(int(cik_int))

    # --- Backlog proxies (common-ish tags; not guaranteed) ---
    backlog_tags = [
        "TransactionPriceAllocatedToRemainingPerformanceObligations",
        "RevenueRemainingPerformanceObligation",
        "RemainingPerformanceObligation",
        "RemainingPerformanceObligationCurrent",
        "RemainingPerformanceObligationNoncurrent",
        "Backlog",
    ]

    # --- Bookings / orders proxies (rare; not guaranteed) ---
    bookings_tags = [
        "Orders",
        "OrderBacklog",
        "SalesOrderBacklog",
        "ContractWithCustomerLiability",  # sometimes used as a pipeline proxy, very imperfect
    ]

    def _latest_and_yoy(companyfacts: dict, tags: List[str]) -> Tuple[float, float]:
        """
        Return (latest_value, yoy_pct) best-effort by scanning all facts units arrays.
        """
        if not companyfacts:
            return (np.nan, np.nan)

        facts = (companyfacts.get("facts") or {})
        # merge all namespaces to scan (us-gaap first)
        namespaces = []
        if "us-gaap" in facts and isinstance(facts["us-gaap"], dict):
            namespaces.append(("us-gaap", facts["us-gaap"]))
        for ns, ns_obj in facts.items():
            if ns == "us-gaap":
                continue
            if isinstance(ns_obj, dict):
                namespaces.append((ns, ns_obj))

        # collect candidate points (end_date_str, value)
        pts: List[Tuple[str, float]] = []
        for _, ns_obj in namespaces:
            for tag in tags:
                obj = ns_obj.get(tag)
                if not obj:
                    continue
                units = obj.get("units") or {}
                for _, arr in units.items():
                    if not isinstance(arr, list):
                        continue
                    for it in arr:
                        v = it.get("val", None)
                        end = it.get("end", None) or it.get("fy", None) or it.get("fp", None)
                        if v is None or end is None:
                            continue
                        try:
                            vv = float(v)
                        except Exception:
                            continue
                        if not np.isfinite(vv):
                            continue
                        pts.append((str(end), vv))

        if not pts:
            return (np.nan, np.nan)

        pts.sort(key=lambda x: x[0])
        latest_key, latest_val = pts[-1]
        # find ~1y earlier by key prefix match (best-effort string compare)
        # If keys are ISO dates, just use the last that is <= (latest_year-1)
        yoy = np.nan
        try:
            latest_year = int(latest_key[:4])
            target_year = latest_year - 1
            prev_candidates = [p for p in pts if p[0].startswith(str(target_year))]
            if prev_candidates:
                prev_key, prev_val = prev_candidates[-1]
                if np.isfinite(prev_val) and prev_val != 0:
                    yoy = (latest_val / prev_val - 1.0) * 100.0
        except Exception:
            pass

        return (latest_val, yoy)

    backlog_latest, backlog_yoy = _latest_and_yoy(cf, backlog_tags)
    bookings_latest, bookings_yoy = _latest_and_yoy(cf, bookings_tags)

    out["backlog_latest"] = backlog_latest
    out["backlog_yoy_pct"] = backlog_yoy
    out["bookings_latest"] = bookings_latest
    out["bookings_yoy_pct"] = bookings_yoy

    # Book-to-bill = bookings / revenue (revenue from SEC facts)
    rev_latest = _latest_fact_value(cf, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"])
    if np.isfinite(bookings_latest) and np.isfinite(rev_latest) and rev_latest != 0:
        out["book_to_bill"] = bookings_latest / rev_latest

    # Retention / renewal: typically not XBRL standardized -> leave NaN unless you add custom tags later
    return out

# =========================
# FDA decision calendar scraping (best-effort)
# =========================
@st.cache_data(ttl=6 * 60 * 60)
def fetch_fda_calendar(ticker: str, company_name: str = "") -> pd.DataFrame:
    targets = []
    t = (ticker or "").upper().strip()
    if t:
        targets.append(t)
    if company_name:
        targets.append(company_name.strip())

    urls = [
        "https://www.rttnews.com/corpinfo/fdacalendar.aspx",
        "https://www.fdatracker.com/fda-calendar/",
        "https://www.tipranks.com/calendars/fda",
    ]

    for url in urls:
        html = fetch_html(url, timeout=18)
        if not html:
            continue
        try:
            soup = BeautifulSoup(html, "lxml")
            tables = soup.find_all("table")
            if not tables:
                continue

            rows_out = []
            for tb in tables[:3]:
                for tr in tb.find_all("tr"):
                    tds = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
                    if len(tds) < 3:
                        continue
                    row_txt = " ".join(tds).upper()
                    if targets and not any(x.upper() in row_txt for x in targets if x):
                        continue
                    rows_out.append(tds)

            if not rows_out:
                continue

            max_len = max(len(r) for r in rows_out)
            cols = [f"col{i+1}" for i in range(max_len)]
            df = pd.DataFrame([r + [""] * (max_len - len(r)) for r in rows_out], columns=cols)

            for c in cols[:3]:
                dtv = pd.to_datetime(df[c], errors="coerce")
                if dtv.notna().mean() > 0.3:
                    df = df.rename(columns={c: "date"})
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    break

            df.insert(0, "source_url", url)
            return df.head(60)
        except Exception:
            continue

    return pd.DataFrame()

# =========================
# ANALYST RATINGS (yfinance)
# =========================
def _to_num(x) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return np.nan

@st.cache_data(ttl=CACHE_TTL_ANALYST)
def fetch_analyst_price_targets_yf(ticker: str) -> Dict[str, float]:
    out = {"mean": np.nan, "high": np.nan, "low": np.nan, "median": np.nan}
    sym = normalize_yf_ticker(ticker)
    try:
        info = fetch_info_one(sym) or {}
        for k_src, k_dst in [
            ("targetMeanPrice", "mean"),
            ("targetHighPrice", "high"),
            ("targetLowPrice", "low"),
            ("targetMedianPrice", "median"),
        ]:
            v = info.get(k_src, None)
            if v is not None and str(v) != "nan":
                try:
                    out[k_dst] = float(v)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        t = yf.Ticker(sym)
        apt = getattr(t, "analyst_price_targets", None)
        if callable(apt):
            aptv = apt()
            if isinstance(aptv, dict):
                for k in out.keys():
                    if k in aptv:
                        try:
                            out[k] = float(aptv[k])
                        except Exception:
                            pass
    except Exception:
        pass

    return out

@st.cache_data(ttl=CACHE_TTL_ANALYST)
def fetch_upgrades_downgrades_yf(ticker: str, max_rows: int = 50) -> pd.DataFrame:
    sym = normalize_yf_ticker(ticker)
    try:
        t = yf.Ticker(sym)
        ud = getattr(t, "upgrades_downgrades", None)
        if ud is None:
            return pd.DataFrame()
        if isinstance(ud, pd.DataFrame) and not ud.empty:
            df = ud.copy()
            if df.index.name is None:
                df.index.name = "Date"
            df = df.reset_index()
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
            return df.head(max_rows)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# =========================
# SECTOR/COMPANY INDICATORS (from Yahoo quarterly statements)
# =========================
@st.cache_data(ttl=CACHE_TTL_SECTOR)
def yf_quarterly_income(sym: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(sym)
        df = getattr(t, "quarterly_financials", None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            out = df.copy()
            out.columns = pd.to_datetime(out.columns, errors="coerce")
            return out
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL_SECTOR)
def yf_quarterly_balance(sym: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(sym)
        df = getattr(t, "quarterly_balance_sheet", None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            out = df.copy()
            out.columns = pd.to_datetime(out.columns, errors="coerce")
            return out
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL_SECTOR)
def yf_quarterly_cashflow(sym: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(sym)
        df = getattr(t, "quarterly_cashflow", None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            out = df.copy()
            out.columns = pd.to_datetime(out.columns, errors="coerce")
            return out
    except Exception:
        pass
    return pd.DataFrame()

def _row_get(df: pd.DataFrame, keys: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    idx_lower = {str(i).lower(): i for i in df.index}
    for k in keys:
        hit = idx_lower.get(k.lower())
        if hit is not None:
            s = df.loc[hit]
            if isinstance(s, pd.Series):
                return s
    return None

def _q_yoy_growth(series: pd.Series) -> float:
    if series is None or series.dropna().shape[0] < 5:
        return np.nan
    s = series.dropna().sort_index()
    if s.shape[0] < 5:
        return np.nan
    last = float(s.iloc[-1])
    prev = float(s.iloc[-5])
    if not np.isfinite(last) or not np.isfinite(prev) or prev == 0:
        return np.nan
    return (last / prev - 1.0) * 100.0

def _q_margin_pct(num: pd.Series, den: pd.Series) -> float:
    if num is None or den is None:
        return np.nan
    a = num.dropna().sort_index()
    b = den.dropna().sort_index()
    if a.empty or b.empty:
        return np.nan
    df = pd.concat([a, b], axis=1, join="inner")
    if df.empty:
        return np.nan
    vnum = float(df.iloc[-1, 0])
    vden = float(df.iloc[-1, 1])
    if not np.isfinite(vnum) or not np.isfinite(vden) or vden == 0:
        return np.nan
    return (vnum / vden) * 100.0

def _days_inventory(balance: pd.DataFrame, income: pd.DataFrame) -> float:
    inv = _row_get(balance, ["inventory"])
    cogs = _row_get(income, ["cost of revenue", "costofrevenue", "cost of goods sold", "costofgoodssold"])
    if inv is None or cogs is None:
        return np.nan
    inv = inv.dropna().sort_index()
    cogs = cogs.dropna().sort_index()
    if inv.empty or cogs.empty:
        return np.nan
    df = pd.concat([inv, cogs], axis=1, join="inner")
    if df.empty:
        return np.nan
    inv_last = float(df.iloc[-1, 0])
    cogs_last = float(df.iloc[-1, 1])
    if not np.isfinite(inv_last) or not np.isfinite(cogs_last) or cogs_last == 0:
        return np.nan
    return (inv_last / (abs(cogs_last) * 4.0)) * 365.0

def _ar_days(balance: pd.DataFrame, income: pd.DataFrame) -> float:
    ar = _row_get(balance, ["net receivables", "accounts receivable", "accountsreceivable"])
    rev = _row_get(income, ["total revenue", "totalrevenue", "revenue"])
    if ar is None or rev is None:
        return np.nan
    ar = ar.dropna().sort_index()
    rev = rev.dropna().sort_index()
    df = pd.concat([ar, rev], axis=1, join="inner")
    if df.empty:
        return np.nan
    ar_last = float(df.iloc[-1, 0])
    rev_last = float(df.iloc[-1, 1])
    if not np.isfinite(ar_last) or not np.isfinite(rev_last) or rev_last == 0:
        return np.nan
    return (ar_last / (rev_last * 4.0)) * 365.0

# =========================
# SEC EDGAR (robust)
# =========================
SEC_BASE = "https://data.sec.gov"
SEC_TICKER_MAP_URLS = [
    "https://www.sec.gov/files/company_tickers.json",
    "https://www.sec.gov/files/company_tickers_exchange.json",
]

def normalize_sec_ticker(t: str) -> str:
    return (t or "").upper().strip().replace(".", "-")

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
        "Accept": "application/json,text/html,*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

def _sec_get_json(url: str, timeout: int = 25, retries: int = 4, throttle_s: float = 0.12) -> Optional[dict]:
    for i in range(retries):
        try:
            time.sleep(throttle_s)
            r = SESSION.get(url, headers=_sec_headers(), timeout=timeout)
            if 200 <= r.status_code < 300:
                return r.json()
            if r.status_code in (403, 429, 500, 502, 503, 504):
                time.sleep(0.6 * (2 ** i))
                continue
            return None
        except Exception:
            time.sleep(0.6 * (2 ** i))
            continue
    return None

def cik10(cik_int: int) -> str:
    return str(int(cik_int)).zfill(10)

@st.cache_data(ttl=24 * 60 * 60)
def fetch_sec_companyfacts(cik_int: int) -> dict:
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik10(cik_int)}.json"
    data = _sec_get_json(url, timeout=25, retries=4, throttle_s=0.12)
    return data or {}

def _latest_fact_value(companyfacts: dict, tags: List[str]) -> float:
    if not companyfacts:
        return np.nan

    facts = (companyfacts.get("facts") or {})
    usg = (facts.get("us-gaap") or {})
    for tag in tags:
        obj = usg.get(tag)
        if not obj:
            continue
        units = obj.get("units") or {}
        for _, arr in units.items():
            if not isinstance(arr, list) or not arr:
                continue
            best = None
            for it in arr:
                v = it.get("val", None)
                end = it.get("end", it.get("fy", None))
                if v is None:
                    continue
                try:
                    v = float(v)
                except Exception:
                    continue
                if not np.isfinite(v):
                    continue
                key = str(end) if end is not None else ""
                if best is None or key > best[0]:
                    best = (key, v)
            if best is not None:
                return float(best[1])

    for ns, ns_obj in facts.items():
        if not isinstance(ns_obj, dict):
            continue
        for tag in tags:
            obj = ns_obj.get(tag)
            if not obj:
                continue
            units = obj.get("units") or {}
            for _, arr in units.items():
                if not isinstance(arr, list) or not arr:
                    continue
                best = None
                for it in arr:
                    v = it.get("val", None)
                    end = it.get("end", it.get("fy", None))
                    if v is None:
                        continue
                    try:
                        v = float(v)
                    except Exception:
                        continue
                    if not np.isfinite(v):
                        continue
                    key = str(end) if end is not None else ""
                    if best is None or key > best[0]:
                        best = (key, v)
                if best is not None:
                    return float(best[1])

    return np.nan

@st.cache_data(ttl=24 * 60 * 60)
def fetch_sec_ticker_map() -> pd.DataFrame:
    for url in SEC_TICKER_MAP_URLS:
        data = _sec_get_json(url)
        if not data:
            continue
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
    return pd.DataFrame(columns=["ticker", "cik", "title"])

@st.cache_data(ttl=CACHE_TTL_META)
def fetch_sec_submissions(cik_int: int) -> dict:
    url = f"{SEC_BASE}/submissions/CIK{cik10(cik_int)}.json"
    data = _sec_get_json(url)
    return data or {}

def _sec_index_url(cik_int: int, accession: str) -> str:
    acc_nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik_int)}/{acc_nodash}/{accession}-index.html"

def _sec_primary_url(cik_int: int, accession: str, primary_doc: str) -> str:
    acc_nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik_int)}/{acc_nodash}/{primary_doc}"

def _resolve_cik_for_ticker(ticker: str) -> Optional[int]:
    t = (ticker or "").upper().strip()
    if not t:
        return None

    overrides = st.session_state.get("CIK_OVERRIDES", {})
    if isinstance(overrides, dict) and t in overrides:
        try:
            return int(overrides[t])
        except Exception:
            pass

    m = fetch_sec_ticker_map()
    if m.empty:
        return None

    t_norm = normalize_sec_ticker(t)
    hit = m[m["ticker"] == t_norm].head(1)
    if hit.empty:
        hit = m[m["ticker"] == t].head(1)

    if hit.empty:
        return None
    return int(hit["cik"].iloc[0])

def fetch_tier1_ratio_best_effort(ticker: str) -> float:
    sym = normalize_yf_ticker(ticker)
    info = fetch_info_one(sym) or {}

    for k in ["tier1CapitalRatio", "tier1RiskBasedCapitalRatio", "cet1Ratio", "commonEquityTier1Ratio"]:
        v = info.get(k)
        try:
            v = float(v)
            if np.isfinite(v):
                return v * 100.0 if v <= 1.5 else v
        except Exception:
            pass

    cik_int = _resolve_cik_for_ticker(sym)
    if cik_int is None:
        return np.nan

    cf = fetch_sec_companyfacts(int(cik_int))
    cand_tags = [
        "Tier1CapitalRatio",
        "Tier1RiskBasedCapitalRatio",
        "CommonEquityTier1CapitalRatio",
        "CommonEquityTier1Ratio",
        "CET1CapitalRatio",
        "CET1Ratio",
        "TotalRiskBasedCapitalRatio",
    ]
    v = _latest_fact_value(cf, cand_tags)
    if np.isfinite(v):
        return v * 100.0 if v <= 1.5 else v

    return np.nan

def get_latest_filings_for_ticker(ticker: str, forms: Optional[List[str]] = None, limit: int = 8) -> pd.DataFrame:
    cik_int = _resolve_cik_for_ticker(ticker)
    if cik_int is None:
        return pd.DataFrame()

    sub = fetch_sec_submissions(cik_int)
    recent = (((sub or {}).get("filings") or {}).get("recent") or {})
    if not recent:
        return pd.DataFrame()

    df = pd.DataFrame(recent)
    if df.empty:
        return pd.DataFrame()

    keep = [c for c in [
        "filingDate", "reportDate", "acceptanceDateTime",
        "form", "accessionNumber", "primaryDocument",
        "isXBRL", "isInlineXBRL"
    ] if c in df.columns]

    df = df[keep].copy()
    df["filingDate"] = pd.to_datetime(df.get("filingDate"), errors="coerce")

    if forms:
        df = df[df["form"].isin(forms)]
    df = df.sort_values("filingDate", ascending=False).head(limit)

    def build_links(row):
        acc = str(row.get("accessionNumber", "") or "")
        doc = str(row.get("primaryDocument", "") or "")
        if not acc:
            return pd.Series({"index_url": "", "primary_url": ""})
        index_url = _sec_index_url(cik_int, acc)
        primary_url = _sec_primary_url(cik_int, acc, doc) if doc else ""
        return pd.Series({"index_url": index_url, "primary_url": primary_url})

    links = df.apply(build_links, axis=1)
    df = pd.concat([df, links], axis=1)

    df.insert(0, "ticker", ticker.upper())
    df.insert(1, "cik", int(cik_int))
    return df

# =========================
# Indicator bundle (stock + sector maps)
# =========================
@st.cache_data(ttl=CACHE_TTL_SECTOR)
def fetch_indicator_bundle(ticker: str) -> Dict[str, object]:
    sym = normalize_yf_ticker(ticker)
    info = fetch_info_one(sym) or {}
    inc = yf_quarterly_income(sym)
    bal = yf_quarterly_balance(sym)
    cfs = yf_quarterly_cashflow(sym)

    scalars: List[Dict[str, object]] = []
    def add_scalar(name, value, unit=None, update=None, source=None, definition=None):
        scalars.append(
            {
                "Indicator": name,
                "Value": value,
                "Unit": unit,
                "Update": update,
                "Source": source,
                "Definition": definition,
            }
        )

    rev_s = _row_get(inc, ["total revenue", "totalrevenue", "revenue"])
    gp_s = _row_get(inc, ["gross profit", "grossprofit"])
    opinc_s = _row_get(inc, ["operating income", "operatingincome"])
    rd_s = _row_get(inc, ["research development", "researchdevelopment", "research & development"])

    rev_yoy = _q_yoy_growth(rev_s) if rev_s is not None else np.nan
    add_scalar("Revenue YoY", rev_yoy, unit="%", update="Quarterly")

    gm = _q_margin_pct(gp_s, rev_s) if gp_s is not None and rev_s is not None else np.nan
    add_scalar("Gross Margin", gm, unit="%", update="Quarterly")

    om = _q_margin_pct(opinc_s, rev_s) if opinc_s is not None and rev_s is not None else np.nan
    add_scalar("Operating Margin", om, unit="%", update="Quarterly")

    ebitda_s = _row_get(inc, ["ebitda"])
    ebitda_margin = _q_margin_pct(ebitda_s, rev_s) if ebitda_s is not None and rev_s is not None else np.nan
    add_scalar("EBITDA Margin", ebitda_margin, unit="%", update="Quarterly", source="Yahoo Finance via yfinance (quarterly_financials)")
    
    cost_ratio = (100.0 - om) if np.isfinite(om) else np.nan
    add_scalar("Total Cost Ratio", cost_ratio, unit="%", update="Quarterly")

    rd_int = _q_margin_pct(rd_s, rev_s) if rd_s is not None and rev_s is not None else np.nan
    add_scalar("R&D Intensity", rd_int, unit="%", update="Quarterly")

    dinv = _days_inventory(bal, inc)
    add_scalar("Days Inventory", dinv, unit="days", update="Quarterly")

    inv_s = _row_get(bal, ["inventory"])
    inv_latest = np.nan
    inv_yoy = np.nan
    if inv_s is not None:
        inv_clean = pd.to_numeric(inv_s, errors="coerce").dropna().sort_index()
        if not inv_clean.empty:
            inv_latest = float(inv_clean.iloc[-1])
        inv_yoy = _q_yoy_growth(inv_s)

    add_scalar("Inventory (latest)", inv_latest, unit="USD", update="Quarterly", source="Yahoo Finance via yfinance (quarterly_balance_sheet)")
    add_scalar("Inventory YoY", inv_yoy, unit="%", update="Quarterly", source="Yahoo Finance via yfinance (quarterly_balance_sheet)")
    
    dso = _ar_days(bal, inc)
    add_scalar("Days Sales Outstanding", dso, unit="days", update="Quarterly")

    ocf = _row_get(cfs, ["total cash from operating activities", "operating cash flow", "totalcashfromoperatingactivities"])
    capex = _row_get(cfs, ["capital expenditures", "capitalexpenditures"])
    if ocf is not None and capex is not None:
        df_cf = pd.concat([ocf.dropna().sort_index(), capex.dropna().sort_index()], axis=1, join="inner")
        fcf_latest = float(df_cf.iloc[-1, 0] + df_cf.iloc[-1, 1]) if not df_cf.empty else np.nan
    else:
        fcf_latest = np.nan
    add_scalar("Free Cash Flow", fcf_latest, unit="USD", update="Quarterly")

    total_debt = _row_get(bal, ["total debt", "totaldebt", "long term debt", "longtermdebt"])
    cash = _row_get(bal, ["cash", "cash and cash equivalents", "cashandcashequivalents"])
    nd_to_ebitda = np.nan
    try:
        if total_debt is not None and cash is not None and ebitda_s is not None:
            d = pd.to_numeric(total_debt, errors="coerce").dropna().sort_index()
            k = pd.to_numeric(cash, errors="coerce").dropna().sort_index()
            e = pd.to_numeric(ebitda_s, errors="coerce").dropna().sort_index()
            df_nd = pd.concat([d, k, e], axis=1, join="inner")
            if not df_nd.empty:
                td = float(df_nd.iloc[-1, 0])
                cs = float(df_nd.iloc[-1, 1])
                eb = float(df_nd.iloc[-1, 2])
                if np.isfinite(td) and np.isfinite(cs) and np.isfinite(eb) and eb != 0:
                    net_debt = td - cs
                    nd_to_ebitda = net_debt / (eb * 4.0)
    except Exception:
        pass
    add_scalar("Net Debt / EBITDA", nd_to_ebitda, unit="x", update="Quarterly", source="Yahoo quarterly statements")
    
    mc = _to_num(info.get("marketCap"))
    fcf_yield = np.nan
    if np.isfinite(fcf_latest) and np.isfinite(mc) and mc > 0:
        fcf_yield = (fcf_latest * 4.0) / mc * 100.0
    add_scalar("FCF Yield (annualized)", fcf_yield, unit="%", update="Quarterly", source="Yahoo cashflow + market cap proxy")
    
    earn = fetch_earnings_dates_yf(sym, limit=24)
    eps_surp = np.nan
    if earn is not None and not earn.empty:
        s_col = None
        for c in ["Surprise(%)", "Surprise (%)", "Surprise %", "Surprise"]:
            if c in earn.columns:
                s_col = c
                break
        if s_col is not None:
            tmp = earn.copy()
            tmp["earnings_date"] = pd.to_datetime(tmp["earnings_date"], errors="coerce", utc=True)
            tmp["surprise_pct"] = pd.to_numeric(tmp[s_col], errors="coerce")
            tmp = tmp.dropna(subset=["earnings_date", "surprise_pct"]).sort_values("earnings_date", ascending=False)
            if not tmp.empty:
                eps_surp = float(tmp["surprise_pct"].iloc[0])

    add_scalar(
        "EPS Surprise",
        eps_surp,
        unit="%",
        update="Quarterly (on earnings)",
        source="Yahoo Finance via yfinance (get_earnings_dates)",
        definition="Most recent available EPS surprise percentage (reported vs estimate).",
    )

    add_scalar("Forward P/E", _to_num(info.get("forwardPE")), unit="x", update="Daily")
    add_scalar("Trailing P/E", _to_num(info.get("trailingPE")), unit="x", update="Daily")
    rg = _to_num(info.get("revenueGrowth"))
    add_scalar("Revenue Growth (Yahoo)", rg * 100.0 if np.isfinite(rg) else np.nan, unit="%", update="Daily")
    eg = _to_num(info.get("earningsGrowth"))
    add_scalar("Earnings Growth (Yahoo)", eg * 100.0 if np.isfinite(eg) else np.nan, unit="%", update="Daily")

    roe = _to_num(info.get("returnOnEquity"))
    add_scalar("Return on Equity", roe * 100.0 if np.isfinite(roe) else np.nan, unit="%", update="Daily")
    add_scalar("Debt/Equity", _to_num(info.get("debtToEquity")), unit="x", update="Daily")

    dy = _to_num(info.get("dividendYield"))
    add_scalar("Dividend Yield", dy * 100.0 if np.isfinite(dy) else np.nan, unit="%", update="Daily")

    pb = _to_num(info.get("priceToBook"))
    add_scalar("Price/Book", pb if np.isfinite(pb) else np.nan, unit="x", update="Daily")

    ev_ebitda = _to_num(info.get("enterpriseToEbitda"))
    add_scalar("EV/EBITDA", ev_ebitda if np.isfinite(ev_ebitda) else np.nan, unit="x", update="Daily")

    bvps = _to_num(info.get("bookValue"))
    add_scalar("Net Asset Value (proxy, BVPS)", bvps if np.isfinite(bvps) else np.nan, unit="USD/share", update="Daily")

    tier1 = fetch_tier1_ratio_best_effort(sym)
    add_scalar(
        "Tier 1 Capital Ratio",
        tier1,
        unit="%",
        update="Quarterly (best-effort)",
        source="SEC XBRL companyfacts (best-effort) / Yahoo (if available)",
        definition="Tier 1 ratio if obtainable from SEC XBRL tags or Yahoo fields (often unavailable).",
    )

    rep = _row_get(cfs, ["repurchase of stock", "repurchaseofstock", "repurchase of capital stock"])
    mc = _to_num(info.get("marketCap"))
    buyback_yield = np.nan
    if rep is not None and mc and np.isfinite(mc) and mc > 0:
        r = rep.dropna().sort_index()
        if not r.empty:
            try:
                buyback_yield = (abs(float(r.iloc[-1])) * 4.0) / float(mc) * 100.0
            except Exception:
                buyback_yield = np.nan
    add_scalar("Buyback Yield (approx)", buyback_yield, unit="%", update="Quarterly", source="Yahoo cashflow + market cap proxy")

    inv_to_sales = np.nan
    try:
        inv_s2 = _row_get(bal, ["inventory"])
        rev_s2 = _row_get(inc, ["total revenue", "totalrevenue", "revenue"])
        if inv_s2 is not None and rev_s2 is not None:
            invc = pd.to_numeric(inv_s2, errors="coerce").dropna().sort_index()
            revc = pd.to_numeric(rev_s2, errors="coerce").dropna().sort_index()
            m = pd.concat([invc, revc], axis=1, join="inner")
            if not m.empty:
                inv_last = float(m.iloc[-1, 0])
                rev_last = float(m.iloc[-1, 1])
                if np.isfinite(inv_last) and np.isfinite(rev_last) and rev_last != 0:
                    inv_to_sales = inv_last / rev_last
    except Exception:
        pass
    add_scalar("Inventory-to-Sales Ratio", inv_to_sales, unit="x", update="Quarterly", source="Yahoo quarterly statements (proxy)")

    backlog_latest = np.nan
    try:
        cik_int = _resolve_cik_for_ticker(sym)
        if cik_int is not None:
            cf = fetch_sec_companyfacts(int(cik_int))
            backlog_latest = _latest_fact_value(cf, [
                "Backlog",
                "OrderBacklog",
                "SalesOrderBacklog",
                "RemainingPerformanceObligation",
                "RevenueRemainingPerformanceObligation",
                "TransactionPriceAllocatedToRemainingPerformanceObligations",
            ])
    except Exception:
        pass
    add_scalar("Order Backlog (latest)", backlog_latest, unit="USD", update="Quarterly (best-effort)", source="SEC XBRL companyfacts (best-effort)")
    
    book_to_bill = np.nan
    try:
        bookings_latest = np.nan
        cik_int = _resolve_cik_for_ticker(sym)
        if cik_int is not None:
            cf = fetch_sec_companyfacts(int(cik_int))
            bookings_latest = _latest_fact_value(cf, ["Orders"])  # rare
            rev_latest = _latest_fact_value(cf, ["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"])
            if np.isfinite(bookings_latest) and np.isfinite(rev_latest) and rev_latest != 0:
                book_to_bill = bookings_latest / rev_latest
    except Exception:
        pass
    add_scalar("Book-to-Bill", book_to_bill, unit="x", update="Quarterly (best-effort)", source="SEC XBRL companyfacts (rare tag)")

    cogs_s = _row_get(inc, ["cost of revenue", "costofrevenue", "cost of goods sold", "costofgoodssold"])
    inv_s2 = _row_get(bal, ["inventory"])
    rev_s2 = _row_get(inc, ["total revenue", "totalrevenue", "revenue"])

    inv_turn = np.nan
    try:
        if cogs_s is not None and inv_s2 is not None:
            cogs = pd.to_numeric(cogs_s, errors="coerce").dropna().sort_index()
            invv = pd.to_numeric(inv_s2, errors="coerce").dropna().sort_index()
            df_it = pd.concat([cogs, invv], axis=1, join="inner")
            if df_it.shape[0] >= 2:
                cogs_last = float(df_it.iloc[-1, 0])
                inv_avg = float(df_it.iloc[-2:, 1].mean())
                if np.isfinite(cogs_last) and np.isfinite(inv_avg) and inv_avg != 0:
                    inv_turn = (abs(cogs_last) * 4.0) / inv_avg
    except Exception:
        pass
    add_scalar("Inventory Turnover (annualized)", inv_turn, unit="x", update="Quarterly", source="Yahoo quarterly (COGS + Inventory)")

    inv_sales = np.nan
    try:
        if inv_s2 is not None and rev_s2 is not None:
            invv = pd.to_numeric(inv_s2, errors="coerce").dropna().sort_index()
            revv = pd.to_numeric(rev_s2, errors="coerce").dropna().sort_index()
            df_is = pd.concat([invv, revv], axis=1, join="inner")
            if not df_is.empty:
                inv_last = float(df_is.iloc[-1, 0])
                rev_last = float(df_is.iloc[-1, 1])
                if np.isfinite(inv_last) and np.isfinite(rev_last) and rev_last != 0:
                    inv_sales = inv_last / rev_last
    except Exception:
        pass
    add_scalar("Inventory / Sales (latest qtr)", inv_sales, unit="x", update="Quarterly", source="Yahoo quarterly (Inventory / Revenue)")
    
    sector_etf_map = {
        "Medical": "XLV",
        "Computer and Technology": "XLK",
        "Finance": "XLF",
        "Utilities": "XLU",
        "Consumer Staples": "XLP",
        "Consumer Discretionary": "XLY",
        "Basic Materials": "XLB",
        "Industrial Products": "XLI",
        "Transportation": "IYT",
        "Retail-Wholesale": "XRT",
        "Auto-Tires-Trucks": "CARZ",
        "Construction": "ITB",
        "Aerospace": "ITA",
        "Business Services": "XLI",
        "Conglomerates": "XLI",
        "Oils-Energy": "XLE",
        "Energy": "XLE",
    }

    driver_series_map = {
        "Consumer Discretionary": [
            ("FRED:UMCSENT", "UMich Consumer Sentiment (FRED: UMCSENT)"),
            ("FRED:CONCCONF", "Conference Board Consumer Confidence (FRED: CONCCONF)"),
            ("FRED:PAYEMS", "Nonfarm Payrolls: Total (FRED: PAYEMS)"),
            ("FRED:UNRATE", "Unemployment Rate (FRED: UNRATE)"),
            ("FRED:CES0600000003", "Avg Hourly Earnings: Prod & Nonsup (FRED: CES0600000003)"),
            ("FRED:DGS10", "10Y Treasury Yield (FRED: DGS10)"),
            ("FRED:TERMCBAUTO48NS", "48-Month New Car Loan Rate (FRED: TERMCBAUTO48NS)"),
            ("FRED:SUBLPDRCSCLGNQ", "SLOOS: Tightening Standards — Credit Card Loans (FRED: SUBLPDRCSCLGNQ)"),
            ("FRED:FEDFUNDS", "Effective Fed Funds Rate (FRED: FEDFUNDS)"),
        ],
        "Consumer Staples": [
            ("ZW=F", "Wheat Futures (Yahoo: ZW=F)"),
            ("ZC=F", "Corn Futures (Yahoo: ZC=F)"),
            ("SB=F", "Sugar #11 Futures (Yahoo: SB=F)"),
            ("KC=F", "Coffee Futures (Yahoo: KC=F)"),
        ],
        "Retail-Wholesale": [
            ("FRED:RSAFS", "Advance Retail Sales: Retail & Food Services (FRED: RSAFS)"),
            ("FRED:RETAILSMSA", "Retailers Sales (FRED: RETAILSMSA)"),
            ("FRED:RETAILIRSA", "Retailers: Inventories to Sales Ratio (FRED: RETAILIRSA)"),
            ("FRED:RETAILIMSA", "Retailers Inventories (FRED: RETAILIMSA)"),
            ("FRED:CONCCONF", "Conference Board Consumer Confidence Index (FRED: CONCCONF)"),
            ("FRED:UMCSENT", "University of Michigan Consumer Sentiment Index (FRED: UMCSENT)"),
            ("FRED:MDSIM2MEUSN", "Visa Spending Momentum Index (FRED: MDSIM2MEUSN)"),
            ("FRED:ECOMPCTSA", "E-commerce as % of total retail sales (FRED: ECOMPCTSA)"),
            ("FRED:ECOMSA", "U.S. Retail E-Commerce Sales (FRED: ECOMSA)"),
        ],
        "Industrial Products": [
            ("FRED:NAPMNOI", "ISM PMI: New Orders (FRED: NAPMNOI)"),
            ("FRED:INDPRO", "Industrial Production Index (FRED: INDPRO)"),
            ("FRED:NEWORDER", "Nondefense Capital Goods New Orders ex Aircraft (FRED: NEWORDER)"),
        ],
        "Basic Materials": [
            ("FRED:DCOILWTICO", "WTI Crude (FRED: DCOILWTICO)"),
            ("FRED:DHHNGSP", "Henry Hub Natural Gas (FRED: DHHNGSP)"),
            ("GC=F", "Gold Futures (Yahoo)"),
            ("FRED:PCOPPUSDM", "Copper (FRED: PCOPPUSDM)"),
            ("FRED:PALUMUSDM", "Aluminum (FRED: PALUMUSDM)"),
            ("FRED:DTWEXBGS", "Broad USD Index (FRED: DTWEXBGS)"),
        ],
        "Computer and Technology": [
            ("FRED:IPG3344S", "Semiconductor Output (US IP, NAICS 3344)"),
            ("FRED:CAPUTLG3344S", "Semiconductor Capacity Utilization (NAICS 3344)"),
        ],
        "Energy": [
            ("CL=F", "WTI Crude (Yahoo)"),
            ("BZ=F", "Brent Crude (Yahoo)"),
            ("RB=F", "RBOB Gasoline (Yahoo)"),
            ("HO=F", "ULSD/Heating Oil (Yahoo)"),
            ("FRED:OVXCLS", "OVX (Crude Oil ETF Volatility, FRED: OVXCLS)"),
            ("EIA:WPULEUS3", "US Refinery Utilization (EIA: WPULEUS3)"),
            ("FRED:IPN213111N", "IP: Drilling Oil & Gas Wells (FRED: IPN213111N)"),
            ("KRBN", "Carbon Allowances (KRBN ETF, Yahoo)"),
        ],
        "Oils-Energy": [
            ("CL=F", "WTI Crude (Yahoo)"),
            ("BZ=F", "Brent Crude (Yahoo)"),
            ("RB=F", "RBOB Gasoline (Yahoo)"),
            ("HO=F", "ULSD/Heating Oil (Yahoo)"),
            ("FRED:OVXCLS", "OVX (Crude Oil ETF Volatility, FRED: OVXCLS)"),
            ("EIA:WPULEUS3", "US Refinery Utilization (EIA: WPULEUS3)"),
            ("FRED:IPN213111N", "IP: Drilling Oil & Gas Wells (FRED: IPN213111N)"),
            ("KRBN", "Carbon Allowances (KRBN ETF, Yahoo)"),
        ],
        "Transportation": [
            ("FRED:TSIFRGHTC", "Freight Transportation Services Index (TSI, FRED: TSIFRGHTC)"),
            ("FRED:FRGSHPUSM649NCIS", "Cass Freight Index — Shipments (FRED: FRGSHPUSM649NCIS)"),
            ("FRED:FRGEXPUSM649NCIS", "Cass Freight Index — Expenditures (FRED: FRGEXPUSM649NCIS)"),
            ("FRED:ISRATIO", "Total Business: Inventories-to-Sales Ratio (FRED: ISRATIO)"),
            ("FRED:DCOILWTICO", "WTI Crude (FRED: DCOILWTICO)"),
            ("FRED:WDFUELUSGULF", "ULSD Diesel Spot (US Gulf Coast, FRED: WDFUELUSGULF)"),
            ("HO=F", "ULSD Futures (NY Harbor, Yahoo: HO=F)"),
            ("FRED:WJFUELUSGULF", "Jet Fuel Spot (US Gulf Coast, FRED: WJFUELUSGULF)"),
            ("FRED:NAPMSDI", "ISM PMI: Supplier Deliveries Index (FRED: NAPMSDI)"),
            ("^DJT", "Dow Jones Transportation Average (Yahoo: ^DJT)"),
            ("^GSPC", "S&P 500 (Yahoo: ^GSPC)"),
            ("BDRY", "Dry Bulk Proxy ETF (BDRY, proxy for Baltic Dry direction)"),
        ],
        "Aerospace": [
            ("FRED:RPMD11", "US Airlines: Revenue Passenger Miles (FRED: RPMD11)"),
            ("FRED:G160461A027NBEA", "Federal: National Defense Current Expenditures (FRED: G160461A027NBEA)"),
            ("FRED:A191RL1Q225SBEA", "Real Federal National Defense Consumption+GI (FRED: A191RL1Q225SBEA)"),
        ],
        "Construction": [
            ("FRED:HOUST", "Housing Starts (FRED: HOUST)"),
            ("FRED:PERMIT", "Building Permits (FRED: PERMIT)"),
            ("FRED:TTLCONS", "Construction Spending: Total (FRED: TTLCONS)"),
            ("FRED:WPUFD4", "PPI: Final Demand Construction (FRED: WPUFD4)"),
            ("FRED:WPU1017", "PPI: Steel Mill Products (FRED: WPU1017)"),
            ("LBS=F", "Lumber Futures (Yahoo: LBS=F)"),
            ("FRED:WDFUELUSGULF", "ULSD Diesel Spot (US Gulf Coast, FRED: WDFUELUSGULF)"),
            ("FRED:CES2000000001", "All Employees: Construction (FRED: CES2000000001)"),
            ("FRED:JTS2300JOL", "JOLTS: Job Openings — Construction (FRED: JTS2300JOL)"),
            ("FRED:JTS2300HIL", "JOLTS: Hires — Construction (FRED: JTS2300HIL)"),
        ],
        "Finance": [
            ("FRED:T10Y2Y", "10Y–2Y Treasury Spread"),
            ("FRED:TOTLL", "Bank Loans & Leases (H.8)"),
            ("FRED:DRALACBS", "Delinquency Rate (All Loans)"),
            ("FRED:CORALACBS", "Charge-Off Rate (All Loans)"),
            ("FRED:DRCCLACBS", "Credit Card Delinquency Rate"),
            ("FRED:CORCCACBS", "Credit Card Charge-Off Rate"),
        ],
        "Auto-Tires-Trucks": [
            ("FRED:TOTALSA", "Auto Sales SAAR (Total Vehicle Sales)"),
            ("FRED:AISRSA", "Auto Inventory/Sales Ratio (Months of Supply)"),
            ("FRED:CUSR0000SETA01", "CPI: New Vehicles"),
            ("FRED:CES0500000003", "Avg Hourly Earnings (Total Private)"),
            ("FRED:TERMCBAUTO48NS", "48-Month New Car Loan Rate"),
            ("FRED:TB3MS", "3-Month T-Bill Rate"),
            ("FRED:WPU1017", "PPI: Steel Mill Products"),
            ("FRED:PALUMUSDM", "Global Aluminum Price (USD/mt)"),
            ("FRED:PCOPPUSDM", "Global Copper Price (USD/mt)"),
            ("PA=F", "Palladium Futures"),
            ("FRED:PRUBBUSDM", "Global Rubber Price (US cents/lb)"),
        ],
    }

    return {
        "scalars": pd.DataFrame(scalars),
        "sector_etf_map": sector_etf_map,
        "driver_series_map": driver_series_map,
    }

@st.cache_data(ttl=60 * 60)
def compute_pe_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    P/E History (proxy): daily price / trailing EPS (EPS is a point-in-time Yahoo field).
    Returns df(date,value).
    """
    sym = normalize_yf_ticker(ticker)

    # trailing EPS from Yahoo info
    info = fetch_info_one(sym) or {}
    trailing_eps = _to_num(info.get("trailingEps"))
    if not np.isfinite(trailing_eps) or trailing_eps == 0:
        return pd.DataFrame()

    def _to_df(px: pd.DataFrame) -> pd.DataFrame:
        if px is None or px.empty:
            return pd.DataFrame()
        col = "Close" if "Close" in px.columns else ("Adj Close" if "Adj Close" in px.columns else None)
        if col is None:
            return pd.DataFrame()
        out = px[[col]].dropna().reset_index()
        if out.empty:
            return pd.DataFrame()
        out.columns = ["date", "value"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["date", "value"]).sort_values("date")
        return out

    # Use your last-good-series memory key (optional but consistent with the rest of your app)
    cache_key = f"PEHIST::{sym}::{period}"

    # 1) Try yf.download
    try:
        px1 = yf.download(sym, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
        df1 = _to_df(px1)
        if not df1.empty:
            df1["value"] = df1["value"] / trailing_eps
            df1 = df1.dropna(subset=["date", "value"]).sort_values("date")
            return _remember_last_good_series(cache_key, df1)
    except Exception:
        pass

    # 2) Try Ticker().history
    try:
        t = yf.Ticker(sym)
        px2 = t.history(period=period, interval="1d", auto_adjust=True)
        df2 = _to_df(px2)
        if not df2.empty:
            df2["value"] = df2["value"] / trailing_eps
            df2 = df2.dropna(subset=["date", "value"]).sort_values("date")
            return _remember_last_good_series(cache_key, df2)
    except Exception:
        pass

    # fallback: last good value if any
    prev = _remember_last_good_series(cache_key, None)
    return prev if isinstance(prev, pd.DataFrame) else pd.DataFrame()

def build_indicator_series(sector_name_exact: str, ticker: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    payload = fetch_indicator_bundle(ticker)
    scalars = payload["scalars"].copy()
    if scalars is not None and not scalars.empty and "Indicator" in scalars.columns:
        scalars["Indicator"] = scalars["Indicator"].map(strip_trailing_parens)

    series: Dict[str, pd.DataFrame] = {}

    def _to_df(px: pd.DataFrame) -> Optional[pd.DataFrame]:
        if px is None or px.empty:
            return None
        col = "Close" if "Close" in px.columns else ("Adj Close" if "Adj Close" in px.columns else None)
        if col is None:
            return None
        out = px[[col]].dropna().reset_index()
        if out.empty:
            return None
        out.columns = ["date", "value"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["date", "value"]).sort_values("date")
        return out if not out.empty else None

    def _yf_series(symbol: str, period="5y") -> Optional[pd.DataFrame]:
        sym = (symbol or "").strip()
        if not sym:
            return None
        cache_key = f"YF::{sym}::{period}"

        try:
            px1 = yf.download(sym, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
            out1 = _to_df(px1)
            if out1 is not None and not out1.empty:
                return _remember_last_good_series(cache_key, out1)
        except Exception:
            pass

        try:
            t = yf.Ticker(sym)
            px2 = t.history(period=period, interval="1d", auto_adjust=True)
            out2 = _to_df(px2)
            if out2 is not None and not out2.empty:
                return _remember_last_good_series(cache_key, out2)
        except Exception:
            pass

        return _remember_last_good_series(cache_key, None)
    
    def _any_series(symbol: str) -> Optional[pd.DataFrame]:
        s = (symbol or "").strip()
        if not s:
            return None

        if s.upper().startswith("TE:"):
            url = s.split(":", 1)[1].strip()
            try:
                # Expect: https://tradingeconomics.com/<country>/<indicator>
                m = re.search(r"tradingeconomics\.com/([^/]+)/([^/?#]+)", url, flags=re.I)
                if m:
                    country_slug = m.group(1).strip().lower()
                    indicator_slug = m.group(2).strip().lower()
                    df_api = fetch_tradingeconomics_hist_via_api(country_slug, indicator_slug)
                    if df_api is not None and not df_api.empty:
                        return df_api[["date", "value"]].copy()
            except Exception:
                pass
            df = fetch_tradingeconomics_hist_from_page(url)
            if df is None or df.empty:
                return None
            return df[["date", "value"]].copy()

        if s.upper().startswith("FRED:"):
            sid = s.split(":", 1)[1].strip()
            df = fetch_fred_series(sid)
            if df is None or df.empty:
                return None
            return df[["date", "value"]].copy()

        if s.upper().startswith("EIA:"):
            code = s.split(":", 1)[1].strip()
            df = fetch_eia_pet_weekly_series(code)
            if df is None or df.empty:
                return None
            return df[["date", "value"]].copy()
        
        out = _yf_series(s)
        if out is not None and not out.empty:
            return out

        # Stooq fallback (fixes broken/blank futures)
        stooq_map = {
            "CL=F": "cl.f",
            "NG=F": "ng.f",
            "GC=F": "gc.f",
            "SI=F": "si.f",
            "HG=F": "hg.f",
            "ALI=F": "al.f",  # best-effort
        }
        st_sym = stooq_map.get(s.upper())
        if st_sym:
            df2 = fetch_stooq_daily(st_sym)
            if df2 is not None and not df2.empty:
                _remember_last_good_series(f"YF::{s}::5y", df2)
                return df2

        return None

    etf_map = payload.get("sector_etf_map", {})
    etf = etf_map.get(sector_name_exact)
    if etf:
        ser = _any_series(etf)
        if ser is not None and not ser.empty:
            series[f"{sector_name_exact} Benchmark ETF ({etf})"] = ser

    driver_map = payload.get("driver_series_map", {})
    for sym, label in driver_map.get(sector_name_exact, []):
        ser = _any_series(sym)
        if ser is not None and not ser.empty:
            series[label] = ser

    if sector_name_exact == "Transportation":
        # DJTA / S&P 500 ratio (indexed to 100 at first common observation)
        djt = series.get("Dow Jones Transportation Average (Yahoo: ^DJT)")
        spx = series.get("S&P 500 (Yahoo: ^GSPC)")

        if (
            djt is not None and not djt.empty and "value" in djt.columns
            and spx is not None and not spx.empty and "value" in spx.columns
        ):
            a = djt.copy()
            b = spx.copy()
            a["date"] = pd.to_datetime(a["date"], errors="coerce")
            b["date"] = pd.to_datetime(b["date"], errors="coerce")
            a["djt"] = pd.to_numeric(a["value"], errors="coerce")
            b["spx"] = pd.to_numeric(b["value"], errors="coerce")

            m = pd.merge(a[["date", "djt"]], b[["date", "spx"]], on="date", how="inner").dropna()
            if not m.empty:
                m["ratio"] = m["djt"] / m["spx"].replace(0, np.nan)
                m = m.dropna(subset=["ratio"]).sort_values("date")
                if not m.empty:
                    base = float(m["ratio"].iloc[0])
                    if np.isfinite(base) and base != 0:
                        m["value"] = (m["ratio"] / base) * 100.0
                        out = m[["date", "value"]].copy()
                        series["DJTA / S&P 500 Ratio (indexed)"] = out
    
    if sector_name_exact == "Auto-Tires-Trucks":
        # Days supply ≈ (months of supply) * 30.4
        mos = series.get("Auto Inventory/Sales Ratio (Months of Supply)")
        if mos is not None and not mos.empty and "value" in mos.columns:
            ds = mos.copy()
            ds["value"] = pd.to_numeric(ds["value"], errors="coerce") * 30.4
            ds = ds.dropna(subset=["date", "value"]).sort_values("date")
            if not ds.empty:
                series["Days Supply (derived, days)"] = ds

        # Affordability proxy: (New Vehicle CPI / Avg Hourly Earnings), rebased to 100 at first obs
        cpi = series.get("CPI: New Vehicles")
        wage = series.get("Avg Hourly Earnings (Total Private)")
        if (
            cpi is not None and not cpi.empty and "value" in cpi.columns
            and wage is not None and not wage.empty and "value" in wage.columns
        ):
            a = cpi.copy()
            b = wage.copy()
            a["date"] = pd.to_datetime(a["date"], errors="coerce")
            b["date"] = pd.to_datetime(b["date"], errors="coerce")
            a["cpi"] = pd.to_numeric(a["value"], errors="coerce")
            b["wage"] = pd.to_numeric(b["value"], errors="coerce")

            m = pd.merge(a[["date", "cpi"]], b[["date", "wage"]], on="date", how="inner").dropna()
            if not m.empty:
                m["ratio"] = m["cpi"] / m["wage"].replace(0, np.nan)
                m = m.dropna(subset=["ratio"]).sort_values("date")
                if not m.empty:
                    base = float(m["ratio"].iloc[0])
                    if np.isfinite(base) and base != 0:
                        m["value"] = (m["ratio"] / base) * 100.0
                        out = m[["date", "value"]].copy()
                        series["Affordability Proxy (CPI New Vehicles / Wages, indexed)"] = out

    if sector_name_exact in ("Energy", "Oils-Energy"):
        # Brent–WTI spread = Brent - WTI
        brent = series.get("Brent Crude (Yahoo)")
        wti = series.get("WTI Crude (Yahoo)")
        if brent is not None and not brent.empty and wti is not None and not wti.empty:
            a = brent.rename(columns={"value": "brent"})
            b = wti.rename(columns={"value": "wti"})
            m = pd.merge(a[["date", "brent"]], b[["date", "wti"]], on="date", how="inner").dropna()
            if not m.empty:
                m["value"] = pd.to_numeric(m["brent"], errors="coerce") - pd.to_numeric(m["wti"], errors="coerce")
                out = m.dropna(subset=["value"])[["date", "value"]].sort_values("date")
                if not out.empty:
                    series["Brent–WTI Spread (Brent - WTI)"] = out

        # 3-2-1 crack (approx) using front-month futures:
        # crack = (2*RBOB + 1*HO - 3*WTI) / 3
        rb = series.get("RBOB Gasoline (Yahoo)")
        ho = series.get("ULSD/Heating Oil (Yahoo)")
        if rb is not None and not rb.empty and ho is not None and not ho.empty and wti is not None and not wti.empty:
            a = rb.rename(columns={"value": "rb"})
            b = ho.rename(columns={"value": "ho"})
            c = wti.rename(columns={"value": "wti"})
            m = a.merge(b[["date", "ho"]], on="date", how="inner").merge(c[["date", "wti"]], on="date", how="inner")
            m = m.dropna()
            if not m.empty:
                m["value"] = (2.0 * pd.to_numeric(m["rb"], errors="coerce") + 1.0 * pd.to_numeric(m["ho"], errors="coerce") - 3.0 * pd.to_numeric(m["wti"], errors="coerce")) / 3.0
                out = m.dropna(subset=["value"])[["date", "value"]].sort_values("date")
                if not out.empty:
                    series["Crack Spread (3-2-1 approx)"] = out

        # Optional: show EIA monthly rig count as a series too
        rigs = fetch_eia_ng_monthly_rig_count()
        if rigs is not None and not rigs.empty:
            series["US Rotary Rig Count (EIA, monthly)"] = rigs

    # ============================================
    # Hide helper / undesired series from charts
    # ============================================
    hidden = set()

    # Transportation hidden series
    if sector_name_exact == "Transportation":
        hidden.update({
            "Dow Jones Transportation Average (Yahoo: ^DJT)",
            "S&P 500 (Yahoo: ^GSPC)",
            "ULSD Futures (NY Harbor, Yahoo: HO=F)",
        })

    # Oils-Energy / Energy hidden series (used only for spreads)
    if sector_name_exact in ("Energy", "Oils-Energy"):
        hidden.update({
            "WTI Crude (Yahoo)",
            "Brent Crude (Yahoo)",
            "RBOB Gasoline (Yahoo)",
            "ULSD/Heating Oil (Yahoo)",
        })

    if hidden:
        series = {k: v for k, v in series.items() if k not in hidden}
    
    return scalars, series

# =========================
# Technical indicators
# =========================
def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    return mid, upper, lower, width

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(window).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(high, low, close)
    atr_ = tr.rolling(window).mean()

    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(window).sum() / atr_.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(window).sum() / atr_.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_ = dx.rolling(window).mean()
    return plus_di, minus_di, adx_

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    v = volume.fillna(0.0)
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * v).cumsum()

def roc(close: pd.Series, window: int = 10) -> pd.Series:
    return (close / close.shift(window) - 1.0) * 100.0

def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(window).mean()
    mad = (tp - sma).abs().rolling(window).mean()
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def vwap_daily(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    tp = (high + low + close) / 3.0
    v = volume.fillna(0.0)
    return (tp * v).cumsum() / v.cumsum().replace(0, np.nan)

# =========================
# Technical scoring
# =========================
def score_ema_trend(close: pd.Series, ema_s: pd.Series) -> float:
    c = safe_last(close)
    e = safe_last(ema_s)
    if not np.isfinite(c) or not np.isfinite(e):
        return 0.0
    slope = safe_last(ema_s.diff())
    if c > e and slope >= 0:
        return 1.0
    if c < e and slope <= 0:
        return -1.0
    return 0.0

def score_rsi_value(rsi: pd.Series) -> float:
    v = safe_last(rsi)
    if not np.isfinite(v):
        return 0.0
    if v <= 30:
        return 1.0
    if v >= 70:
        return -1.0
    dv = safe_last(rsi.diff())
    if np.isfinite(dv) and dv > 0:
        return 0.25
    if np.isfinite(dv) and dv < 0:
        return -0.25
    return 0.0

def score_macd_value(macd_line: pd.Series, signal_line: pd.Series, hist: pd.Series) -> float:
    m = safe_last(macd_line)
    s = safe_last(signal_line)
    h = safe_last(hist)
    if not np.isfinite(m) or not np.isfinite(s) or not np.isfinite(h):
        return 0.0
    cross = 1.0 if m > s else (-1.0 if m < s else 0.0)
    accel = safe_last(hist.diff())
    bonus = 0.25 if np.isfinite(accel) and accel > 0 else (-0.25 if np.isfinite(accel) and accel < 0 else 0.0)
    return float(np.clip(cross + bonus, -1.0, 1.0))

def score_kdj_value(k: pd.Series, d: pd.Series) -> float:
    kv, dv = safe_last(k), safe_last(d)
    if not np.isfinite(kv) or not np.isfinite(dv):
        return 0.0
    return 0.25 if kv > dv else (-0.25 if kv < dv else 0.0)

def score_boll_width(bb_width: pd.Series) -> float:
    w = safe_last(bb_width)
    if not np.isfinite(w):
        return 0.0
    return float(np.clip((0.05 - w) / 0.05, -1.0, 1.0))

def score_adx_trend(adx_s: pd.Series, plus_di: pd.Series, minus_di: pd.Series) -> float:
    a = safe_last(adx_s)
    p = safe_last(plus_di)
    m = safe_last(minus_di)
    if not np.isfinite(a) or not np.isfinite(p) or not np.isfinite(m):
        return 0.0
    if a < 18:
        return 0.0
    return 0.75 if p > m else (-0.75 if m > p else 0.0)

def score_obv(obv_s: pd.Series) -> float:
    if obv_s is None or obv_s.dropna().shape[0] < 20:
        return 0.0
    slope = safe_last(obv_s.diff().rolling(10).mean())
    if not np.isfinite(slope):
        return 0.0
    return 0.5 if slope > 0 else (-0.5 if slope < 0 else 0.0)

def score_roc(roc_s: pd.Series) -> float:
    r = safe_last(roc_s)
    if not np.isfinite(r):
        return 0.0
    return float(np.clip(r / 10.0, -1.0, 1.0))

def score_cci(cci_s: pd.Series) -> float:
    c = safe_last(cci_s)
    if not np.isfinite(c):
        return 0.0
    if c > 100:
        return 0.5
    if c < -100:
        return -0.5
    return 0.0

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
uni, key_cols_present, _ = normalize_universe_keep_original(uni_raw)

tickers_all = sorted([t for t in uni["ticker_norm"].dropna().unique().tolist() if t])
ret_map = fetch_returns_since(date_sel, tickers_all)

st.subheader("Opportunity Set")
if key_cols_present:
    show_uni = uni[key_cols_present].copy()
    cols = list(show_uni.columns)
    ticker_col = None
    for c in cols:
        if str(c).lower() in ["ticker", "symbol"]:
            ticker_col = c

    returns = (
        show_uni[ticker_col].astype(str).str.upper().str.strip().map(ret_map) * 100
        if ticker_col is not None
        else uni["ticker_norm"].astype(str).str.upper().str.strip().map(ret_map) * 100
    )
    show_uni.insert(0, "__Return", returns)

    if ticker_col is not None:
        cols2 = list(show_uni.columns)
        cols2.remove("__Return")
        try:
            tpos = cols2.index(ticker_col)
            insert_pos = tpos + 1
        except Exception:
            insert_pos = 1
        cols2.insert(insert_pos, "__Return")
        show_uni = show_uni[cols2]

    show_uni = show_uni.rename(columns={"__Return": "Return"})
    styled_uni = show_uni.style.map(color_return, subset=["Return"])

    st.dataframe(
        styled_uni,
        use_container_width=True,
        height=320,
        column_config={"Return": st.column_config.NumberColumn("Return", format="%.2f%%")},
    )
else:
    st.dataframe(uni.iloc[:, :12], use_container_width=True, height=320)

ticker_sel = st.selectbox("Select ticker", tickers_all, index=0 if tickers_all else None)
if not ticker_sel:
    st.stop()

row = uni.loc[uni["ticker_norm"] == ticker_sel].head(1)
sector_exact = str(row["sector_norm"].iloc[0]) if not row.empty else "Unknown"
industry = str(row["industry_norm"].iloc[0]) if not row.empty else ""
st.caption(f"Selected: **{ticker_sel}** | Sector: **{sector_exact}** | Industry: **{industry}**")

info = fetch_info_one(normalize_yf_ticker(ticker_sel))
co_name = (info or {}).get("shortName", "") or (info or {}).get("longName", "")

# =========================
# Indicators (Sector vs Stock)
# =========================
st.subheader("Fundamental Indicators")

scalars_df, series_map = build_indicator_series(sector_exact, ticker_sel)
scalars_df = scalars_df if isinstance(scalars_df, pd.DataFrame) else pd.DataFrame()

tab_sector, tab_stock = st.tabs(["Sector Indicators", "Stock Indicators"])

with tab_sector:
    if not series_map:
        st.info("No sector indicator series available for this sector.")
    else:
        cols = st.columns(2)
        i = 0
        for name, ser in series_map.items():
            with cols[i % 2]:
                st.write(f"**{name}**")

                if ser is None or ser.empty or ("value" not in ser.columns) or ser["value"].dropna().empty:
                    st.warning(f"{name}: no data returned (provider temporarily empty/blocked).")
                    i += 1
                    continue

                ser2 = ser.dropna(subset=["date", "value"]).sort_values("date").tail(3000)
                if ser2.empty:
                    st.warning(f"{name}: no usable data points after cleaning.")
                    i += 1
                    continue

                fig = px.line(ser2, x="date", y="value")
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True, key=f"sector_series_{sector_exact}_{i}")
            i += 1

with tab_stock:
    def _scalar_value(indicator: str) -> float:
        if scalars_df.empty or "Indicator" not in scalars_df.columns:
            return np.nan
        hit = scalars_df[scalars_df["Indicator"].astype(str).str.lower().eq(indicator.lower())]
        if hit.empty:
            return np.nan
        return pd.to_numeric(hit["Value"].iloc[0], errors="coerce")

    def _fmt_value(ind: str, kind: str) -> str:
        v = _scalar_value(ind)
        if not np.isfinite(v):
            return "N/A"
        if kind == "pct":
            return f"{v:.2f}%"
        if kind == "x":
            return f"{v:.2f}x"
        if kind == "usd":
            # show as dollars, no decimals by default
            return f"${v:,.0f}"
        if kind == "num":
            return f"{v:.2f}"
        if kind == "days":
            return f"{v:.0f}d"
        return str(v)

    def sector_kpi_config(sector: str) -> List[Tuple[str, str, str]]:
        """
        Returns list of (label, indicator_name_in_scalars_df, kind)
        kind in: pct, x, usd, num, days
        """
        s = (sector or "").strip()

        if s == "Aerospace":
            return [
                ("Book-to-Bill", "Book-to-Bill", "x"),
                ("Order Backlog", "Order Backlog (latest)", "usd"),
                ("Operating Margin", "Operating Margin", "pct"),
                ("Net Debt / EBITDA", "Net Debt / EBITDA", "x"),
            ]
        
        if s == "Consumer Discretionary":
            return [
                ("Revenue YoY", "Revenue YoY", "pct"),
                ("Gross Margin", "Gross Margin", "pct"),
                ("Inventory Turnover", "Inventory Turnover (annualized)", "x"),
                ("EV/EBITDA", "EV/EBITDA", "x"),
            ]
        
        if s == "Construction":
            return [
                ("Book-to-Bill", "Book-to-Bill", "x"),
                ("Order Backlog", "Order Backlog (latest)", "usd"),
                ("Operating Margin", "Operating Margin", "pct"),
                ("EV/EBITDA", "EV/EBITDA", "x"),
            ]
        
        if s == "Consumer Staples":
            return [
                ("Gross Margin", "Gross Margin", "pct"),
                ("Trailing P/E", "Trailing P/E", "x"),
                ("Dividend Yield", "Dividend Yield", "pct"),
                ("EV/EBITDA", "EV/EBITDA", "x"),
            ]
        
        if s == "Retail-Wholesale":
            return [
                ("Gross Margin", "Gross Margin", "pct"),
                ("Inventory Turnover", "Inventory Turnover (annualized)", "x"),
                ("Inv/Sales", "Inventory / Sales (latest qtr)", "x"),
                ("EV/EBITDA", "EV/EBITDA", "x"),
            ]
        
        if s == "Industrial Products":
            return [
                ("Book-to-Bill", "Book-to-Bill", "x"),
                ("Order Backlog", "Order Backlog (latest)", "usd"),
                ("Inventory/Sales", "Inventory-to-Sales Ratio", "x"),
                ("EV/EBITDA", "EV/EBITDA", "x"),
            ]
        
        if s == "Basic Materials":
            # REMOVE: Revenue YoY / Operating Margin / Total Cost Ratio / EPS Surprise
            # Also no "Valuation (Basic Materials)"
            return [
                ("P/B", "Price/Book", "x"),
                ("EV/EBITDA", "EV/EBITDA", "x"),
                ("Days Inventory", "Days Inventory", "days"),
                ("Days Sales Outstanding", "Days Sales Outstanding", "days"),
            ]
        
        if s == "Medical":
            # Keep Medical-specific: use more relevant KPIs (and keep the Medical extra sections below)
            return [
                ("Revenue YoY", "Revenue YoY", "pct"),
                ("R&D Intensity", "R&D Intensity", "pct"),
                ("Gross Margin", "Gross Margin", "pct"),
                ("EPS Surprise", "EPS Surprise", "pct"),
            ]

        if s == "Business Services":
            return [
                ("Revenue YoY", "Revenue YoY", "pct"),
                ("EBITDA Margin", "EBITDA Margin", "pct"),   # we'll add this scalar next
                ("Operating Margin", "Operating Margin", "pct"),
                ("EV/EBITDA", "EV/EBITDA", "x"),
            ]
        
        if s == "Computer and Technology":
            return [
                ("Revenue YoY", "Revenue YoY", "pct"),
                ("FCF Yield", "FCF Yield (annualized)", "pct"),
                ("Inventory YoY", "Inventory YoY", "pct"),
                ("EV/EBITDA", "EV/EBITDA", "x"),
            ]

        if s == "Auto-Tires-Trucks":
            return [
                ("Days Inventory", "Days Inventory", "days"),
                ("R&D Intensity", "R&D Intensity", "pct"),
                ("Inventory YoY", "Inventory YoY", "pct"),
                ("EV/EBITDA", "EV/EBITDA", "x"),
            ]
        
        if s == "Transportation":
            return [
                ("Gross Margin", "Gross Margin", "pct"),
                ("Operating Margin", "Operating Margin", "pct"),
                ("FCF Yield", "FCF Yield (annualized)", "pct"),
                ("Days Sales Outstanding", "Days Sales Outstanding", "days"),
            ]

        if s in ("Oils-Energy", "Energy"):
            return [
                ("Net Debt/EBITDA", "Net Debt / EBITDA", "x"),
                ("EV/EBITDA", "EV/EBITDA", "x"),
                ("FCF Yield", "FCF Yield (annualized)", "pct"),
                ("Dividend Yield", "Dividend Yield", "pct"),
            ]
        
        # Default for other non-Finance sectors (your current behavior)
        return [
            ("Revenue YoY", "Revenue YoY", "pct"),
            ("Operating Margin", "Operating Margin", "pct"),
            ("Total Cost Ratio", "Total Cost Ratio", "pct"),
            ("EPS Surprise", "EPS Surprise", "pct"),
        ]
    
    if sector_exact == "Finance":
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Tier 1 Ratio", f"{_scalar_value('Tier 1 Capital Ratio'):.2f}%" if np.isfinite(_scalar_value("Tier 1 Capital Ratio")) else "N/A")
        with c2:
            st.metric("Dividend Yield", f"{_scalar_value('Dividend Yield'):.2f}%" if np.isfinite(_scalar_value("Dividend Yield")) else "N/A")
        with c3:
            st.metric("Buyback Yield (approx)", f"{_scalar_value('Buyback Yield (approx)'):.2f}%" if np.isfinite(_scalar_value("Buyback Yield (approx)")) else "N/A")
        with c4:
            st.metric("P/B", f"{_scalar_value('Price/Book'):.2f}x" if np.isfinite(_scalar_value("Price/Book")) else "N/A")
        with c5:
            st.metric("ROE", f"{_scalar_value('Return on Equity'):.2f}%" if np.isfinite(_scalar_value("Return on Equity")) else "N/A")

        st.markdown("#### Buyback Announcements (News)")
        news_window_label_fin = st.selectbox("Buyback news window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="fin_buyback_news_window")
        days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
        fin_days = days_map.get(news_window_label_fin, 7)

        q_buyback = (
            f'({ticker_sel} OR "{co_name}") '
            f'("share repurchase" OR "share buyback" OR "repurchase program" OR '
            f'"repurchase authorization" OR "accelerated share repurchase" OR ASR OR "buyback program")'
        )
        bb_news = fetch_google_news_rss_query(q_buyback, days=fin_days)
        if bb_news is None or bb_news.empty:
            st.info("No recent buyback-announcement RSS headlines found.")
        else:
            for _, n in bb_news.head(20).iterrows():
                t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                title = str(n.get("title", ""))
                link = str(n.get("link", ""))
                src = str(n.get("source", ""))
                st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_")

    else:
        kpis = sector_kpi_config(sector_exact)
        ncols = max(1, min(4, len(kpis)))
        cols = st.columns(ncols)

        for i, (label, indicator, kind) in enumerate(kpis):
            with cols[i % ncols]:
                st.metric(label, _fmt_value(indicator, kind))

        if sector_exact == "Aerospace":
            st.markdown("#### Aerospace: Backlog / Book-to-Bill / Orders / Budgets / Air Traffic / Production Rates")

            aero_window = st.selectbox(
                "Aerospace headlines window", ["1w", "2w", "1m", "2m", "3m"],
                index=0, key="aero_headlines_window"
            )
            days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
            aero_days = days_map.get(aero_window, 7)

            queries = {
                "Backlog (company filings / quarterly reports)": (
                    f'({ticker_sel} OR "{co_name}") AND (backlog OR "order backlog" OR "funded backlog" OR "unfunded backlog")'
                ),
                "Book-to-bill / bookings / order intake": (
                    f'({ticker_sel} OR "{co_name}") AND ("book-to-bill" OR "book to bill" OR bookings OR "order intake")'
                ),
                "Major air shows (Paris / Farnborough / Dubai) — order totals": (
                    '("Paris Air Show" OR "Farnborough Airshow" OR "Dubai Airshow") AND (orders OR "order tally" OR "order total" OR commitments)'
                ),
                "Big contract wins (DoD / primes / missile defense / space)": (
                    f'({ticker_sel} OR "{co_name}") AND ("contract award" OR "contract win" OR "IDIQ" OR "option exercised" OR "task order")'
                ),
                "US defense budget announcements / NDAA / appropriations": (
                    '("defense budget" OR NDAA OR "National Defense Authorization Act" OR appropriations) AND (DoD OR Pentagon)'
                ),
                "Allied budgets (NATO / EU) announcements": (
                    '(NATO OR "defense spending" OR "military budget") AND (Germany OR UK OR France OR Poland OR "European Union" OR allies)'
                ),
                "Global air traffic (IATA) + airline order headlines": (
                    '(IATA OR "air traffic" OR "passenger demand" OR RPK OR load factor OR "airline orders" OR "aircraft orders")'
                ),
                "TSA checkpoint numbers / US travel demand (headline proxy)": (
                    '("TSA" OR "checkpoint" OR "passenger volumes" OR "passenger throughput") AND (daily OR week OR record)'
                ),
                "Production rate plans / supply chain commentary (FlightGlobal / Aviation Week)": (
                    '("Aviation Week" OR FlightGlobal OR "production rate" OR "rate increase" OR "line rate" OR "supply chain" OR "engine availability")'
                ),
            }

            for title, q in queries.items():
                st.markdown(f"**{title}**")
                df_news = fetch_google_news_rss_query(q, days=aero_days)
                if df_news is None or df_news.empty:
                    st.caption("No recent RSS headlines found.")
                    continue
                for _, n in df_news.head(12).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    ttl = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{ttl}]({link})  \n  _{src}_")
        
        if sector_exact == "Consumer Discretionary":
            st.markdown("#### Macro / Credit / Policy Catalysts")
        
            cd_window = st.selectbox(
                "Consumer Discretionary headlines window", ["1w", "2w", "1m", "2m", "3m"],
                index=0, key="cd_headlines_window"
            )
            days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
            cd_days = days_map.get(cd_window, 7)
        
            # Heuristic “key metric” query based on industry text (best-effort)
            ind_l = (industry or "").lower()
            if any(k in ind_l for k in ["retail", "apparel", "department", "specialty"]):
                key_metric_q = f'({ticker_sel} OR "{co_name}") AND ("same-store sales" OR "comparable sales" OR comps OR traffic OR "transactions")'
                key_metric_title = "Key metric (industry): Same-store sales / traffic"
            elif any(k in ind_l for k in ["auto", "automobile", "vehicle", "dealership"]):
                key_metric_q = f'({ticker_sel} OR "{co_name}") AND (deliveries OR "unit sales" OR incentives OR "days supply" OR "transaction price")'
                key_metric_title = "Key metric (industry): Units / deliveries / incentives"
            elif any(k in ind_l for k in ["travel", "leisure", "hotel", "airline", "cruise"]):
                key_metric_q = f'({ticker_sel} OR "{co_name}") AND (booking OR bookings OR demand OR occupancy OR RevPAR OR load\ factor)'
                key_metric_title = "Key metric (industry): Bookings / occupancy / RevPAR"
            else:
                key_metric_q = f'({ticker_sel} OR "{co_name}") AND (demand OR "consumer spending" OR "discretionary spending" OR "big ticket")'
                key_metric_title = "Key metric (industry): Demand / discretionary spend signals"
        
            queries = {
                "Consumer confidence / sentiment (survey read-through)": (
                    '(University of Michigan OR "consumer sentiment" OR "consumer confidence") '
                    'AND ("big-ticket" OR "big ticket" OR "major purchase" OR "large purchase" OR durable OR "buying conditions")'
                ),
                "Labor market (jobs / unemployment / wage prints + revisions)": (
                    '(nonfarm payrolls OR NFP OR unemployment OR jobless OR "wage growth" OR "average hourly earnings") '
                    'AND (BLS OR "Bureau of Labor Statistics")'
                ),
                "Rates / consumer credit (auto loans, long-end yields)": (
                    '("10-year yield" OR "10 year yield" OR Treasury) AND (auto loan OR "consumer credit" OR "loan rates")'
                ),
                "Senior Loan Officer Opinion Survey (credit tightening/loosening)": (
                    '(SLOOS OR "Senior Loan Officer" OR "lending standards") AND (consumer OR credit\ card OR auto)'
                ),
                "FOMC calendar / Fed policy headlines": (
                    '(FOMC OR "Fed meeting" OR "Federal Reserve") AND (calendar OR decision OR minutes OR dot\ plot OR "rate cut" OR "rate hike")'
                ),
                key_metric_title: key_metric_q,
                "Trade agreements / tariffs (policy risk to discretionary demand & imports)": (
                    '(tariff OR tariffs OR "trade agreement" OR "trade deal" OR "Section 301" OR "import duties") '
                    'AND (retail OR apparel OR consumer OR "consumer goods" OR automobiles)'
                ),
            }
    
            for title, q in queries.items():
                st.markdown(f"**{title}**")
                df_news = fetch_google_news_rss_query(q, days=cd_days)
                if df_news is None or df_news.empty:
                    st.caption("No recent RSS headlines found.")
                    continue
                for _, n in df_news.head(12).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    ttl = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{ttl}]({link})  \n  _{src}_")
        
        if sector_exact == "Construction":
            st.markdown("#### Backlog / ABI / Spending / Contract Awards")

            cons_window = st.selectbox(
                "Construction headlines window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="cons_headlines_window"
            )
            days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
            cons_days = days_map.get(cons_window, 7)

            queries = {
                "ABC Construction Backlog Indicator (CBI) / industry survey": (
                    '("Associated Builders and Contractors" OR ABC) AND '
                    '("Construction Backlog Indicator" OR CBI OR backlog) '
                    'AND (monthly OR survey OR report)'
                ),
                "ABI (Architectural Billings Index) — AIA": (
                    '("Architectural Billings Index" OR ABI) AND '
                    '("American Institute of Architects" OR AIA)'
                ),
                "Construction spending / demand commentary": (
                    '(construction spending OR "nonresidential construction" OR "residential construction" '
                    'OR "infrastructure spending")'
                ),
                "Major contract awards (DOT / DoD / government)": (
                    '("contract award" OR "awarded contract" OR "design-build" OR "IDIQ") AND '
                    '(DOT OR "Department of Transportation" OR DoD OR "Department of Defense" OR '
                    '"Army Corps of Engineers" OR FHWA OR FAA OR "General Services Administration" OR GSA)'
                ),
                "Single big contract win (company-specific)": (
                    f'({ticker_sel} OR "{co_name}") AND '
                    '("contract award" OR "awarded" OR "selected" OR "wins" OR "winning bidder" OR '
                    '"lowest responsible bidder" OR "notice to proceed" OR NTP OR "project award")'
                ),
                "Input costs / ENR cost reports (headline proxy)": (
                    '("Engineering News-Record" OR ENR) AND '
                    '("construction cost" OR "cost index" OR "materials prices" OR "labor costs")'
                ),
            }

            for title, q in queries.items():
                st.markdown(f"**{title}**")
                df_news = fetch_google_news_rss_query(q, days=cons_days)
                if df_news is None or df_news.empty:
                    st.caption("No recent RSS headlines found.")
                    continue

                for _, n in df_news.head(12).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    ttl = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{ttl}]({link})  \n  _{src}_")
        
        if sector_exact == "Consumer Staples":
            st.markdown("#### Valuation: P/E History")
    
            pe_hist = compute_pe_history(ticker_sel)
    
            if pe_hist is None or pe_hist.empty:
                st.caption("No P/E history available (missing trailing EPS).")
            else:
                fig_pe = px.line(pe_hist, x="date", y="value", title="P/E History (Price / Trailing EPS)")
                fig_pe.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_pe, use_container_width=True)
            
            st.markdown("#### Catalysts (Volumes / Input Costs / GLP-1)")

            cs_window = st.selectbox(
                "Consumer Staples headlines window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="cs_headlines_window"
            )
            days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
            cs_days = days_map.get(cs_window, 7)

            queries = {
                "Category volumes / share (Nielsen / IRI / Circana)":
                    '("Nielsen" OR "IRI" OR "Circana" OR "scanner data" OR "category data" OR "category volumes") '
                    'AND (snacks OR beverage OR beverages OR soda OR soft drinks OR cereal OR "packaged food" OR "consumer staples")',

                "Company: volumes / price-mix / elasticity":
                    f'({ticker_sel} OR "{co_name}") AND '
                    '("volume" OR "volumes" OR "unit sales" OR "price/mix" OR "price mix" OR elasticity OR "trade down" OR "downtrading")',

                "Input costs pass-through (wheat/corn/sugar/coffee)":
                    f'({ticker_sel} OR "{co_name}") AND '
                    '(wheat OR corn OR sugar OR coffee OR "input costs" OR "commodity inflation" OR "cost inflation" OR "pricing actions")',

                "Promotions / private label pressure":
                    f'({ticker_sel} OR "{co_name}") AND '
                    '(promotion OR promotions OR discounting OR "private label" OR "store brand" OR "price competition")',

                "GLP-1 / anti-obesity drugs (demand shift risk/opportunity)":
                    '(GLP-1 OR Ozempic OR Wegovy OR Mounjaro OR Zepbound OR "weight loss drug" OR "anti-obesity") '
                    'AND (snacks OR beverage OR beverages OR soda OR "grocery" OR "food demand" OR "consumer staples")',
            }

            for title, q in queries.items():
                st.markdown(f"**{title}**")
                df_news = fetch_google_news_rss_query(q, days=cs_days)
                if df_news is None or df_news.empty:
                    st.caption("No recent RSS headlines found.")
                    continue
                for _, n in df_news.head(10).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    ttl = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{ttl}]({link})  \n  _{src}_")
        
        if sector_exact == "Retail-Wholesale":
            st.markdown("#### Retail / Wholesale Catalysts")

            rw_window = st.selectbox("Retail headlines window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="rw_headlines_window")
            days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
            rwdays = days_map.get(rw_window, 7)

            queries = {
                "Same-Store Sales (Comp Sales) / Customer Traffic": (
                    f'({ticker_sel} OR "{co_name}") AND '
                    '("same-store sales" OR "comparable sales" OR comps OR "customer traffic" OR footfall OR "transactions")'
                ),
                "E-commerce YoY / Online Sales Growth": (
                    f'({ticker_sel} OR "{co_name}") AND '
                    '("e-commerce" OR "online sales" OR "digital sales") AND (YoY OR "year over year" OR growth)'
                ),
                "Inventory / Promotions / Margin Pressure": (
                    f'({ticker_sel} OR "{co_name}") AND '
                    '(inventory OR "inventory levels" OR markdowns OR promotions OR "gross margin")'
                ),
                "Supply chain / Brand power / Pricing power": (
                    f'({ticker_sel} OR "{co_name}") AND '
                    '("supply chain" OR "lead times" OR "in-stock" OR "brand power" OR "pricing power")'
                ),
                "Credit card spending (Visa/Mastercard reports)": (
                    '(Visa OR Mastercard OR "SpendingPulse" OR "spending momentum") AND '
                    '(consumer spending OR "card spending" OR "credit card spending")'
                ),
            }

            for title, q in queries.items():
                st.markdown(f"**{title}**")
                df_news = fetch_google_news_rss_query(q, days=rwdays)
                if df_news is None or df_news.empty:
                    st.caption("No recent RSS headlines found.")
                    continue
                for _, n in df_news.head(10).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    ttl = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{ttl}]({link})  \n  _{src}_")
        
        if sector_exact == "Industrial Products":
            st.markdown("#### Industrial Products: Order / Production / CAPEX Commentary")

            ip_window = st.selectbox("Industrial Products headlines window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="ip_headlines_window")
            days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
            ip_days = days_map.get(ip_window, 7)

            queries = {
                "Production ramp-ups / capacity adds": (
                    f'({ticker_sel} OR "{co_name}") '
                    f'("production ramp" OR ramp-up OR "ramp up" OR "start of production" OR SOP OR '
                    f'"capacity expansion" OR "capacity add" OR "line rate" OR "throughput")'
                ),
                "Order backlog / bookings commentary": (
                    f'({ticker_sel} OR "{co_name}") '
                    f'(backlog OR bookings OR "order book" OR "order intake" OR "book-to-bill" OR billings)'
                ),
                "Outlooks on capex-driven demand": (
                    f'({ticker_sel} OR "{co_name}") '
                    f'(capex OR "capital spending" OR "capital expenditure" OR "industrial demand" OR '
                    f'"equipment demand" OR "factory automation" OR "maintenance capex")'
                ),
            }

            for title, q in queries.items():
                st.markdown(f"**{title}**")
                df_news = fetch_google_news_rss_query(q, days=ip_days)
                if df_news is None or df_news.empty:
                    st.caption("No recent RSS headlines found.")
                    continue
                for _, n in df_news.head(10).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    ttl = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{ttl}]({link})  \n  _{src}_")
        
        if sector_exact in ("Energy", "Oils-Energy"):
            st.markdown("#### Oils–Energy Catalysts")

            energy_window = st.selectbox("Energy headlines window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="energy_headlines_window")
            days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
            edays = days_map.get(energy_window, 7)

            queries = {
                "OPEC+ quota decisions": '(OPEC OR "OPEC+" OR "OPEC plus") AND (quota OR cut OR cuts OR "production target" OR "output policy")',
                "COT net positioning (oil)": '("CFTC" OR "commitments of traders" OR COT) AND (WTI OR Brent OR crude) AND (net long OR net short OR positioning)',
                "Spare capacity / surplus capacity": '(spare capacity OR "surplus capacity") AND (OPEC OR crude OR oil)',
                "Rig count / shale productivity / DUCs": '("rig count" OR "drilling productivity" OR DUC OR "drilled but uncompleted") AND (EIA OR shale OR Permian OR Bakken)',
                "Carbon pricing": '("carbon price" OR "carbon pricing" OR "EU ETS" OR "cap and trade" OR allowance) AND (oil OR energy OR refinery OR airline)',
            }

            for title, q in queries.items():
                st.markdown(f"**{title}**")
                df_news = fetch_google_news_rss_query(q, days=edays)
                if df_news is None or df_news.empty:
                    st.caption("No recent RSS headlines found.")
                    continue
                for _, n in df_news.head(8).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    ttl = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{ttl}]({link})  \n  _{src}_")
        
        if sector_exact == "Transportation":
            st.markdown("#### Freight / Rates / Logistics Headlines")
            q_trans = (
                '("Cass Freight Index" OR "DAT dry van" OR "DAT reefer" OR '
                '"Baltic Dry Index" OR "BDI" OR "Freightos Baltic Index" OR "FBX" OR '
                '"Logistics Managers Index" OR "LMI" OR "supplier deliveries" OR '
                '"spot rates" OR "truckload spot rates" OR "reefer rates")'
            )
            trans_news = fetch_google_news_rss_query(q_trans, days=30)
            if trans_news is None or trans_news.empty:
                st.info("No recent transportation/freight RSS headlines found.")
            else:
                for _, n in trans_news.head(18).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    title = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_")
    
        if sector_exact == "Basic Materials":
            st.markdown("#### Inventories & Supply News")

            bm_window_label = st.selectbox(
                "Supply news window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="bm_supply_news_window"
            )
            days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
            bm_days = days_map.get(bm_window_label, 7)

            q_supply = (
                '(LME warehouse stocks OR "LME stock" OR "EIA inventories" OR "crude inventories" '
                'OR gasoline inventories OR "Baker Hughes rig count" OR rig counts) '
                'AND (metals OR copper OR aluminum OR steel OR oil OR gasoline)'
            )

            supp_news = fetch_google_news_rss_query(q_supply, days=bm_days)

            if supp_news is None or supp_news.empty:
                st.info("No recent supply/inventory RSS headlines found.")
            else:
                for _, n in supp_news.head(16).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    title = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_")

        if sector_exact == "Auto-Tires-Trucks":
            st.markdown("#### Affordability & Incentives Headlines")
            auto_window_label = st.selectbox(
                "Auto headlines window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="auto_headlines_window"
            )
            days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
            auto_days = days_map.get(auto_window_label, 7)

            q_aff = (
                '("Cox Automotive" OR "Moody\'s" OR affordability OR "vehicle affordability" '
                'OR incentives OR "transaction price" OR "monthly payment") '
                'AND (auto OR "new vehicle" OR car OR truck)'
            )
            aff_news = fetch_google_news_rss_query(q_aff, days=auto_days)
            if aff_news is None or aff_news.empty:
                st.info("No recent affordability/incentives RSS headlines found.")
            else:
                for _, n in aff_news.head(14).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    title = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_")

            st.markdown("#### EV Adoption & Mix Headlines")
            q_ev = (
                '("EV adoption" OR "electric vehicle sales" OR "EV share" OR "plug-in" OR BEV OR PHEV '
                'OR "charging network" OR "battery demand") AND (US OR "United States")'
            )
            ev_news = fetch_google_news_rss_query(q_ev, days=auto_days)
            if ev_news is None or ev_news.empty:
                st.info("No recent EV adoption RSS headlines found.")
            else:
                for _, n in ev_news.head(14).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    title = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_")

        if sector_exact == "Medical":
            st.markdown("#### Earnings Surprise History")
            es = fetch_earnings_surprise_history(ticker_sel, limit=28)
            if es is None or es.empty:
                st.info("No earnings surprise history available from Yahoo/yfinance for this ticker.")
            else:
                es2 = es.rename(columns={"earnings_date": "date", "surprise_pct": "value"}).copy()
                es2["date"] = pd.to_datetime(es2["date"], errors="coerce")
                es2["value"] = pd.to_numeric(es2["value"], errors="coerce")
                es2 = es2.dropna(subset=["date", "value"]).sort_values("date")
                fig_es = px.line(es2, x="date", y="value", title="Earnings Surprise (%)")
                fig_es.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_es, use_container_width=True, key="earn_surprise_med")

            st.markdown("#### Policy + FDA Approvals News")
            news_window_label = st.selectbox(
                "Medical catalysts news window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="med_news_window"
            )
            days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
            news_days = days_map.get(news_window_label, 7)

            q_policy_fda = (
                f'({ticker_sel} OR "{co_name}") '
                f'(FDA approval OR PDUFA OR "complete response letter" OR CRL OR "FDA decision" '
                f'OR "advisory committee" OR CMS OR Medicare OR Medicaid OR "healthcare policy")'
            )
            n1 = fetch_google_news_rss_query(q_policy_fda, days=news_days)
            if n1 is None or n1.empty:
                st.write("No recent policy/FDA-related RSS news found.")
            else:
                for _, n in n1.head(20).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    title = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_")

            st.markdown("#### M&A News")
            q_ma = (
                f'({ticker_sel} OR "{co_name}") '
                f'(acquisition OR acquire OR merger OR "merger agreement" OR "strategic review" '
                f'OR takeover OR "go-private")'
            )
            n2 = fetch_google_news_rss_query(q_ma, days=news_days)
            if n2 is None or n2.empty:
                st.write("No recent M&A RSS news found.")
            else:
                for _, n in n2.head(15).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    title = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_")

            st.markdown("#### FDA Decision Calendar")
            fda_cal = fetch_fda_calendar(ticker_sel, company_name=co_name)
            if fda_cal is None or fda_cal.empty:
                st.info("No FDA calendar rows found (source may be blocked or changed).")
            else:
                st.dataframe(fda_cal, use_container_width=True, height=240)

        if sector_exact == "Business Services":
            st.markdown("#### Margins: EBITDA vs Operating")
            ts = fetch_business_services_fundamentals(ticker_sel)
            opm_df = ts.get("operating_margin_pct", pd.DataFrame())
            ebdm_df = ts.get("ebitda_margin_pct", pd.DataFrame())

            mdfs = []
            if opm_df is not None and not opm_df.empty:
                a = opm_df.rename(columns={"value": "Operating Margin"}).set_index("date")
                mdfs.append(a)
            if ebdm_df is not None and not ebdm_df.empty:
                b = ebdm_df.rename(columns={"value": "EBITDA Margin"}).set_index("date")
                mdfs.append(b)

            if not mdfs:
                st.info("Margin history unavailable (EBITDA and/or operating income series missing).")
            else:
                m = pd.concat(mdfs, axis=1).reset_index().sort_values("date")
                fig_m = px.line(m, x="date", y=[c for c in m.columns if c != "date"], title="Margins (%)")
                fig_m.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_m, use_container_width=True, key="bs_margins")

            st.markdown("#### M&A News")
            news_window_label_bs = st.selectbox("M&A news window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="bs_ma_news_window")
            days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
            bs_days = days_map.get(news_window_label_bs, 7)

            q_bs_ma = (
                f'({ticker_sel} OR "{co_name}") '
                f'(acquisition OR acquire OR merger OR "merger agreement" OR "strategic review" '
                f'OR takeover OR "deal" OR "divestiture" OR "asset sale")'
            )
            bs_ma = fetch_google_news_rss_query(q_bs_ma, days=bs_days)
            if bs_ma is None or bs_ma.empty:
                st.info("No recent M&A RSS headlines found.")
            else:
                for _, n in bs_ma.head(20).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    title = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_")

            st.markdown("#### Book-to-Bill / Backlog / Bookings / Retention")
            ops = fetch_business_services_operational_kpis_best_effort(ticker_sel) or {}

            def _fmt_num(x, kind="num"):
                try:
                    x = float(x)
                except Exception:
                    return "N/A"
                if not np.isfinite(x):
                    return "N/A"
                if kind == "pct":
                    return f"{x:.2f}%"
                if kind == "x":
                    return f"{x:.2f}x"
                if kind == "usd":
                    return f"${x:,.0f}"
                return f"{x:.2f}"

            ops_tbl = pd.DataFrame([{
                "Book-to-Bill": _fmt_num(ops.get("book_to_bill", np.nan), "x"),
                "Backlog (latest)": _fmt_num(ops.get("backlog_latest", np.nan), "usd"),
                "Backlog YoY": _fmt_num(ops.get("backlog_yoy_pct", np.nan), "pct"),
                "Bookings (latest)": _fmt_num(ops.get("bookings_latest", np.nan), "usd"),
                "Bookings YoY": _fmt_num(ops.get("bookings_yoy_pct", np.nan), "pct"),
                "Client retention": _fmt_num(ops.get("retention_rate_pct", np.nan), "pct"),
                "Renewal rate": _fmt_num(ops.get("renewal_rate_pct", np.nan), "pct"),
            }])

            st.dataframe(ops_tbl, use_container_width=True, height=80)
            st.caption("Note: book-to-bill/backlog/bookings/retention are often non-standard and may not appear in SEC XBRL; N/A is normal.")

        if sector_exact == "Computer and Technology":
            st.markdown("#### Earnings Surprise History")
            es = fetch_earnings_surprise_history(ticker_sel, limit=28)
            if es is None or es.empty:
                st.info("No earnings surprise history available from Yahoo/yfinance for this ticker.")
            else:
                es2 = es.rename(columns={"earnings_date": "date", "surprise_pct": "value"}).copy()
                es2["date"] = pd.to_datetime(es2["date"], errors="coerce")
                es2["value"] = pd.to_numeric(es2["value"], errors="coerce")
                es2 = es2.dropna(subset=["date", "value"]).sort_values("date")
                fig_es = px.line(es2, x="date", y="value", title="Earnings Surprise (%)")
                fig_es.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_es, use_container_width=True, key="earn_surprise_tech")

            st.markdown("#### Semiconductor Sales & Book-to-Bill Headlines")
            # Note: SEMI discontinued the classic monthly NA book-to-bill in 2017; we track via headlines now.
            tech_window_label = st.selectbox(
                "Semiconductor headlines window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="tech_semi_news_window"
            )
            days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
            tech_days = days_map.get(tech_window_label, 7)

            q_semi_sales_btb = (
                '("semiconductor sales" OR WSTS OR "SIA semiconductor sales" OR "chip sales") '
                'OR ("semiconductor" AND ("book-to-bill" OR "book to bill" OR bookings OR billings))'
            )
            semi_news = fetch_google_news_rss_query(q_semi_sales_btb, days=tech_days)
            if semi_news is None or semi_news.empty:
                st.info("No recent semiconductor sales / book-to-bill RSS headlines found.")
            else:
                for _, n in semi_news.head(18).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    title = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_")

            st.markdown("#### Tech Research & Product Launch News")
            launch_window_label = st.selectbox(
                "Product launch news window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="tech_launch_news_window"
            )
            launch_days = days_map.get(launch_window_label, 7)

            q_launch = (
                f'({ticker_sel} OR "{co_name}") '
                f'("product launch" OR "launch event" OR "reviews" OR "hands-on" OR '
                f'"first impressions" OR "benchmark" OR "unboxing" OR "preorder" OR '
                f'"sell-through" OR "demand" OR "channel checks" OR "research note" OR '
                f'"analyst note" OR "downgrade" OR "upgrade")'
            )
            launch_news = fetch_google_news_rss_query(q_launch, days=launch_days)
            if launch_news is None or launch_news.empty:
                st.info("No recent product-launch/research RSS headlines found.")
            else:
                for _, n in launch_news.head(20).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    title = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_")

            st.markdown("#### Cloud Segment Growth & CAPEX Commentary")
            cloud_window_label = st.selectbox(
                "Cloud/CAPEX news window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="tech_cloud_news_window"
            )
            cloud_days = days_map.get(cloud_window_label, 7)

            q_cloud_capex = (
                f'({ticker_sel} OR "{co_name}") '
                f'("cloud" OR "AI infrastructure" OR "data center" OR datacenter OR "hyperscale" OR '
                f'"cloud revenue" OR "cloud growth" OR "ARR" OR "subscription") '
                f'(capex OR "capital expenditures" OR "capital spending" OR "data center capex" OR "GPU capex")'
            )
            cloud_news = fetch_google_news_rss_query(q_cloud_capex, days=cloud_days)
            if cloud_news is None or cloud_news.empty:
                st.info("No recent cloud-growth/CAPEX RSS headlines found.")
            else:
                for _, n in cloud_news.head(20).iterrows():
                    t = pd.to_datetime(n.get("time"), utc=True, errors="coerce")
                    t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
                    title = str(n.get("title", ""))
                    link = str(n.get("link", ""))
                    src = str(n.get("source", ""))
                    st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_")

# =========================
# Technical Analysis
# =========================
st.subheader("Technical Analysis")

with st.spinner("Fetching price history…"):
    px_panel = fetch_prices_panel([normalize_yf_ticker(ticker_sel)], period=DEFAULT_PRICE_PERIOD, interval=DEFAULT_PRICE_INTERVAL)

df_t = get_ohlc_from_panel(px_panel, normalize_yf_ticker(ticker_sel))

if df_t.empty or "Close" not in df_t.columns:
    st.warning("No price data for selected ticker.")
    st.stop()

df_t = df_t.dropna(subset=["Close"])
close = df_t["Close"].astype(float)
high = df_t["High"].astype(float) if "High" in df_t.columns else close
low = df_t["Low"].astype(float) if "Low" in df_t.columns else close
volume = df_t["Volume"].astype(float) if "Volume" in df_t.columns else pd.Series(index=close.index, data=np.nan)

ema_spans = [5, 10, 20, 60, 120, 250]
emas = {f"EMA{n}": ema(close, n) for n in ema_spans}
rsi14 = rsi_series(close, 14)
macd_line, signal_line, hist = macd(close, 12, 26, 9)
k_line, d_line, j_line = kdj(high, low, close, n=9, k=3, d=3)

bb_mid, bb_up, bb_dn, bb_width = bollinger(close, 20, 2.0)
atr14 = atr(high, low, close, 14)
plus_di, minus_di, adx14 = adx(high, low, close, 14)
obv_s = obv(close, volume) if volume.notna().any() else pd.Series(index=close.index, data=np.nan)
roc10 = roc(close, 10)
cci20 = cci(high, low, close, 20)
vwap_s = vwap_daily(high, low, close, volume) if volume.notna().any() else pd.Series(index=close.index, data=np.nan)

price_df = pd.DataFrame({"Date": close.index, "Close": close.values})
for n in ema_spans:
    price_df[f"EMA{n}"] = emas[f"EMA{n}"].values
price_df["BB_Mid"] = bb_mid.values
price_df["BB_Up"] = bb_up.values
price_df["BB_Dn"] = bb_dn.values
if vwap_s is not None:
    price_df["VWAP"] = vwap_s.values
price_df = price_df.dropna(subset=["Close"]).copy()

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df["Close"], name="Close"))
for n in ema_spans:
    fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df[f"EMA{n}"], name=f"EMA{n}"))
fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df["BB_Up"], name="BB Up", line=dict(width=1)))
fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df["BB_Mid"], name="BB Mid", line=dict(width=1)))
fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df["BB_Dn"], name="BB Dn", line=dict(width=1)))
if "VWAP" in price_df.columns:
    fig_price.add_trace(go.Scatter(x=price_df["Date"], y=price_df["VWAP"], name="VWAP", line=dict(width=1, dash="dot")))
fig_price.update_layout(title=f"{ticker_sel} — Price / EMAs / Bollinger / VWAP", height=460, legend=dict(orientation="h"))
st.plotly_chart(fig_price, use_container_width=True)

rsi_df = pd.DataFrame({"Date": rsi14.index, "RSI14": rsi14.values}).dropna()
fig_rsi = px.line(rsi_df, x="Date", y="RSI14", title="RSI(14)")
fig_rsi.update_layout(height=260)
st.plotly_chart(fig_rsi, use_container_width=True)

macd_df = pd.DataFrame({"Date": close.index, "MACD": macd_line.values, "Signal": signal_line.values, "Hist": hist.values}).dropna()
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=macd_df["Date"], y=macd_df["MACD"], name="MACD"))
fig_macd.add_trace(go.Scatter(x=macd_df["Date"], y=macd_df["Signal"], name="Signal"))
fig_macd.add_trace(go.Bar(x=macd_df["Date"], y=macd_df["Hist"], name="Hist"))
fig_macd.update_layout(
    title="MACD(12,26,9)",
    height=320,
    margin=dict(l=10, r=10, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font=dict(size=10), bgcolor="rgba(255,255,255,0.0)"),
)
st.plotly_chart(fig_macd, use_container_width=True)

kdj_df = pd.DataFrame({"Date": close.index, "K": k_line.values, "D": d_line.values, "J": j_line.values}).dropna()
fig_kdj = px.line(kdj_df, x="Date", y=["K", "D", "J"], title="KDJ(9,3,3)")
fig_kdj.update_layout(height=320)
st.plotly_chart(fig_kdj, use_container_width=True)

atr_df = pd.DataFrame({"Date": atr14.index, "ATR14": atr14.values}).dropna()
fig_atr = px.line(atr_df, x="Date", y="ATR14", title="ATR(14)")
fig_atr.update_layout(height=260)
st.plotly_chart(fig_atr, use_container_width=True)

adx_df = pd.DataFrame({"Date": adx14.index, "ADX14": adx14.values, "+DI": plus_di.values, "-DI": minus_di.values}).dropna()
fig_adx = px.line(adx_df, x="Date", y=["ADX14", "+DI", "-DI"], title="ADX(14) with +DI/-DI")
fig_adx.update_layout(height=320)
st.plotly_chart(fig_adx, use_container_width=True)

if obv_s is not None and obv_s.dropna().shape[0] > 10:
    obv_df = pd.DataFrame({"Date": obv_s.index, "OBV": obv_s.values}).dropna()
    fig_obv = px.line(obv_df, x="Date", y="OBV", title="On-Balance Volume (OBV)")
    fig_obv.update_layout(height=260)
    st.plotly_chart(fig_obv, use_container_width=True)

roc_df = pd.DataFrame({"Date": roc10.index, "ROC10": roc10.values}).dropna()
fig_roc = px.line(roc_df, x="Date", y="ROC10", title="Rate of Change ROC(10) [%]")
fig_roc.update_layout(height=260)
st.plotly_chart(fig_roc, use_container_width=True)

cci_df = pd.DataFrame({"Date": cci20.index, "CCI20": cci20.values}).dropna()
fig_cci = px.line(cci_df, x="Date", y="CCI20", title="CCI(20)")
fig_cci.update_layout(height=260)
st.plotly_chart(fig_cci, use_container_width=True)

ema_cluster = float(np.nanmean([score_ema_trend(close, emas[f"EMA{n}"]) for n in ema_spans]))
sig_df = pd.DataFrame([
    {"Indicator": "EMA Cluster (5/10/20/60/120/250)", "Score": float(np.clip(ema_cluster, -1, 1))},
    {"Indicator": "RSI(14)", "Score": score_rsi_value(rsi14)},
    {"Indicator": "MACD(12,26,9)", "Score": score_macd_value(macd_line, signal_line, hist)},
    {"Indicator": "KDJ (K vs D)", "Score": score_kdj_value(k_line, d_line)},
    {"Indicator": "Bollinger Width", "Score": score_boll_width(bb_width)},
    {"Indicator": "ADX Trend (+DI/-DI)", "Score": score_adx_trend(adx14, plus_di, minus_di)},
    {"Indicator": "OBV Trend", "Score": score_obv(obv_s) if obv_s is not None else 0.0},
    {"Indicator": "ROC(10)", "Score": score_roc(roc10)},
    {"Indicator": "CCI(20)", "Score": score_cci(cci20)},
])
composite = float(sig_df["Score"].mean()) if not sig_df.empty else 0.0
st.markdown(f"#### Technical Analysis Composite Score: **{composite:+.2f}**")

st.dataframe(sig_df, use_container_width=True, height=260, column_config={"Score": st.column_config.NumberColumn(format="%.2f")})
fig_sig = px.bar(sig_df, x="Indicator", y="Score", title="Technical Indicator Scores")
fig_sig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig_sig, use_container_width=True)

# =========================
# News Sentiment (default 1w)
# =========================
st.subheader("News Sentiment")

news_window_label = st.selectbox("News window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="news_window_main")
days_map = {"1w": 7, "2w": 14, "1m": 30, "2m": 60, "3m": 90}
news_days = days_map.get(news_window_label, 7)

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
        st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_ | polarity: `{pol:+.2f}`")

# =========================
# Analyst Ratings
# =========================
st.subheader("Analyst Ratings")

pt = fetch_analyst_price_targets_yf(ticker_sel)
ud = fetch_upgrades_downgrades_yf(ticker_sel, max_rows=50)

cpt1, _ = st.columns([1.2, 1])
with cpt1:
    st.markdown("#### Price Target Summary")
    pt_df = pd.DataFrame([{
        "Mean": pt.get("mean"),
        "Median": pt.get("median"),
        "High": pt.get("high"),
        "Low": pt.get("low"),
    }])
    st.dataframe(
        pt_df,
        use_container_width=True,
        height=90,
        column_config={
            "Mean": st.column_config.NumberColumn(format="%.2f"),
            "Median": st.column_config.NumberColumn(format="%.2f"),
            "High": st.column_config.NumberColumn(format="%.2f"),
            "Low": st.column_config.NumberColumn(format="%.2f"),
        },
    )

st.markdown("#### Upgrades / Downgrades")
if ud is None or ud.empty:
    st.info("No upgrades/downgrades data available for this ticker from yfinance.")
else:
    cols_pref = [c for c in ["Date", "Firm", "ToGrade", "FromGrade", "Action"] if c in ud.columns]
    cols_rest = [c for c in ud.columns if c not in cols_pref]
    st.dataframe(ud[cols_pref + cols_rest], use_container_width=True, height=260)

# =========================
# SEC Filings & Earnings
# =========================
st.subheader("SEC Filings & Earnings")

if "CIK_OVERRIDES" not in st.session_state:
    st.session_state["CIK_OVERRIDES"] = {}

with st.expander("Fix EDGAR for tickers with missing CIK (optional)", expanded=False):
    st.markdown(
        """
If EDGAR is unavailable because **ticker→CIK mapping is missing** (or SEC blocks submissions fetch),
enter the issuer's **CIK** (digits only).
        """
    )
    st.link_button("🔎 Open SEC CIK Lookup", "https://www.sec.gov/search-filings/cik-lookup")

    cur = st.session_state["CIK_OVERRIDES"].get(ticker_sel, "")
    cik_in = st.text_input(f"CIK override for {ticker_sel}", value=str(cur) if cur else "")
    cbtn1, cbtn2 = st.columns(2)
    with cbtn1:
        if st.button("Save CIK override"):
            try:
                v = int(str(cik_in).strip())
                st.session_state["CIK_OVERRIDES"][ticker_sel] = v
                st.success(f"Saved CIK override for {ticker_sel}: {v}")
            except Exception:
                st.error("Please enter digits only (e.g., 320193).")
    with cbtn2:
        if st.button("Clear CIK override"):
            st.session_state["CIK_OVERRIDES"].pop(ticker_sel, None)
            st.success("Cleared override.")

c_sec1, c_sec2 = st.columns([1.3, 1])
with c_sec1:
    forms_sel = st.multiselect("Filings to show", ["10-K", "10-Q", "8-K"], default=[])
with c_sec2:
    filings_limit = st.slider("Max filings", min_value=3, max_value=20, value=8, step=1)

filings_df = get_latest_filings_for_ticker(ticker_sel, forms=forms_sel, limit=filings_limit)

if filings_df.empty:
    st.info("SEC filings unavailable (ticker->CIK mapping missing OR submissions fetch blocked). Try the CIK override above.")
else:
    show_df = filings_df.copy()
    show_df["filingDate"] = pd.to_datetime(show_df["filingDate"], errors="coerce").dt.date
    cols = ["ticker", "cik", "form", "filingDate", "reportDate", "accessionNumber", "index_url", "primary_url"]
    cols = [c for c in cols if c in show_df.columns]
    st.dataframe(
        show_df[cols],
        use_container_width=True,
        height=300,
        column_config={
            "index_url": st.column_config.LinkColumn("EDGAR Index (best)"),
            "primary_url": st.column_config.LinkColumn("Primary Doc"),
        },
    )
    st.caption("Use **EDGAR Index** as the default.")

st.markdown("#### Earnings (Forecasts / Actuals / Surprises)")

earn_df = fetch_earnings_dates_yf(ticker_sel, limit=12)
if earn_df is None or earn_df.empty:
    st.info("Earnings forecast/actual/surprise table is unavailable for this ticker from yfinance (Yahoo fields can be missing).")
else:
    earn_df = earn_df.copy()
    earn_df["earnings_date"] = pd.to_datetime(earn_df["earnings_date"], errors="coerce", utc=True)
    show_cols = ["earnings_date"]
    for c in ["EPS Estimate", "Reported EPS", "Surprise(%)", "Revenue Estimate", "Reported Revenue"]:
        if c in earn_df.columns:
            show_cols.append(c)
    st.dataframe(earn_df[show_cols].head(12), use_container_width=True, height=260)

    nxt = nearest_earnings_catalyst(earn_df)
    if nxt is not None:
        nxt_ts = pd.Timestamp(nxt)
        if nxt_ts.tzinfo is None:
            nxt_utc = nxt_ts.tz_localize("UTC")
        else:
            nxt_utc = nxt_ts.tz_convert("UTC")

        try:
            import zoneinfo
            tor = zoneinfo.ZoneInfo("America/Toronto")
            nxt_tor = nxt_utc.to_pydatetime().astimezone(tor)
            st.success(
                "Nearest earnings catalyst: "
                f"**{nxt_utc.strftime('%Y-%m-%d %H:%M')} UTC** / "
                f"**{nxt_tor.strftime('%Y-%m-%d %H:%M')} America/Toronto**"
            )
        except Exception:
            st.success(f"Nearest earnings catalyst (UTC): {nxt_utc.strftime('%Y-%m-%d %H:%M')}")

# =========================
# Investor Relations
# =========================
st.subheader("Investor Relations")

website = (info or {}).get("website", "")
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
            st.markdown("**IR / Press / News links:**")
            for txt, href in links:
                st.markdown(f"- [{txt}]({href})")
