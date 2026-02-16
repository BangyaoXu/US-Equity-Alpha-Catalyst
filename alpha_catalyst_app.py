# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import glob
import time
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
# FRED: US Healthcare Policy Uncertainty Index (EPUHEALTHCARE)
# =========================
@st.cache_data(ttl=24 * 60 * 60)
def fetch_fred_epu_healthcare() -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=EPUHEALTHCARE"
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
# FDA decision calendar scraping
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

@st.cache_data(ttl=CACHE_TTL_SECTOR)
def fetch_indicator_bundle(ticker: str) -> Dict[str, object]:
    sym = normalize_yf_ticker(ticker)
    info = fetch_info_one(sym) or {}
    inc = yf_quarterly_income(sym)
    bal = yf_quarterly_balance(sym)
    cfs = yf_quarterly_cashflow(sym)

    scalars: List[Dict[str, object]] = []

    def add_scalar(name, value, unit, update):
        scalars.append({"Indicator": name, "Value": value, "Unit": unit, "Update": update})

    rev_s = _row_get(inc, ["total revenue", "totalrevenue", "revenue"])
    gp_s = _row_get(inc, ["gross profit", "grossprofit"])
    opinc_s = _row_get(inc, ["operating income", "operatingincome"])
    rd_s = _row_get(inc, ["research development", "researchdevelopment", "research & development"])

    rev_yoy = _q_yoy_growth(rev_s) if rev_s is not None else np.nan
    add_scalar("Revenue YoY", rev_yoy, "%", "Quarterly")

    gm = _q_margin_pct(gp_s, rev_s) if gp_s is not None and rev_s is not None else np.nan
    add_scalar("Gross Margin", gm, "%", "Quarterly")

    om = _q_margin_pct(opinc_s, rev_s) if opinc_s is not None and rev_s is not None else np.nan
    add_scalar("Operating Margin", om, "%", "Quarterly")

    # Medical cost ratio proxy: 1 - operating margin
    cost_ratio = (100.0 - om) if np.isfinite(om) else np.nan
    add_scalar("Total Cost Ratio", cost_ratio, "%", "Quarterly")

    rd_int = _q_margin_pct(rd_s, rev_s) if rd_s is not None and rev_s is not None else np.nan
    add_scalar("R&D Intensity", rd_int, "%", "Quarterly")

    dinv = _days_inventory(bal, inc)
    add_scalar("Days Inventory", dinv, "days", "Quarterly")

    dso = _ar_days(bal, inc)
    add_scalar("Days Sales Outstanding", dso, "days", "Quarterly")

    ocf = _row_get(cfs, ["total cash from operating activities", "operating cash flow", "totalcashfromoperatingactivities"])
    capex = _row_get(cfs, ["capital expenditures", "capitalexpenditures"])
    if ocf is not None and capex is not None:
        df_cf = pd.concat([ocf.dropna().sort_index(), capex.dropna().sort_index()], axis=1, join="inner")
        fcf_latest = float(df_cf.iloc[-1, 0] + df_cf.iloc[-1, 1]) if not df_cf.empty else np.nan
    else:
        fcf_latest = np.nan
    add_scalar("Free Cash Flow", fcf_latest, "USD", "Quarterly")

    earn = fetch_earnings_dates_yf(sym, limit=8)
    eps_surp = np.nan
    if earn is not None and not earn.empty:
        for col in ["Surprise(%)", "Surprise (%)", "Surprise %", "Surprise"]:
            if col in earn.columns:
                eps_surp = _to_num(earn[col].iloc[0])
                break
    add_scalar("EPS Surprise", eps_surp, "%", "Quarterly (earnings)")

    add_scalar("Forward P/E", _to_num(info.get("forwardPE")), "x", "Daily")
    rg = _to_num(info.get("revenueGrowth"))
    add_scalar("Revenue Growth (Yahoo)", rg * 100.0 if np.isfinite(rg) else np.nan, "%", "Daily")
    eg = _to_num(info.get("earningsGrowth"))
    add_scalar("Earnings Growth (Yahoo)", eg * 100.0 if np.isfinite(eg) else np.nan, "%", "Daily")
    roe = _to_num(info.get("returnOnEquity"))
    add_scalar("Return on Equity", roe * 100.0 if np.isfinite(roe) else np.nan, "%", "Daily")
    add_scalar("Debt/Equity", _to_num(info.get("debtToEquity")), "x", "Daily")
    dy = _to_num(info.get("dividendYield"))
    add_scalar("Dividend Yield", dy * 100.0 if np.isfinite(dy) else np.nan, "%", "Daily")

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
    }

    driver_series_map = {
        "Basic Materials": [("HG=F", "Copper Futures"), ("GC=F", "Gold Futures")],
        "Energy": [("CL=F", "WTI Crude"), ("NG=F", "Natural Gas")],
        "Transportation": [("CL=F", "WTI Crude"), ("BZ=F", "Brent Crude")],
        "Aerospace": [("ITA", "Aerospace & Defense ETF")],
        "Construction": [("ITB", "Homebuilders ETF")],
    }

    return {
        "scalars": pd.DataFrame(scalars),
        "sector_etf_map": sector_etf_map,
        "driver_series_map": driver_series_map,
    }

def build_indicator_series(sector_name_exact: str, ticker: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    payload = fetch_indicator_bundle(ticker)
    scalars = payload["scalars"].copy()
    if scalars is not None and not scalars.empty and "Indicator" in scalars.columns:
        scalars["Indicator"] = scalars["Indicator"].map(strip_trailing_parens)

    series: Dict[str, pd.DataFrame] = {}

    def _yf_series(symbol: str, period="5y") -> Optional[pd.DataFrame]:
        try:
            px = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
            if px is None or px.empty or "Close" not in px.columns:
                return None
            out = px[["Close"]].dropna().reset_index()
            out.columns = ["date", "value"]
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out["value"] = pd.to_numeric(out["value"], errors="coerce")
            out = out.dropna(subset=["date", "value"]).sort_values("date")
            return out
        except Exception:
            return None

    etf_map = payload.get("sector_etf_map", {})
    etf = etf_map.get(sector_name_exact)
    if etf:
        ser = _yf_series(etf)
        if ser is not None and not ser.empty:
            series[f"{sector_name_exact} Benchmark ETF ({etf})"] = ser

    driver_map = payload.get("driver_series_map", {})
    for sym, label in driver_map.get(sector_name_exact, []):
        ser = _yf_series(sym)
        if ser is not None and not ser.empty:
            series[label] = ser

    if sector_name_exact == "Medical":
        epu = fetch_fred_epu_healthcare()
        if epu is not None and not epu.empty:
            series["U.S. Healthcare Policy Uncertainty Index"] = epu

    return scalars, series

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

def cik10(cik_int: int) -> str:
    return str(int(cik_int)).zfill(10)

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
# Technical scoring (unchanged)
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
st.subheader("Indicators")

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
                fig = px.line(ser.tail(3000), x="date", y="value")
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            i += 1

with tab_stock:
    # --- quick KPI line (no table)
    def _scalar_value(indicator: str) -> float:
        if scalars_df.empty or "Indicator" not in scalars_df.columns:
            return np.nan
        hit = scalars_df[scalars_df["Indicator"].astype(str).str.lower().eq(indicator.lower())]
        if hit.empty:
            return np.nan
        return pd.to_numeric(hit["Value"].iloc[0], errors="coerce")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Revenue YoY", f"{_scalar_value('Revenue YoY'):.2f}%" if np.isfinite(_scalar_value("Revenue YoY")) else "N/A")
    with kpi2:
        st.metric("Operating Margin", f"{_scalar_value('Operating Margin'):.2f}%" if np.isfinite(_scalar_value("Operating Margin")) else "N/A")
    with kpi3:
        st.metric("Total Cost Ratio", f"{_scalar_value('Total Cost Ratio'):.2f}%" if np.isfinite(_scalar_value("Total Cost Ratio")) else "N/A")
    with kpi4:
        st.metric("EPS Surprise", f"{_scalar_value('EPS Surprise'):.2f}%" if np.isfinite(_scalar_value("EPS Surprise")) else "N/A")

    st.markdown("#### Earnings Surprise History")
    es = fetch_earnings_surprise_history(ticker_sel, limit=28)
    if es is None or es.empty:
        st.info("No earnings surprise history available from Yahoo/yfinance for this ticker.")
    else:
        es2 = es.rename(columns={"earnings_date": "date", "surprise_pct": "value"}).copy()
        es2["date"] = pd.to_datetime(es2["date"], errors="coerce")
        es2["value"] = pd.to_numeric(es2["value"], errors="coerce")
        es2 = es2.dropna(subset=["date", "value"]).sort_values("date")
        fig = px.line(es2, x="date", y="value", title="Earnings Surprise (%)")
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if sector_exact == "Medical":
        st.markdown("#### Policy + FDA Approvals News")
        news_window_label = st.selectbox("Medical catalysts news window", ["1w", "2w", "1m", "2m", "3m"], index=0, key="med_news_window")
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
            n1 = n1.head(20).copy()
            for _, n in n1.iterrows():
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
            n2 = n2.head(15).copy()
            for _, n in n2.iterrows():
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
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font=dict(size=10),
        bgcolor="rgba(255,255,255,0.0)"
    ),
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
st.markdown(f"#### Indicator Signals (Composite Score: **{composite:+.2f}**)")

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
