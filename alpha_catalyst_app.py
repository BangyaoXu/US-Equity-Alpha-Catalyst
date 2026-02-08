from __future__ import annotations

import re
import math
import glob
import time
from dataclasses import dataclass
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
from dateutil.parser import parse as dt_parse
from textblob import TextBlob

# =========================
# APP CONFIG
# =========================
st.set_page_config(layout="wide", page_title="Equity Alpha & Catalyst Dashboard")

UNIVERSE_GLOB = "selected_universe_*.csv"
DEFAULT_PRICE_PERIOD = "1y"        # enough for MA/vol + 3M momentum
NEWS_WINDOW_DAYS = 30             # for intensity zscore
NEWS_RECENT_DAYS = 7              # for current catalysts/sentiment
CACHE_TTL_PRICES = 60 * 60        # 1h
CACHE_TTL_META = 60 * 60          # 1h
CACHE_TTL_NEWS = 30 * 60          # 30m

# Sector ETF mapping (US-centric; you can extend later)
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Communication Services": "XLC",
    "Real Estate": "XLRE",
}

# =========================
# UTILITIES
# =========================
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


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def clamp(x, lo=-3.0, hi=3.0):
    if pd.isna(x):
        return np.nan
    return max(lo, min(hi, x))


def sigmoid(x):
    if pd.isna(x):
        return np.nan
    return 1 / (1 + math.exp(-x))


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
    # sort by date desc
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def sentiment_polarity(text: str) -> float:
    # TextBlob polarity in [-1, 1]
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


# =========================
# DATA FETCH
# =========================
@st.cache_data(ttl=CACHE_TTL_PRICES)
def fetch_prices_panel(tickers: List[str], period: str = DEFAULT_PRICE_PERIOD) -> pd.DataFrame:
    # MultiIndex columns: (Ticker, Field)
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


@st.cache_data(ttl=CACHE_TTL_META)
def fetch_recs_one(ticker: str) -> pd.DataFrame:
    """
    Analyst revision proxy:
    Try yfinance 'upgrades_downgrades' (preferred). If unavailable, try 'recommendations' history.
    We'll derive a simple net score over last 30 days.
    """
    t = yf.Ticker(ticker)
    # yfinance versions differ; try multiple attributes
    for attr in ["upgrades_downgrades", "recommendations"]:
        try:
            df = getattr(t, attr)
            if df is None:
                continue
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.copy()
                return df
        except Exception:
            pass

    # try getter methods
    for fn in ["get_upgrades_downgrades", "get_recommendations"]:
        try:
            f = getattr(t, fn)
            df = f()
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.copy()
        except Exception:
            pass

    return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL_META)
def fetch_earnings_dates_one(ticker: str) -> pd.DataFrame:
    """
    Earnings proximity score:
    yfinance provides earnings dates table for many tickers via get_earnings_dates().
    We'll grab next upcoming date if present.
    """
    t = yf.Ticker(ticker)
    try:
        df = t.get_earnings_dates(limit=12)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL_NEWS)
def fetch_google_news_rss(ticker: str, days: int = NEWS_WINDOW_DAYS) -> pd.DataFrame:
    """
    Free, no API key:
    Google News RSS query: "{ticker} stock"
    Returns: published datetime (UTC-ish), title, link, source
    """
    q = f"{ticker} stock"
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"

    try:
        feed = feedparser.parse(url)
        rows = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        for e in feed.entries:
            # Parse published
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
            # Some entries embed source in title "Headline - Source"
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
# FEATURE ENGINEERING
# =========================
def sector_etf_for(sector: str) -> Optional[str]:
    if not isinstance(sector, str):
        return None
    return SECTOR_ETF_MAP.get(sector.strip())


def earnings_proximity_score(next_earn_date: Optional[datetime]) -> float:
    """
    Score higher when earnings is closer (but not same-day hype):
    - Peak score around 7–14 days before earnings (common run-up window)
    - Decays outside [0, 45] days
    """
    if next_earn_date is None:
        return np.nan
    now = datetime.now(timezone.utc)
    if next_earn_date.tzinfo is None:
        next_earn_date = next_earn_date.replace(tzinfo=timezone.utc)
    d = (next_earn_date - now).days

    if d < 0:
        return 0.0
    if d > 45:
        return 0.0

    # triangular peak at 10 days
    peak = 10.0
    width = 20.0
    score = max(0.0, 1.0 - abs(d - peak) / width)
    return float(score)


def analyst_revision_score(recs: pd.DataFrame) -> float:
    """
    Convert upgrades/downgrades/recommendation changes to a simple net score last 30 days.
    Robust to tz-naive vs tz-aware datetime issues and yfinance schema differences.

    Output roughly in [-1, +1].
    """
    if recs is None or recs.empty:
        return np.nan

    df = recs.copy()

    # -------------------------
    # Build a UTC-aware datetime column df["dt"]
    # -------------------------
    dt_col = None

    # If index is datetime-like
    if isinstance(df.index, pd.DatetimeIndex):
        df["dt"] = df.index
    else:
        # try common columns
        for c in ["Date", "date", "datetime", "Datetime", "EpochGradeDate", "epochGradeDate"]:
            if c in df.columns:
                dt_col = c
                break
        if dt_col is not None:
            # EpochGradeDate is often seconds since epoch
            if "epoch" in dt_col.lower():
                df["dt"] = pd.to_datetime(df[dt_col], unit="s", errors="coerce", utc=True)
            else:
                df["dt"] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
        else:
            df["dt"] = pd.NaT

    # Normalize timezone: make everything UTC-aware
    # - If tz-naive, localize to UTC
    # - If tz-aware, convert to UTC
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    try:
        if hasattr(df["dt"].dt, "tz") and df["dt"].dt.tz is None:
            df["dt"] = df["dt"].dt.tz_localize("UTC")
        else:
            df["dt"] = df["dt"].dt.tz_convert("UTC")
    except Exception:
        # If conversion fails, coerce to UTC-naive and compare to naive cutoff below
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

    # -------------------------
    # Cutoff (match tz-ness)
    # -------------------------
    cutoff_aware = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)

    # If dt ended up tz-naive, make cutoff naive too
    if isinstance(df["dt"].dtype, pd.DatetimeTZDtype):
        cutoff = cutoff_aware
    else:
        cutoff = cutoff_aware.tz_localize(None)

    df = df[df["dt"] >= cutoff]
    if df.empty:
        return 0.0

    # -------------------------
    # Grade mapping -> numeric
    # -------------------------
    def grade_to_num(g: str) -> Optional[float]:
        if not isinstance(g, str):
            return None
        g = g.lower()

        pos = ["strong buy", "buy", "outperform", "overweight", "accumulate"]
        neu = ["hold", "neutral", "market perform", "equal-weight", "equal weight", "sector weight"]
        neg = ["strong sell", "sell", "underperform", "underweight", "reduce"]

        if any(p in g for p in pos):
            return 1.0
        if any(p in g for p in neu):
            return 0.0
        if any(p in g for p in neg):
            return -1.0
        return None

    cols = {c.lower(): c for c in df.columns}

    score = 0.0
    n = 0

    # upgrades_downgrades style
    to_col = cols.get("tograde")
    fr_col = cols.get("fromgrade")
    if to_col and fr_col:
        for _, r in df.iterrows():
            to_v = grade_to_num(r.get(to_col))
            fr_v = grade_to_num(r.get(fr_col))
            if to_v is None or fr_v is None:
                continue
            score += (to_v - fr_v)
            n += 1

    # fallback: single grade column
    if n == 0:
        grade_col = None
        for c in df.columns:
            if "grade" in c.lower():
                grade_col = c
                break
        if grade_col:
            for _, r in df.iterrows():
                v = grade_to_num(r.get(grade_col))
                if v is None:
                    continue
                score += v
                n += 1

    if n == 0:
        return np.nan

    # normalize [-1, +1]
    return float(np.tanh(score / max(1.0, n)))


def news_intensity_features(news_df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Returns:
      - recent_count_7d
      - intensity_z (7d count vs daily counts over 30d)
      - avg_sent_7d
    """
    if news_df is None or news_df.empty or "time" not in news_df.columns:
        return 0.0, np.nan, 0.0

    df = news_df.dropna(subset=["time"]).copy()
    if df.empty:
        return 0.0, np.nan, 0.0

    # Ensure UTC datetimes
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    if df.empty:
        return 0.0, np.nan, 0.0

    now = pd.Timestamp.utcnow()
    df30 = df[df["time"] >= (now - pd.Timedelta(days=NEWS_WINDOW_DAYS))]
    df7 = df[df["time"] >= (now - pd.Timedelta(days=NEWS_RECENT_DAYS))]

    recent_count_7d = float(len(df7))

    # Daily counts over 30d for zscore baseline
    if df30.empty:
        intensity_z = np.nan
    else:
        daily = df30.set_index("time").resample("D").size()
        # compare 7d count to expected 7d from daily mean/std
        mu = daily.mean()
        sd = daily.std()
        if sd == 0 or np.isnan(sd):
            intensity_z = 0.0
        else:
            intensity_z = float((recent_count_7d / 7.0 - mu) / sd)

    # Average sentiment over last 7d
    if df7.empty:
        avg_sent = 0.0
    else:
        sents = [sentiment_polarity(t) for t in df7["title"].astype(str).tolist()]
        avg_sent = float(np.mean(sents)) if sents else 0.0

    return recent_count_7d, intensity_z, avg_sent


def catalyst_flags_from_titles(titles: List[str]) -> str:
    txt = " ".join([t.lower() for t in titles[:50]])
    flags = []
    # Earnings / guidance
    if any(k in txt for k in ["earnings", "results", "q1", "q2", "q3", "q4", "quarter", "revenue", "eps"]):
        flags.append("Earnings/Results")
    if any(k in txt for k in ["guidance", "outlook", "forecast", "raises", "cuts"]):
        flags.append("Guidance/Outlook")
    # M&A / corporate
    if any(k in txt for k in ["acquire", "acquisition", "merger", "buyout", "takeover", "spin-off", "spinoff"]):
        flags.append("M&A/Corp")
    # Analyst
    if any(k in txt for k in ["upgrade", "downgrade", "initiated", "raises target", "price target", "pt"]):
        flags.append("Analyst")
    # Legal / regulatory
    if any(k in txt for k in ["sec", "doj", "regulator", "lawsuit", "probe", "antitrust", "ban"]):
        flags.append("Reg/Legal")
    # Macro/sector shocks
    if any(k in txt for k in ["tariff", "sanction", "oil", "rates", "inflation", "fed", "opec", "chip", "ai"]):
        flags.append("Macro/Sector")
    return ", ".join(flags) if flags else ""


# =========================
# LOAD UNIVERSE
# =========================
def normalize_universe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect at least ticker/sector/company. We’ll tolerate different column names.
    """
    df = df.copy()

    colmap = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    tcol = pick("ticker", "symbol")
    ccol = pick("company", "name", "company_name")
    scol = pick("sector", "gics_sector")
    icol = pick("industry", "subindustry", "gics_industry")

    if tcol is None:
        raise ValueError("Universe CSV must contain a ticker/symbol column.")

    df["ticker"] = df[tcol].astype(str).str.upper().str.strip()
    df["company"] = df[ccol].astype(str) if ccol else df["ticker"]
    df["sector"] = df[scol].astype(str) if scol else "Unknown"
    df["industry"] = df[icol].astype(str) if icol else ""

    # keep originals too
    return df


# =========================
# BUILD FEATURES TABLE
# =========================
def build_feature_table(universe: pd.DataFrame, prices_panel: pd.DataFrame, sector_px: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, u in universe.iterrows():
        ticker = u["ticker"]
        sector = u["sector"]
        company = u["company"]

        # Price series
        if isinstance(prices_panel.columns, pd.MultiIndex) and ticker in prices_panel.columns.levels[0]:
            close = prices_panel[ticker]["Close"].dropna()
            vol = prices_panel[ticker]["Volume"].dropna() if "Volume" in prices_panel[ticker].columns else pd.Series(dtype=float)
        else:
            continue

        if close.empty:
            continue

        rets = compute_returns(close)
        vol20 = realized_vol(close, 20)
        vol60 = realized_vol(close, 60)
        rsi14 = compute_rsi(close, 14)

        ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
        dist_50 = (close.iloc[-1] / ma50 - 1) if not pd.isna(ma50) else np.nan
        dist_200 = (close.iloc[-1] / ma200 - 1) if not pd.isna(ma200) else np.nan
        vol_surge_20 = volume_surge(vol, 20) if not vol.empty else np.nan

        # Fundamentals
        info = fetch_info_one(ticker)
        mkt_cap = _safe_float(info.get("marketCap"))
        pe = _safe_float(info.get("trailingPE"))
        fwd_pe = _safe_float(info.get("forwardPE"))
        beta = _safe_float(info.get("beta"))

        # News + sentiment + catalysts + intensity
        news = fetch_google_news_rss(ticker, days=NEWS_WINDOW_DAYS)
        titles7 = []
        if news is not None and not news.empty:
            news["time"] = pd.to_datetime(news["time"], utc=True, errors="coerce")
            cutoff7 = pd.Timestamp.utcnow() - pd.Timedelta(days=NEWS_RECENT_DAYS)
            titles7 = news.loc[news["time"] >= cutoff7, "title"].astype(str).tolist()

        news_count_7d, news_int_z, sent_7d = news_intensity_features(news)
        catalysts = catalyst_flags_from_titles(titles7)

        # Earnings proximity
        earn_df = fetch_earnings_dates_one(ticker)
        next_earn = None
        if earn_df is not None and not earn_df.empty:
            # index often is DatetimeIndex
            try:
                idx = pd.to_datetime(earn_df.index, utc=True, errors="coerce")
                idx = idx[idx >= pd.Timestamp.utcnow()]
                if len(idx) > 0:
                    next_earn = idx.min().to_pydatetime()
            except Exception:
                next_earn = None
        earn_score = earnings_proximity_score(next_earn)

        # Analyst revisions proxy
        recs = fetch_recs_one(ticker)
        analyst_score = analyst_revision_score(recs)

        # Relative strength vs sector ETF
        sec_etf = sector_etf_for(sector)
        rs_excess_1m = np.nan
        rs_excess_3m = np.nan
        if sec_etf and isinstance(sector_px.columns, pd.MultiIndex) and sec_etf in sector_px.columns.levels[0]:
            sec_close = sector_px[sec_etf]["Close"].dropna()
            if len(sec_close) > 70 and len(close) > 70:
                # align dates
                aligned = pd.concat([close, sec_close], axis=1, join="inner").dropna()
                if aligned.shape[0] > 70:
                    s_close = aligned.iloc[:, 0]
                    s_sec = aligned.iloc[:, 1]
                    # excess returns
                    if len(aligned) >= 21:
                        rs_excess_1m = (s_close.iloc[-1] / s_close.iloc[-21] - 1) - (s_sec.iloc[-1] / s_sec.iloc[-21] - 1)
                    if len(aligned) >= 63:
                        rs_excess_3m = (s_close.iloc[-1] / s_close.iloc[-63] - 1) - (s_sec.iloc[-1] / s_sec.iloc[-63] - 1)

        # Breadth feature: above 50DMA / 200DMA
        above_50 = 1.0 if (not pd.isna(ma50) and close.iloc[-1] > ma50) else 0.0 if not pd.isna(ma50) else np.nan
        above_200 = 1.0 if (not pd.isna(ma200) and close.iloc[-1] > ma200) else 0.0 if not pd.isna(ma200) else np.nan

        rows.append({
            "Ticker": ticker,
            "Company": company,
            "Sector": sector,
            "Industry": u.get("industry", ""),

            "Price": float(close.iloc[-1]),
            "Ret 1D": rets["ret_1d"],
            "Ret 1W": rets["ret_5d"],
            "Ret 1M": rets["ret_1m"],
            "Ret 3M": rets["ret_3m"],
            "Ret 6M": rets["ret_6m"],

            "Vol 20D": vol20,
            "Vol 60D": vol60,
            "RSI 14": rsi14,
            "VolSurge 20D": vol_surge_20,
            "Dist 50DMA": dist_50,
            "Dist 200DMA": dist_200,
            "Above 50DMA": above_50,
            "Above 200DMA": above_200,

            "MktCap": mkt_cap,
            "PE": pe,
            "Fwd PE": fwd_pe,
            "Beta": beta,

            "NewsCount 7D": news_count_7d,
            "NewsIntensity Z": news_int_z,
            "Sentiment 7D": sent_7d,
            "Catalysts": catalysts,

            "Next Earnings (UTC)": next_earn.strftime("%Y-%m-%d") if next_earn else "",
            "Earnings Prox Score": earn_score,
            "Analyst Rev Score": analyst_score,

            "Sector ETF": sec_etf or "",
            "RS Excess 1M": rs_excess_1m,
            "RS Excess 3M": rs_excess_3m,
        })

    df = pd.DataFrame(rows)
    return df


def build_alpha_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced alpha view:
    - Momentum (1M, 3M, 6M)
    - Relative strength vs sector (excess 1M, 3M)
    - Vol-adjusted momentum: (1M / Vol60)
    - News intensity zscore + sentiment
    - Earnings proximity (run-up)
    - Analyst revisions
    - Volume surge (attention)
    - Penalize overly extended + overbought (very high RSI + far above 50DMA)
    """

    out = df.copy()

    # Core derived features
    out["Mom 1M z"] = zscore(out["Ret 1M"])
    out["Mom 3M z"] = zscore(out["Ret 3M"])
    out["Mom 6M z"] = zscore(out["Ret 6M"])

    out["RSEx 1M z"] = zscore(out["RS Excess 1M"])
    out["RSEx 3M z"] = zscore(out["RS Excess 3M"])

    out["VolAdj 1M"] = pd.to_numeric(out["Ret 1M"], errors="coerce") / pd.to_numeric(out["Vol 60D"], errors="coerce")
    out["VolAdj 1M z"] = zscore(out["VolAdj 1M"])

    out["NewsInt z"] = pd.to_numeric(out["NewsIntensity Z"], errors="coerce").clip(-3, 3)
    out["NewsInt z"] = out["NewsInt z"].fillna(0.0)

    out["Sent z"] = zscore(out["Sentiment 7D"]).fillna(0.0)
    out["Analyst z"] = zscore(out["Analyst Rev Score"]).fillna(0.0)

    # Earnings proximity already [0,1], keep but zscore for mixing
    out["Earn z"] = zscore(out["Earnings Prox Score"]).fillna(0.0)

    out["VolSurge z"] = zscore(out["VolSurge 20D"]).fillna(0.0)

    # Mean reversion risk penalty: too hot
    # RSI above ~75 and far above 50DMA tends to mean short-term pullback risk
    rsi = pd.to_numeric(out["RSI 14"], errors="coerce")
    dist50 = pd.to_numeric(out["Dist 50DMA"], errors="coerce")
    penalty = np.maximum(0, (rsi - 75) / 10) + np.maximum(0, (dist50 - 0.08) / 0.05)
    out["Overheat Penalty"] = pd.Series(penalty, index=out.index).fillna(0.0).clip(0, 3)

    # Composite weights (tuned for 1–3 month horizon)
    # You can adjust easily in sidebar later
    w = {
        "Mom 1M z": 0.30,
        "Mom 3M z": 0.20,
        "Mom 6M z": 0.10,
        "RSEx 1M z": 0.12,
        "RSEx 3M z": 0.08,
        "VolAdj 1M z": 0.08,
        "NewsInt z": 0.05,
        "Sent z": 0.03,
        "Earn z": 0.03,
        "Analyst z": 0.04,
        "VolSurge z": 0.02,
        "Overheat Penalty": -0.10,
    }

    score = np.zeros(len(out))
    for k, wk in w.items():
        score += wk * pd.to_numeric(out[k], errors="coerce").fillna(0.0).values

    out["AlphaScore"] = score
    out["AlphaRank"] = out["AlphaScore"].rank(ascending=False, method="min").astype(int)

    # A more interpretable 0-100
    out["AlphaScore 0-100"] = 100 * (pd.Series(score).rank(pct=True).values)
    return out


# =========================
# UI
# =========================
st.title("📈 Equity Alpha & Catalyst Dashboard")

files = pick_universe_files()
if not files:
    st.error(f"No files found matching {UNIVERSE_GLOB} in this folder.")
    st.stop()

date_options = [d for d, _ in files]
file_map = {d: f for d, f in files}

colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    date_sel = st.selectbox("Universe date", date_options, index=0)
with colB:
    max_stocks = st.number_input("Max tickers to load", min_value=10, max_value=2000, value=300, step=10)
with colC:
    refresh = st.button("Refresh caches", help="Force refresh cached data (prices/news/meta).")

if refresh:
    st.cache_data.clear()
    st.success("Caches cleared. Re-run will refetch.")

universe_path = file_map[date_sel]
st.caption(f"Using: `{Path(universe_path).name}`")

uni_raw = pd.read_csv(universe_path)
universe = normalize_universe(uni_raw)

# Optional filters
st.sidebar.header("Filters")
sector_list = sorted(universe["sector"].dropna().unique().tolist())
sector_sel = st.sidebar.multiselect("Sector", sector_list, default=sector_list)
universe = universe[universe["sector"].isin(sector_sel)]

# Limit tickers for performance
tickers = universe["ticker"].dropna().astype(str).str.upper().unique().tolist()
tickers = tickers[: int(max_stocks)]

# Prepare sector ETF list for RS calc
sector_etfs = sorted({sector_etf_for(s) for s in universe["sector"].unique().tolist() if sector_etf_for(s)})
sector_etfs = [e for e in sector_etfs if e]

with st.spinner("Fetching prices (tickers + sector ETFs)…"):
    px_panel = fetch_prices_panel(tickers, period=DEFAULT_PRICE_PERIOD)
    sec_panel = fetch_prices_panel(sector_etfs, period=DEFAULT_PRICE_PERIOD) if sector_etfs else pd.DataFrame()

with st.spinner("Building features (fundamentals, earnings, analyst proxy, news, sentiment)…"):
    feats = build_feature_table(universe[universe["ticker"].isin(tickers)], px_panel, sec_panel)

if feats.empty:
    st.warning("No usable tickers found (check CSV tickers).")
    st.stop()

alpha = build_alpha_score(feats)

# =========================
# TOP SUMMARY
# =========================
st.subheader("Alpha Leaderboard")

# quick sector KPI panel
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Universe Size", f"{len(alpha)}")
kpi2.metric("Avg 1M Return", f"{(alpha['Ret 1M'].mean(skipna=True) * 100):.2f}%")
kpi3.metric("Avg News Intensity Z", f"{alpha['NewsIntensity Z'].mean(skipna=True):.2f}")
kpi4.metric("Avg AlphaScore", f"{alpha['AlphaScore'].mean(skipna=True):.2f}")

# main ranking table
display_cols = [
    "AlphaRank", "Ticker", "Company", "Sector",
    "AlphaScore 0-100",
    "Ret 1M", "Ret 3M", "RS Excess 1M", "Vol 60D",
    "NewsCount 7D", "NewsIntensity Z", "Sentiment 7D",
    "Next Earnings (UTC)", "Analyst Rev Score",
    "RSI 14", "Dist 50DMA",
    "Catalysts",
]
table = alpha.sort_values(["AlphaRank"]).reset_index(drop=True)[display_cols]

# formatting
fmt_pct = lambda x: f"{x*100:.2f}%" if pd.notna(x) else ""
fmt_float = lambda x: f"{x:.2f}" if pd.notna(x) else ""

st.dataframe(
    table.style.format({
        "AlphaScore 0-100": "{:.1f}",
        "Ret 1M": fmt_pct, "Ret 3M": fmt_pct, "RS Excess 1M": fmt_pct,
        "Vol 60D": "{:.2f}",
        "NewsIntensity Z": "{:.2f}",
        "Sentiment 7D": "{:.2f}",
        "Analyst Rev Score": "{:.2f}",
        "RSI 14": "{:.1f}",
        "Dist 50DMA": fmt_pct,
    }),
    use_container_width=True,
    height=520
)

# =========================
# SECTOR ANALYTICS
# =========================
st.subheader("Sector Analytics")

sector_agg = (
    alpha.groupby("Sector")
    .agg(
        n=("Ticker", "count"),
        avg_alpha=("AlphaScore", "mean"),
        avg_1m=("Ret 1M", "mean"),
        avg_news=("NewsIntensity Z", "mean"),
        breadth_50=("Above 50DMA", "mean"),
    )
    .reset_index()
    .sort_values("avg_alpha", ascending=False)
)

c1, c2 = st.columns([1, 1])
with c1:
    fig = px.bar(sector_agg, x="Sector", y="avg_alpha", title="Average AlphaScore by Sector")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig2 = px.bar(sector_agg, x="Sector", y="avg_1m", title="Average 1M Return by Sector")
    st.plotly_chart(fig2, use_container_width=True)

st.dataframe(
    sector_agg.style.format({
        "avg_alpha": "{:.2f}",
        "avg_1m": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "",
        "avg_news": "{:.2f}",
        "breadth_50": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "",
    }),
    use_container_width=True
)

# =========================
# DEEP DIVE
# =========================
st.subheader("Stock Deep Dive")

left, right = st.columns([1.2, 1])
with left:
    ticker_sel = st.selectbox("Select ticker", alpha.sort_values("AlphaRank")["Ticker"].tolist())
with right:
    st.write("")

row = alpha[alpha["Ticker"] == ticker_sel].iloc[0].to_dict()

# Price chart
close = None
if isinstance(px_panel.columns, pd.MultiIndex) and ticker_sel in px_panel.columns.levels[0]:
    close = px_panel[ticker_sel]["Close"].dropna()

if close is not None and not close.empty:
    cdf = close.reset_index()
    cdf.columns = ["Date", "Close"]
    figp = px.line(cdf, x="Date", y="Close", title=f"{ticker_sel} — Price (Adj)")
    st.plotly_chart(figp, use_container_width=True)

# Feature cards
m1, m2, m3, m4 = st.columns(4)
m1.metric("AlphaScore (0-100)", f"{row.get('AlphaScore 0-100', np.nan):.1f}")
m2.metric("1M Return", f"{row.get('Ret 1M', np.nan) * 100:.2f}%")
m3.metric("News Intensity Z", f"{row.get('NewsIntensity Z', np.nan):.2f}")
m4.metric("Earnings Prox Score", f"{row.get('Earnings Prox Score', np.nan):.2f}")

d1, d2 = st.columns([1, 1])

with d1:
    st.markdown("**Key Signals**")
    sig = pd.DataFrame([{
        "RS Excess 1M": row.get("RS Excess 1M"),
        "RS Excess 3M": row.get("RS Excess 3M"),
        "Vol 60D": row.get("Vol 60D"),
        "RSI 14": row.get("RSI 14"),
        "Dist 50DMA": row.get("Dist 50DMA"),
        "VolSurge 20D": row.get("VolSurge 20D"),
        "Analyst Rev Score": row.get("Analyst Rev Score"),
        "Sentiment 7D": row.get("Sentiment 7D"),
        "NewsCount 7D": row.get("NewsCount 7D"),
        "Next Earnings (UTC)": row.get("Next Earnings (UTC)"),
        "Catalysts": row.get("Catalysts"),
    }])
    st.dataframe(sig.style.format({
        "RS Excess 1M": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "",
        "RS Excess 3M": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "",
        "Vol 60D": "{:.2f}",
        "RSI 14": "{:.1f}",
        "Dist 50DMA": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "",
        "VolSurge 20D": "{:.2f}",
        "Analyst Rev Score": "{:.2f}",
        "Sentiment 7D": "{:.2f}",
        "NewsCount 7D": "{:.0f}",
    }), use_container_width=True)

with d2:
    st.markdown("**Recent News (Google News RSS)**")
    news_df = fetch_google_news_rss(ticker_sel, days=NEWS_WINDOW_DAYS)
    if news_df is None or news_df.empty:
        st.write("No recent RSS news found.")
    else:
        news_df["time"] = pd.to_datetime(news_df["time"], utc=True, errors="coerce")
        cutoff7 = pd.Timestamp.utcnow() - pd.Timedelta(days=NEWS_RECENT_DAYS)
        recent = news_df[news_df["time"] >= cutoff7].copy()
        if recent.empty:
            recent = news_df.head(12).copy()
        recent = recent.head(12)

        for _, n in recent.iterrows():
            t = n.get("time")
            t_str = t.strftime("%Y-%m-%d") if pd.notna(t) else ""
            title = str(n.get("title", ""))
            link = str(n.get("link", ""))
            src = str(n.get("source", ""))
            sent = sentiment_polarity(title)
            st.markdown(f"- **{t_str}** [{title}]({link})  \n  _{src}_ | sentiment: `{sent:+.2f}`")

# =========================
# EXPORT
# =========================
st.subheader("Export")
csv_out = alpha.sort_values("AlphaRank").to_csv(index=False).encode("utf-8")
st.download_button("Download ranked table (CSV)", data=csv_out, file_name=f"alpha_rank_{date_sel}.csv", mime="text/csv")

st.caption(
    "Notes: Analyst/Earnings availability varies by ticker; free sources can be sparse. "
    "This app is designed to degrade gracefully and can be upgraded to premium APIs later."
)
