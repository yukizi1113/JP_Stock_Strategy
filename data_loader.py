"""
データローダー
stooq / yfinance / JPX xls からデータを取得する共通ユーティリティ。
"""
from __future__ import annotations
import io
import re
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import date, timedelta
from typing import List, Optional

from config import JPX_DATA_URL, JPX_EXCLUDE_TYPES, PRICE_SUFFIX

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────── JPX ユニバース ───────────────────────────────

def build_jpx_universe(exclude_types: frozenset = JPX_EXCLUDE_TYPES) -> pd.DataFrame:
    """
    JPXの data_j.xls を取得し、上場普通株式のDataFrameを返す。
    Columns: ticker, company, market_type
    """
    try:
        import xlrd
    except ImportError:
        raise ImportError("xlrd が必要です: pip install xlrd")

    r = requests.get(JPX_DATA_URL, timeout=30)
    r.raise_for_status()
    wb = xlrd.open_workbook(file_contents=r.content)
    ws = wb.sheet_by_index(0)

    rows = []
    for i in range(1, ws.nrows):
        raw_code = ws.cell_value(i, 1)   # B列: コード
        company  = ws.cell_value(i, 2)   # C列: 銘柄名
        mtype    = str(ws.cell_value(i, 3)).strip()  # D列: 市場区分

        if mtype in exclude_types:
            continue

        # ticker 正規化: 数値→文字列, 末尾"0"除去（英字付き ticker 対応）
        raw_str = str(raw_code).strip()
        if raw_str.endswith(".0"):
            raw_str = raw_str[:-2]
        ticker = re.sub(r"0$", "", raw_str) if re.match(r"^\d+$", raw_str) else raw_str

        rows.append({"ticker": ticker, "company": str(company).strip(), "market_type": mtype})

    return pd.DataFrame(rows)


# ─────────────────────────── 株価データ取得 ──────────────────────────────

def fetch_prices_stooq(
    tickers: List[str],
    start: str,
    end: str,
    field: str = "Close",
) -> pd.DataFrame:
    """
    pandas-datareader (stooq) で複数銘柄の終値を取得。
    Returns: DataFrame(index=date, columns=ticker)
    """
    frames = {}
    for tk in tickers:
        symbol = f"{tk}.JP"
        try:
            df = web.DataReader(symbol, "stooq", start=start, end=end)
            if field in df.columns and not df.empty:
                frames[tk] = df[field].sort_index()
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index()


def fetch_prices_yfinance(
    tickers: List[str],
    start: str,
    end: str,
    field: str = "Close",
) -> pd.DataFrame:
    """
    yfinance で複数銘柄の終値を取得（stooq の代替）。
    Returns: DataFrame(index=date, columns=ticker)
    """
    yf_tickers = [f"{tk}{PRICE_SUFFIX}" for tk in tickers]
    raw = yf.download(yf_tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if len(tickers) == 1:
        prices = raw[[field]].rename(columns={field: tickers[0]})
    else:
        prices = raw[field].copy()
        prices.columns = [c.replace(PRICE_SUFFIX, "") for c in prices.columns]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.sort_index()


def fetch_prices(
    tickers: List[str],
    start: str,
    end: Optional[str] = None,
    source: str = "yfinance",
) -> pd.DataFrame:
    """
    統一インターフェース: source="stooq" or "yfinance"
    """
    end = end or date.today().isoformat()
    if source == "stooq":
        return fetch_prices_stooq(tickers, start, end)
    return fetch_prices_yfinance(tickers, start, end)


# ─────────────────────────── リターン計算 ─────────────────────────────────

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    monthly = prices.resample("ME").last()
    return monthly.pct_change().dropna(how="all")


# ─────────────────────────── ベンチマーク ──────────────────────────────────

def fetch_benchmark(
    start: str,
    end: Optional[str] = None,
    ticker: str = "^N225",
) -> pd.Series:
    end = end or date.today().isoformat()
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    s = raw["Close"].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "benchmark"
    return s


# ─────────────────────────── バックテスト指標 ──────────────────────────────

def performance_stats(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    rf: float = 0.001,  # 無リスク金利 年率0.1%
) -> dict:
    """
    共通パフォーマンス指標を計算する。
    returns: 日次リターン系列
    """
    r = returns.dropna()
    ann_ret  = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol  = r.std() * np.sqrt(252)
    sharpe   = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan

    cum      = (1 + r).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd   = drawdown.min()

    stats = {
        "ann_return":  round(ann_ret, 4),
        "ann_vol":     round(ann_vol, 4),
        "sharpe":      round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar":      round(ann_ret / abs(max_dd), 4) if max_dd != 0 else np.nan,
    }

    if benchmark is not None:
        b = benchmark.reindex(r.index).pct_change().dropna()
        r_aligned = r.reindex(b.index).dropna()
        b_aligned = b.reindex(r_aligned.index)
        beta = np.cov(r_aligned, b_aligned)[0, 1] / np.var(b_aligned) if len(b_aligned) > 2 else np.nan
        alpha = ann_ret - rf - beta * ((1 + b_aligned.mean()) ** 252 - 1 - rf)
        stats["beta"]  = round(float(beta), 4)
        stats["alpha"] = round(float(alpha), 4)

    return stats
