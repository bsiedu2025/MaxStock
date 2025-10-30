# -*- coding: utf-8 -*-
# app/pages/13_Foreign_Flow_Detail.py
# Foreign Flow Detail — Streamlit
# - Mobile Mode toggle
# - Sticky filter bar + shadow
# - Hide non-trading days
# - Quick range: All, 1Y, 6M, 3M, 1M, 5D, Custom
# - Candlestick + MA5/20 + panah MACD + alert cross
# - MACD panel + cross markers
# - Scanner cepat (persist di session_state) + Export + Kirim ke Backtest
# - Backtest: SEMUA metrik dihitung di dalam backtest_macd(...), metrik cards pakai popover info (ikon ℹ️)

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import math
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from html import escape

# === DB utils (harus ada di project lu)
from db_utils import get_db_connection, get_db_name

THEME = "#3AA6A0"
st.set_page_config(page_title="Foreign Flow Detail", page_icon="📈", layout="wide")

# === Global CSS (tanpa f-string supaya aman parse)
st.markdown("""
<style>
section.main > div { padding-top: .5rem; }
.sticky-filter { position: sticky; top: 0; z-index: 1000; background: rgba(255,255,255,.96);
  backdrop-filter: blur(6px); border-bottom: 1px solid #e5e7eb; padding:.5rem .25rem .75rem .25rem;
  box-shadow: 0 8px 16px rgba(0,0,0,.06), 0 1px 0 rgba(0,0,0,.04); }
.checks-row { display:flex; align-items:center; gap:1.25rem; }
.checks-row div[data-testid='stCheckbox']{margin-bottom:0!important}

/* Metric cards grid */
.mcards { display:grid; grid-template-columns: repeat(6,minmax(0,1fr)); gap:12px; }
.mcard { border:1px solid #e5e7eb; border-radius:12px; padding:10px 12px; background:#fff;
         box-shadow: 0 1px 3px rgba(0,0,0,.06); position:relative; }
.mcard-top { display:flex; align-items:center; justify-content:space-between; gap:8px; }
.mcard-label { color:#64748b; font-size:12px; font-weight:700; margin-bottom:2px; }

/* CSS-only popover via <details> */
.mcard-info { position:relative; }
.mcard-info > summary { list-style:none; cursor:pointer; display:inline-flex; align-items:center;
  padding:0 6px; height:18px; border-radius:9999px; border:1px solid #c7d2fe; background:#eef2ff;
  color:#4338ca; font-size:12px; user-select:none; }
.mcard-info > summary::-webkit-details-marker { display:none; }
.mcard-info[open] > summary { background:#e0e7ff; }
.mcard-pop { position:absolute; top:24px; right:0; width:240px; z-index:100;
  background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:10px 12px;
  box-shadow:0 8px 24px rgba(2,6,23,.12); color:#334155; font-size:12px; }

.mcard-value { font-size:22px; font-weight:800; line-height:1.1; margin-top:2px; }
.metric-note { color:#64748b; font-size:12px; margin-top:6px; }

@media (max-width: 980px) { .mcards { grid-template-columns: repeat(3,minmax(0,1fr)); } }
@media (max-width: 640px) { .mcards { grid-template-columns: repeat(2,minmax(0,1fr)); } }

/* Radios like chips */
div[role='radiogroup']{display:flex;flex-wrap:wrap;gap:.25rem .5rem}
div[role='radiogroup'] label{border:1px solid #e5e7eb;padding:6px 12px;border-radius:9999px;background:#fff}
</style>
""", unsafe_allow_html=True)

st.title("📈 Pergerakan Harian Saham — Fokus Foreign Flow")
st.caption(f"DB aktif: **{get_db_name()}**")

# === DB helpers
def _alive(conn):
    try:
        conn.reconnect(attempts=2, delay=1)
    except Exception:
        pass

def _close(conn):
    try:
        if hasattr(conn, "is_connected"):
            if conn.is_connected():
                conn.close()
        else:
            conn.close()
    except Exception:
        pass

@st.cache_data(ttl=300)
def list_codes():
    con = get_db_connection(); _alive(con)
    try:
        df = pd.read_sql("SELECT DISTINCT kode_saham FROM data_harian ORDER BY 1", con)
        return df["kode_saham"].tolist()
    finally:
        _close(con)

@st.cache_data(ttl=300)
def date_bounds():
    con = get_db_connection(); _alive(con)
    try:
        s = pd.read_sql("SELECT MIN(trade_date) mn, MAX(trade_date) mx FROM data_harian", con)
        mn, mx = s.loc[0, "mn"], s.loc[0, "mx"]
        if pd.isna(mn) or pd.isna(mx):
            today = date.today()
            return (today - timedelta(days=60), today)
        return (pd.to_datetime(mn).date(), pd.to_datetime(mx).date())
    finally:
        _close(con)

@st.cache_data(ttl=300)
def get_trade_dates(kode: str) -> pd.Series:
    con = get_db_connection(); _alive(con)
    try:
        q = "SELECT trade_date FROM data_harian WHERE kode_saham=%s ORDER BY trade_date"
        df = pd.read_sql(q, con, params=[kode])
        if df.empty:
            return pd.Series(dtype="datetime64[ns]")
        return pd.to_datetime(df["trade_date"])
    finally:
        _close(con)

@st.cache_data(ttl=300, show_spinner=False)
def load_series(kode: str, start: date, end: date) -> pd.DataFrame:
    con = get_db_connection(); _alive(con)
    try:
        cols = pd.read_sql(
            "SELECT LOWER(column_name) col FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'data_harian'", con
        )
        avail = set(cols["col"].tolist())
        base = ["trade_date", "kode_saham"]
        if "nama_perusahaan" in avail:
            base.append("nama_perusahaan")
        price_cols = ["sebelumnya", "open_price", "first_trade", "tertinggi", "terendah", "penutupan", "selisih"]
        opt = ["nilai", "volume", "foreign_buy", "foreign_sell", "net_foreign_value", "bid", "offer", "spread_bps", "close_price"]
        select_cols = [c for c in base + price_cols + opt if c in avail]
        if not {"trade_date", "kode_saham"}.issubset(select_cols):
            raise RuntimeError("Kolom minimal trade_date & kode_saham harus ada")
        sql = "SELECT " + ", ".join(select_cols) + " FROM data_harian WHERE kode_saham=%s AND trade_date BETWEEN %s AND %s ORDER BY trade_date"
        df = pd.read_sql(sql, con, params=[kode, start, end])
        for c in (set(price_cols + opt + ["nama_perusahaan"]) - set(df.columns.str.lower())):
            df[c] = np.nan
        ordered = base + price_cols + [c for c in opt if c in df.columns]
        df = df[[c for c in ordered if c in df.columns]]
        return df
    finally:
        _close(con)

# === Helpers
def _compute_rangebreaks(trade_dates: pd.Series, start: date, end: date, hide_non_trading: bool):
    if not hide_non_trading:
        return []
    rb = [dict(bounds=["sat", "mon"])]
    if trade_dates is not None and len(trade_dates) > 0:
        td = pd.to_datetime(trade_dates).dt.normalize().unique()
        td_set = set(td)
        all_days = pd.date_range(start, end, freq="D")
        weekdays = all_days[all_days.weekday < 5]
        holidays = [d.to_pydatetime() for d in weekdays if d.to_datetime64() not in td_set]
        if len(holidays) > 0:
            rb.append(dict(values=holidays))
    return rb

def _apply_time_axis(fig, trade_dates, start, end, hide):
    fig.update_xaxes(rangebreaks=_compute_rangebreaks(trade_dates, start, end, hide))
    return fig

def ensure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "net_foreign_value" not in df.columns or df["net_foreign_value"].isna().all():
        if {"foreign_buy", "foreign_sell"}.issubset(df.columns):
            df["foreign_buy"] = pd.to_numeric(df.get("foreign_buy"), errors="coerce")
            df["foreign_sell"] = pd.to_numeric(df.get("foreign_sell"), errors="coerce")
            df["net_foreign_value"] = df["foreign_buy"].fillna(0) - df["foreign_sell"].fillna(0)
        else:
            df["net_foreign_value"] = np.nan
    if "spread_bps" not in df.columns or df["spread_bps"].isna().all():
        if {"bid", "offer"}.issubset(df.columns):
            b = pd.to_numeric(df["bid"], errors="coerce")
            o = pd.to_numeric(df["offer"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                sp = (o - b) / ((o + b) / 2) * 10000
            sp[(b <= 0) | (o <= 0) | b.isna() | o.isna()] = np.nan
            df["spread_bps"] = sp
        else:
            df["spread_bps"] = np.nan
    for c in ["nilai", "volume", "foreign_buy", "foreign_sell", "net_foreign_value", "close_price",
              "penutupan", "sebelumnya", "tertinggi", "terendah", "selisih", "open_price", "first_trade"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df

def add_rolling(df: pd.DataFrame, adv_mode: str) -> pd.DataFrame:
    df = df.sort_values("trade_date").copy()
    nilai = pd.to_numeric(df.get("nilai"), errors="coerce")
    map_win = {"1 Tahun": 252, "6 Bulan": 120, "3 Bulan": 60, "1 Bulan": 20}
    if adv_mode == "All (Default)":
        df["adv"] = nilai.expanding().mean()
        df.attrs["adv_label"] = "All"
    else:
        win = map_win.get(adv_mode, 20)
        df["adv"] = nilai.rolling(win, min_periods=1).mean()
        df.attrs["adv_label"] = adv_mode
    df["vol_avg"] = pd.to_numeric(df.get("volume"), errors="coerce").rolling(20, min_periods=1).mean()
    df["ratio"] = df["nilai"] / df["adv"]
    df["cum_nf"] = df["net_foreign_value"].fillna(0).cumsum()
    return df

def idr_short(x: float) -> str:
    try:
        n = float(x)
    except Exception:
        return "-"
    a = abs(n)
    for nama, v in [("Triliun", 1e12), ("Miliar", 1e9), ("Juta", 1e6), ("Ribu", 1e3)]:
        if a >= v:
            return f"{n / v:,.2f}".replace(",", ".") + " " + nama
    return f"{n:,.0f}".replace(",", ".")

# === MACD
def get_macd_params():
    preset = st.session_state.get("macd_preset", "12-26-9 (Standard)")
    presets = {
        "12-26-9 (Standard)": (12, 26, 9),
        "5-35-5 (Fast)": (5, 35, 5),
        "8-17-9": (8, 17, 9),
        "10-30-9": (10, 30, 9),
        "20-50-9": (20, 50, 9),
    }
    if preset != "Custom":
        return presets.get(preset, (12, 26, 9))
    f = int(st.session_state.get("macd_fast", 12))
    s = int(st.session_state.get("macd_slow", 26)); s = max(s, f + 1)
    g = int(st.session_state.get("macd_signal", 9))
    return max(f, 1), s, max(g, 1)

def macd_series(close: pd.Series, fast: int, slow: int, sig: int):
    close = pd.to_numeric(close, errors="coerce").dropna()
    ema_f = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_s = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False, min_periods=1).mean()
    delta = macd - signal
    return macd, signal, delta

def macd_cross_flags(delta: pd.Series):
    prev = delta.shift(1)
    return (prev <= 0) & (delta > 0), (prev >= 0) & (delta < 0)

# === Bulk fetch untuk Scanner
@st.cache_data(ttl=600, show_spinner=False)
def fetch_ohlc_bulk(codes, start, end, chunk_size: int = 200):
    if not codes:
        return pd.DataFrame()
    con = get_db_connection(); _alive(con)
    try:
        cols = pd.read_sql(
            "SELECT LOWER(column_name) col FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'data_harian'", con
        )
        have = set(cols["col"].tolist())
        price_candidates = ["penutupan", "close_price", "closing", "close"]
        price_col = next((c for c in price_candidates if c in have), None)
        if price_col is None:
            raise RuntimeError("Butuh kolom harga penutupan (penutupan/close_price)")
        nf_expr = "net_foreign_value" if "net_foreign_value" in have else (
            "(foreign_buy - foreign_sell)" if {"foreign_buy", "foreign_sell"}.issubset(have) else "NULL"
        )
        parts = []
        start_param, end_param = pd.to_datetime(start), pd.to_datetime(end)
        for i in range(0, len(codes), max(1, int(chunk_size))):
            ch = codes[i:i + chunk_size]
            if not ch:
                continue
            placeholders = ",".join(["%s"] * len(ch))
            sql = (
                "SELECT kode_saham, trade_date, " + price_col + " AS close, " + nf_expr + " AS net_foreign_value "
                "FROM data_harian WHERE trade_date BETWEEN %s AND %s AND kode_saham IN (" + placeholders + ") "
                "ORDER BY kode_saham, trade_date"
            )
            params = [start_param, end_param] + list(ch)
            parts.append(pd.read_sql(sql, con, params=params))
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        if df.empty:
            return df
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        if "net_foreign_value" in df.columns:
            df["net_foreign_value"] = pd.to_numeric(df["net_foreign_value"], errors="coerce")
        return df.dropna(subset=["close"])
    finally:
        _close(con)

def scan_universe_fast(df_bulk, fast, slow, sig, nf_window=5, filter_nf=True, only_recent_days=15, require_above_zero=False):
    if df_bulk is None or df_bulk.empty:
        return pd.DataFrame()
    out = []
    global_end = pd.to_datetime(df_bulk["trade_date"].max())
    for kd, g in df_bulk.groupby("kode_saham"):
        g = g.sort_values("trade_date")
        close = g["close"]
        macd, signal, delta = macd_series(close, fast, slow, sig)
        bull, bear = macd_cross_flags(delta)
        last_type, last_date = None, None
        if bull.any():
            last_type, last_date = "Bullish", g.loc[bull, "trade_date"].iloc[-1]
        if bear.any():
            last_bear_date = g.loc[bear, "trade_date"].iloc[-1]
            if last_date is None or pd.to_datetime(last_bear_date) > pd.to_datetime(last_date):
                last_type, last_date = "Bearish", last_bear_date
        if last_date is None:
            continue
        days_ago = (global_end.normalize() - pd.to_datetime(last_date).normalize()).days
        nf_sum = np.nan; nf_ok = True
        if filter_nf and "net_foreign_value" in g.columns:
            nf = g["net_foreign_value"].fillna(0.0)
            nf_sum = float(nf.tail(int(nf_window)).sum()); nf_ok = nf_sum >= 0
        macd_above = bool(macd.iloc[-1] > 0)
        qualifies = True
        if only_recent_days is not None:
            qualifies = qualifies and days_ago <= int(only_recent_days)
        if require_above_zero:
            qualifies = qualifies and macd_above
        if filter_nf:
            qualifies = qualifies and nf_ok
        out.append({
            "kode": kd, "last_cross": last_type, "last_cross_date": pd.to_datetime(last_date).date(),
            "days_ago": int(days_ago), "macd_above_zero": macd_above, f"NF_sum_{int(nf_window)}d": nf_sum,
            "qualifies": bool(qualifies), "close_last": float(close.iloc[-1]),
        })
    return pd.DataFrame(out)

# === Backtest (SEMUA metrik di sini)
def backtest_macd(df_price, fast, slow, sig, require_above_zero=False, nf_window=0, require_nf=False, fee_bp=0):
    # pick close
    close = None
    if "penutupan" in df_price.columns and df_price["penutupan"].notna().any():
        close = pd.to_numeric(df_price["penutupan"], errors="coerce")
    elif "close_price" in df_price.columns and df_price["close_price"].notna().any():
        close = pd.to_numeric(df_price["close_price"], errors="coerce")
    if close is None or close.dropna().empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "trades": 0, "winrate": np.nan, "profit_factor": np.nan, "max_dd_pct": np.nan, "cagr_pct": np.nan,
            "total_return_pct": np.nan, "avg_trade_pct": np.nan, "median_trade_pct": np.nan,
            "best_trade_pct": np.nan, "worst_trade_pct": np.nan, "avg_hold_days": np.nan,
            "sharpe": np.nan, "vol_annual_pct": np.nan, "calmar_ratio": np.nan,
        }

    valid = close.notna()
    dates = pd.to_datetime(df_price.loc[valid, "trade_date"]).reset_index(drop=True)
    close_v = close.loc[valid].reset_index(drop=True)

    macd, signal, delta = macd_series(close_v, fast, slow, sig)
    bull, bear = macd_cross_flags(delta)

    nf_sum = None
    if require_nf and "net_foreign_value" in df_price.columns:
        nf_ser = pd.to_numeric(df_price["net_foreign_value"], errors="coerce").fillna(0)
        nf_sum_full = nf_ser.rolling(int(max(1, nf_window)), min_periods=1).sum()
        nf_sum = nf_sum_full.loc[valid].reset_index(drop=True)

    trades = []
    in_pos = False
    entry_price = None
    entry_date = None
    equity = 1.0
    fee = float(fee_bp) / 10000.0
    equity_curve = []

    for i in range(len(close_v)):
        d = dates.iloc[i]
        nf_ok = True
        if nf_sum is not None:
            nf_ok = nf_sum.iloc[i] >= 0
        cond_above = (macd.iloc[i] > 0) if require_above_zero else True

        if (not in_pos) and bool(bull.iloc[i]) and nf_ok and cond_above:
            in_pos = True
            entry_price = float(close_v.iloc[i])
            entry_date = d
        elif in_pos and bool(bear.iloc[i]):
            exit_price = float(close_v.iloc[i]); exit_date = d
            ret = (exit_price / entry_price - 1.0) - 2 * fee
            equity *= (1.0 + ret)
            trades.append({
                "entry_date": entry_date.date(), "entry_price": entry_price,
                "exit_date": exit_date.date(), "exit_price": exit_price,
                "ret_pct": ret * 100.0, "hold_days": int((exit_date.date() - entry_date.date()).days),
            })
            in_pos = False; entry_price = None; entry_date = None
        equity_curve.append({"date": d, "equity": equity})

    if in_pos:
        exit_price = float(close_v.iloc[-1]); exit_date = dates.iloc[-1]
        ret = (exit_price / entry_price - 1.0) - 2 * fee
        equity *= (1.0 + ret)
        trades.append({
            "entry_date": entry_date.date(), "entry_price": entry_price,
            "exit_date": exit_date.date(), "exit_price": exit_price,
            "ret_pct": ret * 100.0, "hold_days": int((exit_date.date() - entry_date.date()).days),
        })
        if equity_curve:
            equity_curve[-1]["equity"] = equity

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve)

    if trades_df.empty:
        stats = {
            "trades": 0, "winrate": np.nan, "profit_factor": np.nan, "max_dd_pct": np.nan, "cagr_pct": np.nan,
            "total_return_pct": (eq_df["equity"].iloc[-1] - 1.0) * 100.0 if not eq_df.empty else np.nan,
            "avg_trade_pct": np.nan, "median_trade_pct": np.nan, "best_trade_pct": np.nan, "worst_trade_pct": np.nan,
            "avg_hold_days": np.nan, "sharpe": np.nan, "vol_annual_pct": np.nan, "calmar_ratio": np.nan,
        }
        return trades_df, eq_df, stats

    wins = trades_df[trades_df["ret_pct"] > 0]
    losses = trades_df[trades_df["ret_pct"] <= 0]
    pf = (wins["ret_pct"].sum() / abs(losses["ret_pct"].sum())) if not losses.empty else float("inf")
    winrate = (len(wins) / len(trades_df)) * 100.0

    if not eq_df.empty:
        ec = eq_df.set_index("date")["equity"]
        roll_max = ec.cummax()
        max_dd = ((ec / roll_max) - 1.0).min() * 100.0
        days = (ec.index.max().date() - ec.index.min().date()).days if len(ec) > 1 else 0
        cagr = ((ec.iloc[-1]) ** (365.0 / days) - 1.0) * 100.0 if days > 0 else np.nan
        total_ret = (ec.iloc[-1] - 1.0) * 100.0
        daily_ret = ec.pct_change().dropna()
        vol_ann = (daily_ret.std() * math.sqrt(252)) * 100.0 if not daily_ret.empty else np.nan
        sharpe = (daily_ret.mean() / daily_ret.std() * math.sqrt(252)) if (len(daily_ret) > 1 and daily_ret.std() != 0) else np.nan
    else:
        max_dd = np.nan; cagr = np.nan; total_ret = np.nan; vol_ann = np.nan; sharpe = np.nan

    stats = {
        "trades": int(len(trades_df)),
        "winrate": float(winrate),
        "profit_factor": float(pf) if math.isfinite(pf) else float("inf"),
        "max_dd_pct": float(max_dd),
        "cagr_pct": float(cagr),
        "total_return_pct": float(total_ret),
        "avg_trade_pct": float(trades_df["ret_pct"].mean()),
        "median_trade_pct": float(trades_df["ret_pct"].median()),
        "best_trade_pct": float(trades_df["ret_pct"].max()),
        "worst_trade_pct": float(trades_df["ret_pct"].min()),
        "avg_hold_days": float(trades_df["hold_days"].mean()) if "hold_days" in trades_df.columns else np.nan,
        "sharpe": float(sharpe) if not pd.isna(sharpe) else np.nan,
        "vol_annual_pct": float(vol_ann) if not pd.isna(vol_ann) else np.nan,
        "calmar_ratio": float((cagr / abs(max_dd)) if (not pd.isna(cagr) and not pd.isna(max_dd) and max_dd != 0) else np.nan),
    }
    return trades_df, eq_df, stats

# === STATE DEFAULTS
codes = list_codes()
min_d, max_d = date_bounds()
st.session_state.setdefault("kode_saham", (codes[0] if codes else None))
st.session_state.setdefault("adv_mode", "All (Default)")
st.session_state.setdefault("date_range", (max(min_d, max_d - relativedelta(years=1)), max_d))
st.session_state.setdefault("range_choice", "1 Tahun")
st.session_state.setdefault("hide_non_trading", True)
st.session_state.setdefault("show_price", True)
st.session_state.setdefault("show_spread", True)
st.session_state.setdefault("macd_preset", "12-26-9 (Standard)")
st.session_state.setdefault("macd_fast", 12)
st.session_state.setdefault("macd_slow", 26)
st.session_state.setdefault("macd_signal", 9)
st.session_state.setdefault("mobile_mode", False)
st.session_state.setdefault("bt_symbol", st.session_state.get("kode_saham"))
st.session_state.setdefault("open_backtest", False)

# === Sticky filter bar
with st.container():
    st.markdown("<div class='sticky-filter'>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([1.3, 1.0, 1.6, 0.9])
    with f1: st.selectbox("Pilih saham", options=codes, key="kode_saham")
    with f2: st.selectbox("ADV window (hari)", ["All (Default)", "1 Tahun", "6 Bulan", "3 Bulan", "1 Bulan"], key="adv_mode")
    with f3:
        dr = st.date_input("Rentang tanggal",
            value=st.session_state.get("date_range"),
            min_value=min_d, max_value=max_d)
        if isinstance(dr, tuple) and dr != st.session_state.get("date_range"):
            st.session_state["date_range"] = dr
            st.session_state["range_choice"] = "Custom"
            st.rerun()
    with f4:
        st.checkbox("📱 Mobile Mode", key="mobile_mode",
                    help="Mode ringkas untuk layar HP (chart lebih pendek, kontrol ditumpuk).")
    st.markdown("</div>", unsafe_allow_html=True)

MOBILE = bool(st.session_state["mobile_mode"])
H_CANDLE = 360 if MOBILE else 460
H_MACD   = 260 if MOBILE else 320
H_OTHER  = 340 if MOBILE else 420
DF_H     = 260 if MOBILE else 360
SCAN_H   = 340 if MOBILE else 420
TRADES_H = 260 if MOBILE else 320

# === Quick range
kode = st.session_state["kode_saham"]
if not kode:
    st.stop()
td_series = get_trade_dates(kode)
td_min = td_series.min().date() if not td_series.empty else min_d
td_max = td_series.max().date() if not td_series.empty else max_d
st.radio("Range cepat", ["All", "1 Tahun", "6 Bulan", "3 Bulan", "1 Bulan", "5 Hari", "Custom"],
         key="range_choice", horizontal=not MOBILE)

def _apply_quick(choice):
    if choice == "Custom":
        return
    start, end = td_min, td_max
    if choice == "1 Tahun":
        end = td_max; start = max(td_min, td_max - relativedelta(years=1))
    elif choice == "6 Bulan":
        end = td_max; start = max(td_min, td_max - relativedelta(months=6))
    elif choice == "3 Bulan":
        end = td_max; start = max(td_min, td_max - relativedelta(months=3))
    elif choice == "1 Bulan":
        end = td_max; start = max(td_min, td_max - relativedelta(months=1))
    elif choice == "5 Hari":
        end = td_max
        if not td_series.empty and len(td_series) >= 5:
            start = td_series.iloc[-5].date()
        else:
            start = max(td_min, td_max - timedelta(days=7))
    elif choice == "All":
        start, end = td_min, td_max
    if st.session_state.get("date_range") != (start, end):
        st.session_state["date_range"] = (start, end); st.rerun()

_apply_quick(st.session_state["range_choice"])

# === Toggles
with st.container():
    st.markdown("<div class='checks-row'>", unsafe_allow_html=True)
    cA, cB, cC = st.columns(3)
    with cA: st.checkbox("Tampilkan harga (jika tersedia)", key="show_price")
    with cB: st.checkbox("Tampilkan spread (bps) jika tersedia", key="show_spread")
    with cC: st.checkbox("Hide non-trading days (skip weekend & libur bursa)", key="hide_non_trading")
    st.markdown("</div>", unsafe_allow_html=True)

# === MACD preset
m1, m2, m3, m4 = st.columns([1.6, .8, .8, .8])
with m1:
    st.selectbox("MACD preset",
        ["12-26-9 (Standard)", "5-35-5 (Fast)", "8-17-9", "10-30-9", "20-50-9", "Custom"], key="macd_preset")
if st.session_state["macd_preset"] == "Custom":
    with m2: st.number_input("Fast EMA", min_value=1, max_value=200, step=1, key="macd_fast")
    with m3: st.number_input("Slow EMA", min_value=2, max_value=400, step=1, key="macd_slow")
    with m4: st.number_input("Signal",   min_value=1, max_value=100, step=1, key="macd_signal")

# === Load & derive data
start, end = st.session_state["date_range"]
df_raw = load_series(kode, start, end)
if df_raw.empty:
    st.warning("Data kosong buat rentang ini.")
    st.stop()
df = add_rolling(ensure_metrics(df_raw), adv_mode=st.session_state["adv_mode"])

# === KPI
st.divider()
nama = (df["nama_perusahaan"].dropna().iloc[-1]
        if "nama_perusahaan" in df.columns and df["nama_perusahaan"].notna().any() else "-")
st.subheader(f"{kode} — {nama}")

# price series
price_series = None
if "penutupan" in df.columns and df["penutupan"].notna().any():
    price_series = pd.to_numeric(df["penutupan"], errors="coerce")
elif "close_price" in df.columns and df["close_price"].notna().any():
    price_series = pd.to_numeric(df["close_price"], errors="coerce")

total_buy  = df["foreign_buy"].sum(skipna=True) if "foreign_buy"  in df.columns else np.nan
total_sell = df["foreign_sell"].sum(skipna=True) if "foreign_sell" in df.columns else np.nan
total_nf   = df["net_foreign_value"].sum(skipna=True)

if MOBILE:
    a, b = st.columns(2)
    with a:
        st.metric("Total Net Foreign", idr_short(total_nf))
        st.metric("Total Foreign Buy", idr_short(total_buy) if pd.notna(total_buy) else "-")
    with b:
        pos_days = (df["net_foreign_value"] > 0).sum()
        st.metric("% Hari Net Buy", f"{(100 * pos_days / len(df)):.0f}%")
        st.metric("Total Foreign Sell", idr_short(total_sell) if pd.notna(total_sell) else "-")
    st.metric("Max Ratio (nilai/ADV)", f"{df['ratio'].max(skipna=True):.2f}x" if df["ratio"].notna().any() else "-")
else:
    m1a, m2a, m3a, m4a, m5a = st.columns(5)
    m1a.metric("Total Net Foreign", idr_short(total_nf))
    m2a.metric("Total Foreign Buy", idr_short(total_buy) if pd.notna(total_buy) else "-")
    m3a.metric("Total Foreign Sell", idr_short(total_sell) if pd.notna(total_sell) else "-")
    pos_days = (df["net_foreign_value"] > 0).sum()
    m4a.metric("% Hari Net Buy", f"{(100 * pos_days / len(df)):.0f}%")
    m5a.metric("Max Ratio (nilai/ADV)", f"{df['ratio'].max(skipna=True):.2f}x" if df["ratio"].notna().any() else "-")

if df["net_foreign_value"].notna().any():
    max_buy_day = df.loc[df["net_foreign_value"].idxmax()]
    max_sell_day = df.loc[df["net_foreign_value"].idxmin()]
    st.caption(f"🔎 Net Buy terbesar: **{max_buy_day['trade_date'].date()}** ({idr_short(max_buy_day['net_foreign_value'])})")
    st.caption(f"🔎 Net Sell terbesar: **{max_sell_day['trade_date'].date()}** ({idr_short(max_sell_day['net_foreign_value'])})")

# === Charts
# Candlestick + MA + MACD arrows
if price_series is not None and price_series.notna().any():
    open_series = None
    if "open_price" in df.columns and df["open_price"].notna().any():
        open_series = pd.to_numeric(df["open_price"], errors="coerce")
    elif "first_trade" in df.columns and df["first_trade"].notna().any():
        open_series = pd.to_numeric(df["first_trade"], errors="coerce")
    else:
        open_series = price_series.copy()
    high_series = pd.to_numeric(df.get("tertinggi"), errors="coerce") if "tertinggi" in df.columns else None
    low_series  = pd.to_numeric(df.get("terendah"),  errors="coerce") if "terendah"  in df.columns else None

    has_ohlc = (
        open_series is not None and high_series is not None and low_series is not None and
        open_series.notna().any() and high_series.notna().any() and low_series.notna().any() and price_series.notna().any()
    )
    if has_ohlc:
        mask = open_series.notna() & high_series.notna() & low_series.notna() & price_series.notna()
        df_cdl = df.loc[mask].copy()
        ma5  = price_series.loc[mask].rolling(5,  min_periods=1).mean()
        ma20 = price_series.loc[mask].rolling(20, min_periods=1).mean()
        figC = go.Figure()
        figC.add_trace(go.Candlestick(
            x=df_cdl["trade_date"], open=open_series[mask], high=high_series[mask],
            low=low_series[mask], close=price_series[mask],
            increasing_line_color="#33B766", decreasing_line_color="#DC3545", name="OHLC"
        ))
        figC.add_trace(go.Scatter(x=df_cdl["trade_date"], y=ma5,  name="MA 5",  line=dict(color="#0d6efd", width=1.8)))
        figC.add_trace(go.Scatter(x=df_cdl["trade_date"], y=ma20, name="MA 20", line=dict(color="#6c757d", width=1.8)))
        # Panah MACD
        try:
            f, s, g = get_macd_params()
            close_m = price_series.dropna()
            trade_m = df.loc[price_series.notna(), "trade_date"].reset_index(drop=True)
            ema_f = close_m.ewm(span=f, adjust=False, min_periods=1).mean()
            ema_s = close_m.ewm(span=s, adjust=False, min_periods=1).mean()
            macd_line = ema_f - ema_s
            signal_line = macd_line.ewm(span=g, adjust=False, min_periods=1).mean()
            delta = macd_line - signal_line
            prev = delta.shift(1)
            bull_idx = (prev <= 0) & (delta > 0)
            bear_idx = (prev >= 0) & (delta < 0)
            t_cdl = df_cdl["trade_date"].reset_index(drop=True)
            map_low  = pd.Series(low_series[mask].values,  index=t_cdl)
            map_high = pd.Series(high_series[mask].values, index=t_cdl)
            bull_dates = [d for d in trade_m[bull_idx] if d in map_low.index]
            bear_dates = [d for d in trade_m[bear_idx] if d in map_high.index]
            if bull_dates:
                yb = [float(map_low[d]) * 0.995 for d in bull_dates]
                figC.add_trace(go.Scatter(x=bull_dates, y=yb, mode="markers", name="Bullish cross (MACD)",
                               marker=dict(symbol="triangle-up", size=12, color="#22c55e")))
            if bear_dates:
                ya = [float(map_high[d]) * 1.005 for d in bear_dates]
                figC.add_trace(go.Scatter(x=bear_dates, y=ya, mode="markers", name="Bearish cross (MACD)",
                               marker=dict(symbol="triangle-down", size=12, color="#ef4444")))
        except Exception:
            pass

        figC.update_layout(
            title="Candlestick (OHLC) + MA5/MA20",
            xaxis_title=None, yaxis_title="Harga (IDR)",
            xaxis_rangeslider_visible=False, height=H_CANDLE,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40),
        )
        _apply_time_axis(figC, df_cdl["trade_date"], start, end, st.session_state["hide_non_trading"])
        st.plotly_chart(figC, use_container_width=True)
    else:
        st.info("Candlestick belum bisa ditampilkan (kolom OHLC tidak lengkap).")

# MACD panel
if price_series is not None and price_series.notna().any():
    sub_m = df[price_series.notna()][["trade_date"]].copy()
    close_m = price_series.dropna()
    f, s, g = get_macd_params()
    ema_f = close_m.ewm(span=f, adjust=False, min_periods=1).mean()
    ema_s = close_m.ewm(span=s, adjust=False, min_periods=1).mean()
    macd_line = ema_f - ema_s
    signal_line = macd_line.ewm(span=g, adjust=False, min_periods=1).mean()
    hist = macd_line - signal_line
    prev = (macd_line - signal_line).shift(1)
    bull = (prev <= 0) & (hist > 0)
    bear = (prev >= 0) & (hist < 0)
    x_bull = sub_m.loc[bull, "trade_date"]; y_bull = macd_line[bull]
    x_bear = sub_m.loc[bear, "trade_date"]; y_bear = macd_line[bear]

    figM = go.Figure()
    colorsM = np.where(hist >= 0, "rgba(51,183,102,0.7)", "rgba(220,53,69,0.7)")
    figM.add_bar(x=sub_m["trade_date"], y=hist, name="Histogram", marker_color=colorsM)
    figM.add_trace(go.Scatter(x=sub_m["trade_date"], y=macd_line, name="MACD", line=dict(width=2, color="#0d6efd")))
    figM.add_trace(go.Scatter(x=sub_m["trade_date"], y=signal_line, name="Signal", line=dict(width=2, color="#fd7e14")))
    figM.add_trace(go.Scatter(x=x_bull, y=y_bull, mode="markers", name="Bullish crossover",
                              marker=dict(symbol="triangle-up", size=12, color="#22c55e")))
    figM.add_trace(go.Scatter(x=x_bear, y=y_bear, mode="markers", name="Bearish crossover",
                              marker=dict(symbol="triangle-down", size=12, color="#ef4444")))
    figM.add_hline(y=0, line_color="#adb5bd", line_width=1)
    figM.update_layout(
        title=f"MACD ({f},{s},{g}) + Crossovers", xaxis_title=None, yaxis_title="MACD",
        height=H_MACD, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40),
    )
    _apply_time_axis(figM, sub_m["trade_date"], start, end, st.session_state["hide_non_trading"])
    st.plotly_chart(figM, use_container_width=True)

# Close vs NF
if price_series is not None and price_series.notna().any():
    fig1 = go.Figure()
    sub_nf = df[df["net_foreign_value"].notna()][["trade_date", "net_foreign_value"]]
    colors = np.where(sub_nf["net_foreign_value"] >= 0, "rgba(51,183,102,0.7)", "rgba(220,53,69,0.7)")
    fig1.add_bar(x=sub_nf["trade_date"], y=sub_nf["net_foreign_value"], name="Net Foreign (Rp)", marker_color=colors, yaxis="y2")
    sub_cl = df[price_series.notna()][["trade_date"]].assign(close=price_series.dropna())
    fig1.add_trace(go.Scatter(x=sub_cl["trade_date"], y=sub_cl["close"], name="Close", mode="lines+markers",
                              line=dict(color=THEME, width=2.2)))
    fig1.update_layout(
        title="Close vs Net Foreign Harian",
        xaxis_title=None, yaxis=dict(title="Close"),
        yaxis2=dict(title="Net Foreign (Rp)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=H_OTHER, margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40),
    )
    _apply_time_axis(fig1, df["trade_date"], start, end, st.session_state["hide_non_trading"])
    st.plotly_chart(fig1, use_container_width=True)

# Cum NF vs Close
if price_series is not None and price_series.notna().any():
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["trade_date"], y=df["cum_nf"], name="Cum. Net Foreign", line=dict(color="#6f42c1", width=2.2)))
    sub_cl = df[price_series.notna()][["trade_date"]].assign(close=price_series.dropna())
    fig2.add_trace(go.Scatter(x=sub_cl["trade_date"], y=sub_cl["close"], name="Close", yaxis="y2", line=dict(color=THEME, width=2), opacity=0.9))
    fig2.update_layout(
        title="Kumulatif Net Foreign vs Close",
        xaxis_title=None, yaxis=dict(title="Cum. Net Foreign"),
        yaxis2=dict(title="Close", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=H_OTHER, margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40),
    )
    _apply_time_axis(fig2, df["trade_date"], start, end, st.session_state["hide_non_trading"])
    st.plotly_chart(fig2, use_container_width=True)

# Volume vs AVG
if "volume" in df.columns:
    fig5 = go.Figure()
    sub_v  = df[df["volume"].notna()][["trade_date", "volume"]]
    fig5.add_bar(x=sub_v["trade_date"], y=sub_v["volume"], name="Volume")
    sub_va = df[df["vol_avg"].notna()][["trade_date", "vol_avg"]]
    fig5.add_trace(go.Scatter(x=sub_va["trade_date"], y=sub_va["vol_avg"], name="Vol AVG(20)"))
    fig5.update_layout(
        title="Volume vs AVG(20)", xaxis_title=None, yaxis_title="Lembar",
        height=H_OTHER, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40),
    )
    _apply_time_axis(fig5, sub_v["trade_date"], start, end, st.session_state["hide_non_trading"])
    st.plotly_chart(fig5, use_container_width=True)

# Spread (bps)
if st.session_state["show_spread"] and "spread_bps" in df.columns:
    sub_sp = df[df["spread_bps"].notna()][["trade_date", "spread_bps"]]
    if not sub_sp.empty:
        sp_vals = sub_sp["spread_bps"].dropna()
        p75 = float(np.nanpercentile(sp_vals, 75)) if len(sp_vals) else None
        fig6 = px.line(sub_sp, x="trade_date", y="spread_bps", title="Spread (bps)")
        if p75 is not None:
            fig6.add_hline(y=p75, line_dash="dot", line_color="#dc3545", annotation_text=f"P75 ≈ {p75:.1f}")
        fig6.update_layout(height=H_OTHER, xaxis_title=None, yaxis_title="bps",
                           margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40))
        _apply_time_axis(fig6, sub_sp["trade_date"], start, end, st.session_state["hide_non_trading"])
        st.plotly_chart(fig6, use_container_width=True)

st.divider()
with st.expander("Tabel data mentah (siap export)"):
    cols_show = [
        "trade_date","kode_saham","nama_perusahaan","sebelumnya","open_price","first_trade",
        "tertinggi","terendah","penutupan","selisih","nilai","adv","ratio","volume","vol_avg",
        "foreign_buy","foreign_sell","net_foreign_value","spread_bps","close_price","cum_nf",
    ]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show], use_container_width=True, height=DF_H)
    st.download_button("⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{kode}_foreign_flow_{start}_to_{end}.csv", mime="text/csv")

st.caption("💡 Aktifkan Mobile Mode untuk tampilan ringkas di HP.")

# ============================
# Scanner — MACD Cross + Net Foreign (persist)
# ============================
st.divider()
with st.expander("🔎 Scanner — MACD Cross + Net Foreign", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1: scan_all = st.checkbox("Scan semua kode", value=True, key="scan_all")
    with c2: recency  = st.number_input("Cross dalam X hari terakhir", min_value=1, max_value=365, value=15, step=1, key="scan_recency")
    with c3: require_above_zero = st.checkbox("Syarat MACD > 0", value=False, key="scan_macdpos")
    d1, d2, _ = st.columns(3)
    with d1: require_nf = st.checkbox("Syarat NF rolling ≥ 0", value=True, key="scan_reqnf")
    with d2: nf_window  = st.number_input("Window NF (hari)", min_value=1, max_value=30, value=5, step=1, key="scan_nfwin")

    watchlist = codes if scan_all else st.multiselect("Watchlist", options=codes, default=[kode] if kode else [], key="scan_watchlist")
    run_scan = st.button("🚀 Jalankan Scan", type="primary", key="btn_scan")

    df_scan = None
    if run_scan:
        with st.spinner("Mengambil data & scanning..."):
            f, s, g = get_macd_params()
            warmup_days = max(s * 5, 150)
            approx_days = int(warmup_days + recency + int(nf_window) + 30)
            start_scan = max(min_d, end - timedelta(days=approx_days))
            df_bulk = fetch_ohlc_bulk(watchlist, start_scan, end)
            df_scan = scan_universe_fast(
                df_bulk, f, s, g, nf_window=int(nf_window), filter_nf=bool(require_nf),
                only_recent_days=int(recency), require_above_zero=bool(require_above_zero)
            )
        st.session_state["scanner_df"] = df_scan
        st.session_state["scanner_meta"] = {"start_scan": start_scan, "end": end, "nf_window": int(nf_window),
                                            "recency": int(recency), "require_nf": bool(require_nf),
                                            "require_above_zero": bool(require_above_zero)}
    else:
        df_scan = st.session_state.get("scanner_df")

    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("🧹 Clear hasil scan", key="btn_clear_scan"):
            for k in ["scanner_df", "scanner_meta"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.stop()

    if df_scan is None or df_scan.empty:
        st.caption("Klik Jalankan Scan untuk mulai. Hasil scan disimpan agar tidak hilang saat klik tombol lain.")
    else:
        nf_col = next((c for c in df_scan.columns if c.startswith("NF_sum_") and c.endswith("d")), None)
        sort_cols, asc = [], []
        if "qualifies" in df_scan.columns: sort_cols.append("qualifies"); asc.append(False)
        if "days_ago" in df_scan.columns: sort_cols.append("days_ago"); asc.append(True)
        if "last_cross_date" in df_scan.columns: sort_cols.append("last_cross_date"); asc.append(False)
        if nf_col: sort_cols.append(nf_col); asc.append(False)
        df_scan_sorted = df_scan.sort_values(sort_cols, ascending=asc) if sort_cols else df_scan.copy()

        # chips
        st.markdown("**Quick filters**")
        ch1, ch2, ch3 = st.columns(3)
        with ch1: chip_recent3 = st.checkbox("≤ 3 hari", key="chip_recent3", value=False)
        with ch2:
            if nf_col and df_scan[nf_col].notna().any():
                nf_p75_val = float(np.nanpercentile(df_scan[nf_col], 75))
                chip_nf_p75 = st.checkbox(f"NF ≥ p75 (≈ {nf_p75_val:,.0f})", key="chip_nf_p75", value=False)
            else:
                nf_p75_val = None
                chip_nf_p75 = st.checkbox("NF ≥ p75", key="chip_nf_p75", value=False, disabled=True)
        with ch3: chip_macd_pos = st.checkbox("MACD > 0", key="chip_macd_pos", value=False)

        view = df_scan_sorted.copy()
        if chip_recent3 and "days_ago" in view.columns: view = view[view["days_ago"] <= 3]
        if chip_macd_pos and "macd_above_zero" in view.columns: view = view[view["macd_above_zero"] == True]
        if chip_nf_p75 and nf_col and nf_p75_val is not None: view = view[view[nf_col] >= nf_p75_val]

        order_cols = ["qualifies", "days_ago", "last_cross_date", "kode", "last_cross", "macd_above_zero", nf_col, "close_last"]
        order_cols = [c for c in order_cols if c and c in view.columns]
        st.dataframe(view[order_cols], use_container_width=True, height=SCAN_H)

        if nf_col and nf_p75_val is not None:
            quick_watch = df_scan_sorted.copy()
            quick_watch = quick_watch[(quick_watch["days_ago"] <= 3) & (quick_watch[nf_col] >= nf_p75_val)]
            st.download_button("⬇️ Export Watchlist (≤3D & NF≥p75)",
                               data=quick_watch.to_csv(index=False).encode("utf-8"),
                               file_name=f"watchlist_quick_{st.session_state.get('scanner_meta', {}).get('start_scan', start)}_to_{end}.csv",
                               mime="text/csv")
        st.download_button("⬇️ Export Current View",
                           data=view.to_csv(index=False).encode("utf-8"),
                           file_name="scan_current_view.csv", mime="text/csv")

        st.markdown("**Kirim ke Backtest (per-ticker)**")
        if view.empty:
            st.caption("— Tidak ada kandidat pada filter yang dipilih.")
        else:
            for i, (_, row) in enumerate(view.head(50).iterrows()):
                cA, cB, cC, cD, cE = st.columns([1.2, 1, .8, 1.2, 1])
                with cA: st.write(f"**{row.get('kode', '-')}**")
                with cB: st.write(row.get("last_cross", "-"))
                with cC:
                    dago = row.get("days_ago", np.nan)
                    st.write(f"D{int(dago)}" if pd.notna(dago) else "-")
                with cD:
                    if nf_col:
                        val = row.get(nf_col, np.nan)
                        st.write(f"NF: {val:,.0f}" if pd.notna(val) else "NF: -")
                    else:
                        st.write("NF: -")
                with cE:
                    if st.button("Kirim ke Backtest", key=f"send_bt_{i}_{row.get('kode','?')}"):
                        st.session_state["bt_symbol"] = row.get("kode")
                        st.session_state["open_backtest"] = True

# ============================
# Backtest — MACD Rules (simple) (popover ℹ️)
# ============================
st.divider()
with st.expander("🧪 Backtest — MACD Rules (simple)", expanded=st.session_state.get("open_backtest", False)):
    c1, c2, c3, c4 = st.columns([1.2, .9, .9, .9])
    current_bt_symbol = st.session_state.get("bt_symbol", st.session_state.get("kode_saham"))
    try:
        idx_default = codes.index(current_bt_symbol) if (codes and current_bt_symbol in codes) else 0
    except Exception:
        idx_default = 0
    with c1:
        bt_symbol = st.selectbox("Kode saham", options=codes, index=idx_default if idx_default < len(codes) else 0, key="bt_sym_sel")
        st.session_state["bt_symbol"] = bt_symbol
    with c2: bt_fee = st.number_input("Biaya/Slippage (bps per sisi)", min_value=0, max_value=100, value=0, step=1, key="bt_fee")
    with c3: bt_require_above = st.checkbox("Entry hanya jika MACD > 0", value=False, key="bt_above")
    with c4: bt_require_nf = st.checkbox("Entry hanya jika NF rolling ≥ 0", value=False, key="bt_reqnf")
    c5, _ = st.columns([0.9, 2.1])
    with c5: bt_nf_window = st.number_input("NF window (hari)", min_value=1, max_value=30, value=5, step=1, key="bt_nfwin")

    df_bt = load_series(bt_symbol, start, end)
    if df_bt.empty:
        st.info("Data kosong untuk backtest.")
    else:
        with st.spinner("Menjalankan backtest..."):
            try:
                f, s, g = get_macd_params()
                trades, eq_df, stats = backtest_macd(
                    df_bt, f, s, g,
                    require_above_zero=bool(bt_require_above),
                    nf_window=int(bt_nf_window),
                    require_nf=bool(bt_require_nf),
                    fee_bp=int(bt_fee),
                )
            except Exception as e:
                st.error("Backtest error: " + str(e))
                trades, eq_df, stats = pd.DataFrame(), pd.DataFrame(), {
                    "trades": 0, "winrate": np.nan, "profit_factor": np.nan, "max_dd_pct": np.nan, "cagr_pct": np.nan,
                    "total_return_pct": np.nan, "avg_trade_pct": np.nan, "median_trade_pct": np.nan,
                    "best_trade_pct": np.nan, "worst_trade_pct": np.nan, "avg_hold_days": np.nan,
                    "sharpe": np.nan, "vol_annual_pct": np.nan, "calmar_ratio": np.nan,
                }

        TT = {
            "Trades": "Jumlah posisi (entry→exit) selama periode backtest.",
            "Win Rate": "Persentase trade yang return-nya > 0 setelah biaya.",
            "Profit Factor": "Total profit / total loss. >1 = profit; ∞ jika tidak ada loss.",
            "Max DD": "Maximum drawdown: penurunan terdalam dari puncak equity ke lembah (%).",
            "Total Return": "Kenaikan equity total dari awal ke akhir (reinvest penuh).",
            "CAGR": "Compounded Annual Growth Rate (tahunan; jika rentang >0 hari).",
            "Avg Trade": "Rata-rata % return per trade (setelah biaya).",
            "Expectancy": "Ekspektasi rata-rata per trade (≈ Avg Trade).",
            "Median Trade": "Median % return per trade.",
            "Avg Hold": "Rata-rata lama memegang posisi (hari).",
            "Best Trade": "Return % terbaik dari satu trade.",
            "Worst Trade": "Return % terburuk dari satu trade.",
        }

        def fmt_pct(x, d=1):
            return "-" if x is None or pd.isna(x) else f"{float(x):.{d}f}%"
        pf = stats.get("profit_factor", np.nan)
        pf_str = "∞" if (isinstance(pf, (float, int, np.floating)) and not math.isfinite(float(pf))) else (f"{float(pf):.2f}" if pf == pf else "-")
        vals = {
            "Trades": f"{int(stats.get('trades', 0))}",
            "Win Rate": fmt_pct(stats.get('winrate'), 1),
            "Profit Factor": pf_str,
            "Max DD": fmt_pct(stats.get('max_dd_pct'), 1),
            "Total Return": fmt_pct(stats.get('total_return_pct'), 1),
            "CAGR": fmt_pct(stats.get('cagr_pct'), 1),
            "Avg Trade": fmt_pct(stats.get('avg_trade_pct'), 2),
            "Expectancy": fmt_pct(stats.get('avg_trade_pct'), 2),
            "Median Trade": fmt_pct(stats.get('median_trade_pct'), 2),
            "Avg Hold": (f"{float(stats.get('avg_hold_days')):.1f} hari"
                        if stats.get('avg_hold_days', np.nan) == stats.get('avg_hold_days', np.nan) else "-"),
            "Best Trade": fmt_pct(stats.get('best_trade_pct'), 2),
            "Worst Trade": fmt_pct(stats.get('worst_trade_pct'), 2),
        }

        def metric_card(label: str, value: str):
            html = """
<div class="mcard">
  <div class="mcard-top">
     <div class="mcard-label">{lbl}</div>
     <details class="mcard-info">
       <summary aria-label="Info">ℹ️</summary>
       <div class="mcard-pop">{desc}</div>
     </details>
  </div>
  <div class="mcard-value">{val}</div>
</div>
""".format(lbl=escape(label), desc=escape(TT.get(label, "")), val=escape(value))
            st.markdown(html, unsafe_allow_html=True)

        st.markdown('<div class="mcards">', unsafe_allow_html=True)
        for label in ["Trades", "Win Rate", "Profit Factor", "Max DD", "Total Return", "CAGR",
                      "Avg Trade", "Expectancy", "Median Trade", "Avg Hold", "Best Trade", "Worst Trade"]:
            metric_card(label, vals[label])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f"<div class='metric-note'>Vol tahunan ≈ {fmt_pct(stats.get('vol_annual_pct'),1)} · "
            f"Sharpe ≈ {'-' if pd.isna(stats.get('sharpe')) else f'{float(stats.get('sharpe')):.2f}'} · "
            f"Calmar ≈ {'-' if pd.isna(stats.get('calmar_ratio')) else f'{float(stats.get('calmar_ratio')):.2f}'}</div>",
            unsafe_allow_html=True,
        )

        if not eq_df.empty:
            figEQ = px.line(eq_df, x="date", y="equity", title="Equity Curve (1.0 = awal)")
            figEQ.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8", annotation_text="Start")
            figEQ.update_traces(line={'width': 2.2})
            figEQ.update_layout(height=H_MACD, margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40))
            _apply_time_axis(figEQ, pd.to_datetime(eq_df["date"]), start, end, st.session_state["hide_non_trading"])
            st.plotly_chart(figEQ, use_container_width=True)
            st.download_button("⬇️ Download Equity Curve CSV",
                               data=eq_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"equity_curve_{bt_symbol}_{start}_to_{end}.csv", mime="text/csv")

        if not trades.empty:
            tv = trades.copy()
            if "entry_price" in tv.columns: tv["entry_price"] = tv["entry_price"].map(lambda x: f"{x:,.0f}".replace(",", "."))
            if "exit_price" in tv.columns:  tv["exit_price"]  = tv["exit_price"].map(lambda x: f"{x:,.0f}".replace(",", "."))
            if "ret_pct" in tv.columns:     tv["ret_pct"]     = tv["ret_pct"].map(lambda x: f"{x:.2f}%")
            if "hold_days" in tv.columns:   tv["hold_days"]   = tv["hold_days"].astype(int)
            cols = [c for c in ["entry_date", "entry_price", "exit_date", "exit_price", "hold_days", "ret_pct"] if c in tv.columns]
            st.dataframe(tv[cols], use_container_width=True, height=TRADES_H)
            st.download_button("⬇️ Download Trades CSV",
                               data=trades.to_csv(index=False).encode("utf-8"),
                               file_name=f"trades_{bt_symbol}_{start}_to_{end}.csv", mime="text/csv")
        else:
            st.info("Tidak ada trade pada parameter/rule ini.")
