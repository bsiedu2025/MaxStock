# app/pages/13_Foreign_Flow_Detail.py
# Daily Stock Movement – Foreign Flow Focus
# v3.5 UIX compact: filter utama 1 baris (Saham, ADV, Rentang Tanggal)
# - Quick range radio tetap di bawahnya
# - Toggle Hide non-trading di ujung kanan baris filter
# - Hindari Session State warning (pakai key saja, default via session_state)

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import math
import streamlit as st
from typing import Optional
from datetime import timedelta, date
import plotly.express as px
import plotly.graph_objects as go

from db_utils import get_db_connection, get_db_name

THEME = "#3AA6A0"
st.set_page_config(page_title="📈 Pergerakan Harian (Foreign Flow)", page_icon="📈", layout="wide")

# --- Minimal styling (pills & spacing) ---
st.markdown(
    """
    <style>
    section.main > div { padding-top: .5rem; }
    /* radio as segmented pills */
    div[role='radiogroup'] { display: flex; flex-wrap: wrap; gap: .25rem .5rem; }
    div[role='radiogroup'] label { border:1px solid #e5e7eb; padding:6px 12px; border-radius:9999px; background:#fff; }
    /* compact labels */
    .stSelectbox > label, .stDateInput > label, .stRadio > label { font-weight: 600; }
    /* inline checks */
    .checks-row { display:flex; align-items:center; gap: 1.25rem; }
    .checks-row { display:flex; align-items:center; gap: 1.25rem; }
    .checks-row div[data-testid='stCheckbox'] { margin-bottom: 0 !important; }
    /* sticky filter bar */
    .sticky-filter { position: sticky; top: 0; z-index: 1000; background: rgba(255,255,255,.96); backdrop-filter: blur(6px); border-bottom: 1px solid #e5e7eb; padding: .5rem .25rem .75rem .25rem; box-shadow: 0 8px 16px rgba(0,0,0,.06), 0 1px 0 rgba(0,0,0,.04); transition: box-shadow .2s ease, background .2s ease; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Pergerakan Harian Saham — Fokus Foreign Flow")
st.caption(f"DB aktif: **{get_db_name()}**")

# ---------- Helpers ----------
def _alive(conn):
    try: conn.reconnect(attempts=2, delay=1)
    except Exception: pass

def _close(conn):
    try:
        if hasattr(conn, "is_connected"):
            if conn.is_connected(): conn.close()
        else:
            conn.close()
    except Exception: pass

@st.cache_data(ttl=300)
def list_codes():
    con = get_db_connection(); _alive(con)
    try:
        s = pd.read_sql("SELECT DISTINCT kode_saham FROM data_harian ORDER BY 1", con)
        return s["kode_saham"].tolist()
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
        return pd.to_datetime(df["trade_date"])  # Series datetime
    finally:
        _close(con)

@st.cache_data(ttl=300, show_spinner=False)
def load_series(kode: str, start: date, end: date) -> pd.DataFrame:
    con = get_db_connection(); _alive(con)
    try:
        cols_df = pd.read_sql(
            "SELECT LOWER(column_name) col FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'data_harian'",
            con
        )
        avail = set(cols_df["col"].tolist())

        base_cols = ["trade_date", "kode_saham"]
        if "nama_perusahaan" in avail:
            base_cols.append("nama_perusahaan")

        price_cols = ["sebelumnya","open_price","first_trade","tertinggi","terendah","penutupan","selisih"]
        optional   = ["nilai","volume","freq","foreign_buy","foreign_sell","net_foreign_value","bid","offer","spread_bps","close_price"]
        select_cols = [c for c in base_cols + price_cols + optional if c in avail]
        if "trade_date" not in select_cols or "kode_saham" not in select_cols:
            raise RuntimeError("Kolom minimal 'trade_date' dan 'kode_saham' harus ada di data_harian")

        sql = f"""
        SELECT {', '.join(select_cols)}
        FROM data_harian
        WHERE kode_saham=%s AND trade_date BETWEEN %s AND %s
        ORDER BY trade_date
        """
        df = pd.read_sql(sql, con, params=[kode, start, end])

        all_want = set(price_cols + optional + ["nama_perusahaan"])
        for c in (all_want - set(df.columns.str.lower())):
            df[c] = np.nan

        ordered = base_cols + price_cols + [c for c in optional if c in df.columns]
        df = df[[c for c in ordered if c in df.columns]]
        return df
    finally:
        _close(con)

# === rangebreaks util ===
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

def _apply_time_axis(fig, trade_dates: pd.Series, start: date, end: date, hide_non_trading: bool):
    fig.update_xaxes(rangebreaks=_compute_rangebreaks(trade_dates, start, end, hide_non_trading))
    return fig

# === metrics helper ===

def ensure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "net_foreign_value" not in df.columns or df["net_foreign_value"].isna().all():
        if {"foreign_buy","foreign_sell"}.issubset(df.columns):
            df["foreign_buy"]  = pd.to_numeric(df.get("foreign_buy"), errors="coerce")
            df["foreign_sell"] = pd.to_numeric(df.get("foreign_sell"), errors="coerce")
            df["net_foreign_value"] = df["foreign_buy"].fillna(0) - df["foreign_sell"].fillna(0)
        else:
            df["net_foreign_value"] = np.nan

    if "spread_bps" not in df.columns or df["spread_bps"].isna().all():
        if {"bid","offer"}.issubset(df.columns):
            b = pd.to_numeric(df["bid"], errors="coerce")
            o = pd.to_numeric(df["offer"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                sp = (o - b) / ((o + b)/2) * 10000
            sp[(b<=0)|(o<=0)|(b.isna())|(o.isna())] = np.nan
            df["spread_bps"] = sp
        else:
            df["spread_bps"] = np.nan

    for c in ["nilai","volume","foreign_buy","foreign_sell","net_foreign_value","close_price","penutupan","sebelumnya","tertinggi","terendah","selisih","open_price","first_trade"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["trade_date"] = pd.to_datetime(df["trade_date"])  # datetime index
    return df

# rolling & derived

def add_rolling(df: pd.DataFrame, adv_mode: str) -> pd.DataFrame:
    df = df.sort_values("trade_date").copy()
    nilai = pd.to_numeric(df.get("nilai"), errors="coerce")

    mode_to_win = {"1 Bulan":20, "3 Bulan":60, "6 Bulan":120, "1 Tahun":252}
    if adv_mode == "All (Default)":
        df["adv"] = nilai.expanding().mean()
        adv_label = "All"
    else:
        win = mode_to_win.get(adv_mode, 20)
        df["adv"] = nilai.rolling(win, min_periods=1).mean()
        adv_label = adv_mode

    df["vol_avg"] = pd.to_numeric(df.get("volume"), errors="coerce").rolling(20, min_periods=1).mean()
    df["ratio"] = df["nilai"] / df["adv"]
    df["cum_nf"] = df["net_foreign_value"].fillna(0).cumsum()
    df.attrs["adv_label"] = adv_label
    return df

# utils

def idr_short(x: float) -> str:
    try:
        n = float(x)
    except Exception:
        return "-"
    a = abs(n)
    for nama,v in [("Triliun",1e12),("Miliar",1e9),("Juta",1e6),("Ribu",1e3)]:
        if a >= v:
            s = f"{n/v:,.2f}".replace(",", ".")
            return f"{s} {nama}"
    return f"{n:,.0f}".replace(",", ".")

def fmt_pct(v):
    return "-" if v is None or pd.isna(v) else f"{v:,.2f}%".replace(",", ".")

def format_money(v, dec=0):
    try: return f"{float(v):,.{dec}f}".replace(",", ".")
    except: return "-"

# --- MACD params helper ---

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
    # Custom
    f = int(st.session_state.get("macd_fast", 12))
    s = int(st.session_state.get("macd_slow", 26))
    g = int(st.session_state.get("macd_signal", 9))
    # guard: ensure fast < slow and >=1
    if f < 1: f = 1
    if s <= f: s = f + 1
    if g < 1: g = 1
    return f, s, g

# --- MACD core helpers ---

def macd_series(close, fast, slow, sig):
    close = pd.to_numeric(close, errors="coerce").dropna()
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig, adjust=False, min_periods=1).mean()
    delta = macd - signal
    return macd, signal, delta


def macd_cross_flags(delta):
    prev = delta.shift(1)
    bull = (prev <= 0) & (delta > 0)
    bear = (prev >= 0) & (delta < 0)
    return bull, bear


def scan_universe(kodes, start, end, fast, slow, sig,
                  nf_window=5, filter_nf=True, only_recent_days=15,
                  require_above_zero=False):
    rows = []
    for kd in kodes:
        try:
            dfk = load_series(kd, start, end)
            if dfk.empty:
                continue
            close = None
            if "penutupan" in dfk.columns and dfk["penutupan"].notna().any():
                close = pd.to_numeric(dfk["penutupan"], errors="coerce")
            elif "close_price" in dfk.columns and dfk["close_price"].notna().any():
                close = pd.to_numeric(dfk["close_price"], errors="coerce")
            if close is None or close.dropna().empty:
                continue
            valid = close.notna()
            dates = pd.to_datetime(dfk.loc[valid, "trade_date"]).reset_index(drop=True)
            close_v = close.loc[valid].reset_index(drop=True)

            macd, signal, delta = macd_series(close_v, fast, slow, sig)
            bull, bear = macd_cross_flags(delta)
            last_type, last_date = None, None
            if bull.any():
                last_bull = dates[bull].iloc[-1]
                last_type, last_date = "Bullish", last_bull
            if bear.any():
                last_bear = dates[bear].iloc[-1]
                if last_date is None or pd.to_datetime(last_bear) > pd.to_datetime(last_date):
                    last_type, last_date = "Bearish", last_bear
            if last_date is None:
                continue
            days_ago = (pd.to_datetime(end) - pd.to_datetime(last_date).normalize()).days

            nf_sum = np.nan
            nf_ok = True
            if filter_nf and "net_foreign_value" in dfk.columns:
                nf_ser = pd.to_numeric(dfk["net_foreign_value"], errors="coerce").fillna(0)
                nf_sum = nf_ser.rolling(int(nf_window), min_periods=1).sum().iloc[-1]
                nf_ok = nf_sum >= 0

            macd_above = bool(macd.iloc[-1] > 0)
            qualifies = True
            if require_above_zero:
                qualifies = qualifies and macd_above
            if filter_nf:
                qualifies = qualifies and nf_ok
            if only_recent_days is not None:
                qualifies = qualifies and (days_ago <= int(only_recent_days))

            rows.append({
                "kode": kd,
                "last_cross": last_type,
                "last_cross_date": pd.to_datetime(last_date).date(),
                "days_ago": int(days_ago),
                "macd_above_zero": macd_above,
                f"NF_sum_{int(nf_window)}d": float(nf_sum) if not np.isnan(nf_sum) else np.nan,
                "qualifies": bool(qualifies),
                "close_last": float(close_v.iloc[-1])
            })
        except Exception:
            # skip problematic code but keep scanning others
            continue
    return pd.DataFrame(rows)


def backtest_macd(df_price, fast, slow, sig,
                   require_above_zero=False, nf_window=0, require_nf=False,
                   fee_bp=0)::
    # choose close
    close = None
    if "penutupan" in df_price.columns and df_price["penutupan"].notna().any():
        close = pd.to_numeric(df_price["penutupan"], errors="coerce")
    elif "close_price" in df_price.columns and df_price["close_price"].notna().any():
        close = pd.to_numeric(df_price["close_price"], errors="coerce")
    if close is None or close.dropna().empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "trades": 0, "winrate": np.nan, "profit_factor": np.nan,
            "max_dd_pct": np.nan, "cagr_pct": np.nan, "total_return_pct": np.nan
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
    in_pos, entry_price, entry_date = False, None, None
    equity = 1.0
    fee = float(fee_bp) / 10000.0
    equity_curve = []

    for i in range(len(close_v)):
        d = dates.iloc[i]
        nf_ok = True
        if nf_sum is not None:
            nf_ok = nf_sum.iloc[i] >= 0
        cond_above = (macd.iloc[i] > 0) if require_above_zero else True

        if (not in_pos) and bull.iloc[i] and nf_ok and cond_above:
            in_pos = True
            entry_price = float(close_v.iloc[i])
            entry_date = d
        elif in_pos and bear.iloc[i]:
            exit_price = float(close_v.iloc[i])
            exit_date = d
            ret = (exit_price / entry_price - 1.0) - 2 * fee
            equity *= (1.0 + ret)
            trades.append({
                "entry_date": entry_date.date(),
                "entry_price": entry_price,
                "exit_date": exit_date.date(),
                "exit_price": exit_price,
                "ret_pct": ret * 100.0
            })
            in_pos = False
            entry_price = None
            entry_date = None
        equity_curve.append({"date": d, "equity": equity})

    # close open trade at end
    if in_pos:
        exit_price = float(close_v.iloc[-1])
        exit_date = dates.iloc[-1]
        ret = (exit_price / entry_price - 1.0) - 2 * fee
        equity *= (1.0 + ret)
        trades.append({
            "entry_date": entry_date.date(),
            "entry_price": entry_price,
            "exit_date": exit_date.date(),
            "exit_price": exit_price,
            "ret_pct": ret * 100.0
        })
        if equity_curve:
            equity_curve[-1]["equity"] = equity

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve)

    if trades_df.empty:
        stats = {
            "trades": 0,
            "winrate": np.nan,
            "profit_factor": np.nan,
            "max_dd_pct": np.nan,
            "cagr_pct": np.nan,
            "total_return_pct": (eq_df["equity"].iloc[-1] - 1.0) * 100.0 if not eq_df.empty else np.nan,
        }
        return trades_df, eq_df, stats

# -------- Bulk fetch for fast scanner --------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_ohlc_bulk(codes, start, end, chunk_size=200):
    """Bulk fetch close & NF untuk banyak kode dengan fallback kolom dinamis dan chunking.
    Menghindari error kolom tidak ada dan query IN yang terlalu panjang.
    """
    if not codes:
        return pd.DataFrame()
    con = get_db_connection(); _alive(con)
    try:
        # --- Deteksi kolom tersedia ---
        cols = pd.read_sql(
            "SELECT LOWER(column_name) AS col FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'data_harian'", con
        )
        have = set(cols['col'].tolist())
        # harga penutupan candidates
        price_candidates = ['penutupan', 'close_price', 'closing', 'close']
        price_col = next((c for c in price_candidates if c in have), None)
        if price_col is None:
            raise RuntimeError("Tidak ditemukan kolom harga penutupan pada data_harian (butuh salah satu dari penutupan/close_price/closing/close)")
        # net foreign:
        nf_expr = None
        if 'net_foreign_value' in have:
            nf_expr = 'net_foreign_value'
        elif {'foreign_buy','foreign_sell'}.issubset(have):
            nf_expr = '(foreign_buy - foreign_sell)'
        else:
            nf_expr = 'NULL'

        # --- Ambil per chunk ---
        all_parts = []
        start_param = pd.to_datetime(start)
        end_param = pd.to_datetime(end)
        for i in range(0, len(codes), max(1, int(chunk_size))):
            chunk = codes[i:i+chunk_size]
            if not chunk:
                continue
            placeholders = ','.join(['%s'] * len(chunk))
            sql = f"""
                SELECT kode_saham,
                       trade_date,
                       {price_col} AS close,
                       {nf_expr} AS net_foreign_value
                FROM data_harian
                WHERE trade_date BETWEEN %s AND %s
                  AND kode_saham IN ({placeholders})
                ORDER BY kode_saham, trade_date
            """
            params = [start_param, end_param] + list(chunk)
            part = pd.read_sql(sql, con, params=params)
            all_parts.append(part)
        if not all_parts:
            return pd.DataFrame()
        df = pd.concat(all_parts, ignore_index=True)
        if df.empty:
            return df
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        if 'net_foreign_value' in df.columns:
            df['net_foreign_value'] = pd.to_numeric(df['net_foreign_value'], errors='coerce')
        df = df.dropna(subset=['close'])
        return df
    finally:
        _close(con)


def scan_universe_fast(df_bulk, fast, slow, sig,
                       nf_window=5, filter_nf=True,
                       only_recent_days=15,
                       require_above_zero=False):
    if df_bulk is None or df_bulk.empty:
        return pd.DataFrame()
    out = []
    global_end = pd.to_datetime(df_bulk['trade_date'].max()) if not df_bulk.empty else pd.to_datetime(end)
    for kd, g in df_bulk.groupby('kode_saham'):
        g = g.sort_values('trade_date').copy()
        close = g['close']
        macd, signal, delta = macd_series(close, fast, slow, sig)
        bull, bear = macd_cross_flags(delta)
        last_type, last_date = None, None
        if bull.any():
            last_type, last_date = 'Bullish', g.loc[bull, 'trade_date'].iloc[-1]
        if bear.any():
            last_bear_date = g.loc[bear, 'trade_date'].iloc[-1]
            if last_date is None or pd.to_datetime(last_bear_date) > pd.to_datetime(last_date):
                last_type, last_date = 'Bearish', last_bear_date
        if last_date is None:
            continue
        days_ago = (global_end.normalize() - pd.to_datetime(last_date).normalize()).days
        nf_sum = np.nan
        nf_ok = True
        if filter_nf and 'net_foreign_value' in g.columns:
            nf = g['net_foreign_value'].fillna(0.0)
            nf_sum = float(nf.tail(int(nf_window)).sum())
            nf_ok = nf_sum >= 0
        macd_above = bool(macd.iloc[-1] > 0)
        qualifies = True
        if only_recent_days is not None:
            qualifies &= (days_ago <= int(only_recent_days))
        if require_above_zero:
            qualifies &= macd_above
        if filter_nf:
            qualifies &= nf_ok
        out.append({
            'kode': kd,
            'last_cross': last_type,
            'last_cross_date': pd.to_datetime(last_date).date(),
            'days_ago': int(days_ago),
            'macd_above_zero': macd_above,
            f'NF_sum_{int(nf_window)}d': nf_sum,
            'qualifies': bool(qualifies),
            'close_last': float(close.iloc[-1])
        })
    return pd.DataFrame(out)

    wins = trades_df[trades_df["ret_pct"] > 0]
    losses = trades_df[trades_df["ret_pct"] <= 0]
    pf = (wins["ret_pct"].sum() / abs(losses["ret_pct"].sum())) if not losses.empty else np.inf
    winrate = (len(wins) / len(trades_df)) * 100.0

    if not eq_df.empty:
        ec = eq_df.set_index("date")["equity"]
        roll_max = ec.cummax()
        max_dd = ((ec / roll_max) - 1.0).min() * 100.0
        days = (ec.index.max().date() - ec.index.min().date()).days if len(ec) > 1 else 0
        cagr = ((ec.iloc[-1]) ** (365.0 / days) - 1.0) * 100.0 if days > 0 else np.nan
        total_ret = (ec.iloc[-1] - 1.0) * 100.0
    else:
        max_dd = np.nan
        cagr = np.nan
        total_ret = np.nan

    stats = {
        "trades": int(len(trades_df)),
        "winrate": float(winrate),
        "profit_factor": float(pf) if not np.isinf(pf) else np.inf,
        "max_dd_pct": float(max_dd),
        "cagr_pct": float(cagr),
        "total_return_pct": float(total_ret),
    }
    return trades_df, eq_df, stats

# ---------- UI (compact filter bar) ----------
codes = list_codes()
min_d, max_d = date_bounds()

# defaults
if "kode_saham" not in st.session_state:
    st.session_state["kode_saham"] = (codes[0] if codes else None)
if "adv_mode" not in st.session_state:
    st.session_state["adv_mode"] = "All (Default)"
if "date_range" not in st.session_state:
    st.session_state["date_range"] = (max(min_d, max_d - relativedelta(years=1)), max_d)
if "range_choice" not in st.session_state:
    st.session_state["range_choice"] = "1 Tahun"
if "hide_non_trading" not in st.session_state:
    st.session_state["hide_non_trading"] = True
if "show_price" not in st.session_state:
    st.session_state["show_price"] = True
if "show_spread" not in st.session_state:
    st.session_state["show_spread"] = True
# MACD defaults
if "macd_preset" not in st.session_state:
    st.session_state["macd_preset"] = "12-26-9 (Standard)"
if "macd_fast" not in st.session_state:
    st.session_state["macd_fast"] = 12
if "macd_slow" not in st.session_state:
    st.session_state["macd_slow"] = 26
if "macd_signal" not in st.session_state:
    st.session_state["macd_signal"] = 9

# === FILTER ROW (one line) ===
with st.container():
    st.markdown("<div class='sticky-filter'>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1.3, 1.0, 1.6])
    with f1:
        st.selectbox("Pilih saham", options=codes, key="kode_saham")
    with f2:
        st.selectbox("ADV window (hari)", ["All (Default)", "1 Tahun", "6 Bulan", "3 Bulan", "1 Bulan"], key="adv_mode")
    with f3:
        dr = st.date_input(
            "Rentang tanggal",
            value=st.session_state.get("date_range", (max(min_d, max_d - relativedelta(years=1)), max_d)),
            min_value=min_d,
            max_value=max_d,
        )
        # jika user edit manual → switch ke Custom
        if isinstance(dr, tuple) and dr != st.session_state.get("date_range"):
            st.session_state["date_range"] = dr
            st.session_state["range_choice"] = "Custom"
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

kode = st.session_state.get("kode_saham")
if not kode:
    st.stop()

# quick range radio (row 2)
with st.container():
    td_series = get_trade_dates(kode)
    td_min = td_series.min().date() if not td_series.empty else min_d
    td_max = td_series.max().date() if not td_series.empty else max_d

    st.radio("Range cepat", ["All", "1 Tahun", "6 Bulan", "3 Bulan", "1 Bulan", "5 Hari", "Custom"], key="range_choice", horizontal=True)

    def _apply_quick(choice: str):
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
            st.session_state["date_range"] = (start, end)
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    _apply_quick(st.session_state.get("range_choice", "1 Tahun"))

# feature toggles (row 3, compact, one line)
with st.container():
    st.markdown("<div class='checks-row'>", unsafe_allow_html=True)
    cA, cB, cC = st.columns([1,1,1])
    with cA:
        st.checkbox("Tampilkan harga (jika tersedia)", key="show_price")
    with cB:
        st.checkbox("Tampilkan spread (bps) jika tersedia", key="show_spread")
    with cC:
        st.checkbox("Hide non-trading days (skip weekend & libur bursa)", key="hide_non_trading")
    st.markdown("</div>", unsafe_allow_html=True)

# MACD preset row
with st.container():
    m1, m2, m3, m4 = st.columns([1.6, 0.8, 0.8, 0.8])
    with m1:
        st.selectbox(
            "MACD preset",
            ["12-26-9 (Standard)", "5-35-5 (Fast)", "8-17-9", "10-30-9", "20-50-9", "Custom"],
            key="macd_preset",
        )
    if st.session_state.get("macd_preset") == "Custom":
        with m2:
            st.number_input("Fast EMA", min_value=1, max_value=200, step=1, key="macd_fast")
        with m3:
            st.number_input("Slow EMA", min_value=2, max_value=400, step=1, key="macd_slow")
        with m4:
            st.number_input("Signal", min_value=1, max_value=100, step=1, key="macd_signal")

# ---------- Data ----------
start, end = st.session_state["date_range"]
adv_mode = st.session_state["adv_mode"]
show_price = st.session_state["show_price"]
show_spread = st.session_state["show_spread"]
hide_non_trading = st.session_state["hide_non_trading"]

df_raw = load_series(kode, start, end)
if df_raw.empty:
    st.warning("Data kosong untuk rentang ini.")
    st.stop()

df = add_rolling(ensure_metrics(df_raw), adv_mode=adv_mode)

# ---------- KPI ----------
st.divider()

nama = df["nama_perusahaan"].dropna().iloc[-1] if "nama_perusahaan" in df.columns and df["nama_perusahaan"].notna().any() else "-"
st.subheader(f"{kode} — {nama}")

price_series = None
if "penutupan" in df.columns and df["penutupan"].notna().any():
    price_series = pd.to_numeric(df["penutupan"], errors="coerce")
elif "close_price" in df.columns and df["close_price"].notna().any():
    price_series = pd.to_numeric(df["close_price"], errors="coerce")

total_buy = df["foreign_buy"].sum(skipna=True) if "foreign_buy" in df.columns else np.nan
total_sell = df["foreign_sell"].sum(skipna=True) if "foreign_sell" in df.columns else np.nan
total_nf   = df["net_foreign_value"].sum(skipna=True)

m1,m2,m3,m4,m5 = st.columns([1,1,1,1,1])
m1.metric("Total Net Foreign", idr_short(total_nf))
m2.metric("Total Foreign Buy", idr_short(total_buy) if pd.notna(total_buy) else "-")
m3.metric("Total Foreign Sell", idr_short(total_sell) if pd.notna(total_sell) else "-")

pos_days = (df["net_foreign_value"] > 0).sum()
pct_pos  = 100 * pos_days / len(df)
m4.metric("% Hari Net Buy", f"{pct_pos:.0f}%")

max_ratio = df["ratio"].max(skipna=True)
m5.metric("Max Ratio (nilai/ADV)", f"{max_ratio:.2f}x" if pd.notna(max_ratio) else "-")

if df["net_foreign_value"].notna().any():
    max_buy_day  = df.loc[df["net_foreign_value"].idxmax()]
    max_sell_day = df.loc[df["net_foreign_value"].idxmin()]
    st.caption(f"🔎 Hari Net Buy terbesar: **{max_buy_day['trade_date'].date()}** ({idr_short(max_buy_day['net_foreign_value'])})")
    st.caption(f"🔎 Hari Net Sell terbesar: **{max_sell_day['trade_date'].date()}** ({idr_short(max_sell_day['net_foreign_value'])})")

# ---------- Charts ----------
# Candlestick
if show_price:
    open_series = None
    if "open_price" in df.columns and df["open_price"].notna().any():
        open_series = pd.to_numeric(df["open_price"], errors="coerce")
    elif "first_trade" in df.columns and df["first_trade"].notna().any():
        open_series = pd.to_numeric(df["first_trade"], errors="coerce")
    elif price_series is not None:
        open_series = price_series.copy()

    high_series = pd.to_numeric(df.get("tertinggi"), errors="coerce") if "tertinggi" in df.columns else None
    low_series  = pd.to_numeric(df.get("terendah"),  errors="coerce") if "terendah"  in df.columns else None
    close_series = price_series

    has_ohlc = (open_series is not None and close_series is not None and high_series is not None and low_series is not None and open_series.notna().any() and close_series.notna().any() and high_series.notna().any() and low_series.notna().any())

    if has_ohlc:
        mask_ohlc = open_series.notna() & high_series.notna() & low_series.notna() & close_series.notna()
        df_cdl = df.loc[mask_ohlc].copy()
        open_c, high_c, low_c, close_c = open_series.loc[mask_ohlc], high_series.loc[mask_ohlc], low_series.loc[mask_ohlc], close_series.loc[mask_ohlc]
        if not df_cdl.empty:
            ma5  = close_c.rolling(5,  min_periods=1).mean()
            ma20 = close_c.rolling(20, min_periods=1).mean()
            figC = go.Figure()
            figC.add_trace(go.Candlestick(x=df_cdl["trade_date"], open=open_c, high=high_c, low=low_c, close=close_c, increasing_line_color="#33B766", decreasing_line_color="#DC3545", name="OHLC"))
            figC.add_trace(go.Scatter(x=df_cdl["trade_date"], y=ma5,  name="MA 5",  line=dict(color="#0d6efd", width=1.8)))
            figC.add_trace(go.Scatter(x=df_cdl["trade_date"], y=ma20, name="MA 20", line=dict(color="#6c757d", width=1.8)))
            figC.update_layout(
                title="Candlestick (OHLC) + MA5/MA20",
                xaxis_title=None,
                yaxis_title="Harga (IDR)",
                xaxis_rangeslider_visible=False,
                height=460,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # === MACD cross markers on candlestick ===
            try:
                # MACD params (follow preset/custom)
                fast, slow, sig = get_macd_params()
                close_m = close_series.dropna()
                trade_m = df.loc[close_series.notna(), "trade_date"].reset_index(drop=True)
                ema_fast = close_m.ewm(span=fast, adjust=False, min_periods=1).mean()
                ema_slow = close_m.ewm(span=slow, adjust=False, min_periods=1).mean()
                macd_line_c = ema_fast - ema_slow
                signal_line_c = macd_line_c.ewm(span=sig, adjust=False, min_periods=1).mean()
                delta_c = macd_line_c - signal_line_c
                prev_delta_c = delta_c.shift(1)
                bull_idx_c = (prev_delta_c <= 0) & (delta_c > 0)
                bear_idx_c = (prev_delta_c >= 0) & (delta_c < 0)

                # Align to candlestick subset (only rows with full OHLC)
                t_cdl = df_cdl["trade_date"].reset_index(drop=True)
                map_low  = pd.Series(low_c.values,  index=t_cdl)
                map_high = pd.Series(high_c.values, index=t_cdl)

                bull_dates = trade_m[bull_idx_c]
                bear_dates = trade_m[bear_idx_c]
                bull_dates = [d for d in bull_dates if d in map_low.index]
                bear_dates = [d for d in bear_dates if d in map_high.index]

                if len(bull_dates) > 0:
                    yb = [float(map_low[d]) * 0.995 for d in bull_dates]
                    figC.add_trace(go.Scatter(x=bull_dates, y=yb, mode="markers", name="Bullish cross (MACD)",
                                              marker=dict(symbol="triangle-up", size=12, color="#22c55e")))
                if len(bear_dates) > 0:
                    ya = [float(map_high[d]) * 1.005 for d in bear_dates]
                    figC.add_trace(go.Scatter(x=bear_dates, y=ya, mode="markers", name="Bearish cross (MACD)",
                                              marker=dict(symbol="triangle-down", size=12, color="#ef4444")))
            except Exception:
                pass

            _apply_time_axis(figC, df_cdl["trade_date"], start, end, hide_non_trading)
            st.plotly_chart(figC, use_container_width=True)
    else:
        st.info("Candlestick belum bisa ditampilkan karena kolom OHLC tidak lengkap pada rentang ini.")

# ---------- MACD ----------
if price_series is not None and price_series.notna().any():
    sub_m = df[price_series.notna()][["trade_date"]].copy()
    close_m = price_series.dropna()

    # MACD params
    fast, slow, sig = get_macd_params()
    ema_fast = close_m.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = close_m.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=sig, adjust=False, min_periods=1).mean()
    hist = macd_line - signal_line

    # Crossover detection
    delta = macd_line - signal_line
    prev_delta = delta.shift(1)
    bull_idx = (prev_delta <= 0) & (delta > 0)
    bear_idx = (prev_delta >= 0) & (delta < 0)

    x_bull = sub_m.loc[bull_idx, "trade_date"]
    y_bull = macd_line[bull_idx]
    x_bear = sub_m.loc[bear_idx, "trade_date"]
    y_bear = macd_line[bear_idx]

    figM = go.Figure()
    colorsM = np.where(hist >= 0, "rgba(51,183,102,0.7)", "rgba(220,53,69,0.7)")
    figM.add_bar(x=sub_m["trade_date"], y=hist, name="Histogram", marker_color=colorsM)

    figM.add_trace(go.Scatter(x=sub_m["trade_date"], y=macd_line,  name="MACD",   line=dict(width=2, color="#0d6efd")))
    figM.add_trace(go.Scatter(x=sub_m["trade_date"], y=signal_line, name="Signal", line=dict(width=2, color="#fd7e14")))

    # Arrow markers on crossovers
    figM.add_trace(go.Scatter(
        x=x_bull, y=y_bull, mode="markers", name="Bullish crossover",
        marker=dict(symbol="triangle-up", size=12, color="#22c55e"),
        hovertemplate="%{x|%Y-%m-%d}<br>MACD: %{y:.2f}<extra>Bullish crossover</extra>"
    ))
    figM.add_trace(go.Scatter(
        x=x_bear, y=y_bear, mode="markers", name="Bearish crossover",
        marker=dict(symbol="triangle-down", size=12, color="#ef4444"),
        hovertemplate="%{x|%Y-%m-%d}<br>MACD: %{y:.2f}<extra>Bearish crossover</extra>"
    ))

    figM.add_hline(y=0, line_color="#adb5bd", line_width=1)
    figM.update_layout(title=f"MACD ({fast},{slow},{sig}) + Crossovers", xaxis_title=None, yaxis_title="MACD",
                       height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    _apply_time_axis(figM, sub_m["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(figM, use_container_width=True)

    # Status ringkas & alert cross terbaru
    try:
        delta_val = float(delta.iloc[-1])
        status = "Bullish" if delta_val > 0 else "Bearish"
        # last cross detection
        last_cross_type = None
        last_cross_date = None
        if bull_idx.any():
            last_bull_date = sub_m.loc[bull_idx, "trade_date"].iloc[-1]
            last_cross_type, last_cross_date = "Bullish", last_bull_date
        if bear_idx.any():
            last_bear_date = sub_m.loc[bear_idx, "trade_date"].iloc[-1]
            if last_cross_date is None or pd.to_datetime(last_bear_date) > pd.to_datetime(last_cross_date):
                last_cross_type, last_cross_date = "Bearish", last_bear_date
        if last_cross_date is not None:
            end_dt = pd.to_datetime(df["trade_date"].max())
            days_ago = (end_dt.date() - pd.to_datetime(last_cross_date).date()).days
            st.caption(f"⚡️ Cross terbaru: **{last_cross_type}** pada **{pd.to_datetime(last_cross_date).date()}** · {days_ago} hari lalu.")
        st.caption(f"📉 MACD status saat ini: **{status}** (MACD - Signal = {delta_val:.2f}).")
    except Exception:
        pass

# Close vs Net Foreign
if show_price and price_series is not None and price_series.notna().any():
    fig1 = go.Figure()
    sub_nf = df[df["net_foreign_value"].notna()][["trade_date","net_foreign_value"]]
    colors = np.where(sub_nf["net_foreign_value"]>=0, "rgba(51,183,102,0.7)", "rgba(220,53,69,0.7)")
    fig1.add_bar(x=sub_nf["trade_date"], y=sub_nf["net_foreign_value"], name="Net Foreign (Rp)", marker_color=colors, yaxis="y2")
    sub_cl = df[price_series.notna()][["trade_date"]].assign(close=price_series.dropna())
    fig1.add_trace(go.Scatter(x=sub_cl["trade_date"], y=sub_cl["close"], name="Close", mode="lines+markers", line=dict(color=THEME, width=2.2)))
    fig1.update_layout(title="Close vs Net Foreign Harian", xaxis_title=None, yaxis=dict(title="Close"), yaxis2=dict(title="Net Foreign (Rp)", overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=420)
    _apply_time_axis(fig1, df["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(fig1, use_container_width=True)
else:
    sub_nf = df[df["net_foreign_value"].notna()]
    fig1b = px.bar(sub_nf, x="trade_date", y="net_foreign_value", title="Net Foreign Harian (Rp)", color=(sub_nf["net_foreign_value"]>=0).map({True:"Net Buy", False:"Net Sell"}), color_discrete_map={"Net Buy":"#33B766","Net Sell":"#DC3545"})
    fig1b.update_layout(height=360, xaxis_title=None, yaxis_title="Net Foreign (Rp)", legend_title=None)
    _apply_time_axis(fig1b, sub_nf["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(fig1b, use_container_width=True)

# Cum Net Foreign vs Close
if show_price and price_series is not None and price_series.notna().any():
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["trade_date"], y=df["cum_nf"], name="Cum. Net Foreign", line=dict(color="#6f42c1", width=2.2)))
    sub_cl = df[price_series.notna()][["trade_date"]].assign(close=price_series.dropna())
    fig2.add_trace(go.Scatter(x=sub_cl["trade_date"], y=sub_cl["close"], name="Close", yaxis="y2", line=dict(color=THEME, width=2), opacity=0.9))
    fig2.update_layout(title="Kumulatif Net Foreign vs Close", xaxis_title=None, yaxis=dict(title="Cum. Net Foreign"), yaxis2=dict(title="Close", overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=420)
    _apply_time_axis(fig2, df["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(fig2, use_container_width=True)
else:
    fig2b = px.line(df, x="trade_date", y="cum_nf", title="Kumulatif Net Foreign")
    fig2b.update_layout(height=360, xaxis_title=None, yaxis_title="Cum. Net Foreign")
    _apply_time_axis(fig2b, df["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(fig2b, use_container_width=True)

# Buy vs Sell stacked
has_buy_sell_cols = set(["foreign_buy","foreign_sell"]).issubset(df.columns)
if has_buy_sell_cols:
    sub = df[["trade_date","foreign_buy","foreign_sell"]].copy()
    sub["foreign_buy"]  = pd.to_numeric(sub["foreign_buy"], errors="coerce")
    sub["foreign_sell"] = pd.to_numeric(sub["foreign_sell"], errors="coerce")
    sub = sub[(sub["foreign_buy"].notna()) | (sub["foreign_sell"].notna())]
    if not sub.empty:
        df_bs = sub.melt(id_vars=["trade_date"], value_vars=["foreign_buy","foreign_sell"], var_name="jenis", value_name="nilai")
        fig3 = px.bar(df_bs, x="trade_date", y="nilai", color="jenis", title="Foreign Buy vs Sell (Rp)", color_discrete_map={"foreign_buy":"#0d6efd","foreign_sell":"#DC3545"})
        fig3.update_layout(barmode="stack", height=360, xaxis_title=None, yaxis_title="Rp", legend_title=None)
        _apply_time_axis(fig3, sub["trade_date"], start, end, hide_non_trading)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Data Foreign Buy/Sell tidak tersedia untuk rentang ini.")
else:
    st.info("Data Foreign Buy/Sell tidak tersedia untuk rentang ini.")

# Nilai vs ADV + Ratio
fig4 = go.Figure()
sub_nilai = df[df["nilai"].notna()][["trade_date","nilai"]]
fig4.add_bar(x=sub_nilai["trade_date"], y=sub_nilai["nilai"], name="Nilai (Rp)")
sub_adv = df[df["adv"].notna()][["trade_date","adv"]]
fig4.add_trace(go.Scatter(x=sub_adv["trade_date"], y=sub_adv["adv"],   name=f"ADV ({df.attrs.get('adv_label','ADV')})"))
sub_ratio = df[df["ratio"].notna()][["trade_date","ratio"]]
fig4.add_trace(go.Scatter(x=sub_ratio["trade_date"], y=sub_ratio["ratio"], name="Ratio (Nilai/ADV)", yaxis="y2"))
fig4.update_layout(title=f"Nilai vs ADV ({df.attrs.get('adv_label','ADV')}) + Ratio", xaxis_title=None, yaxis=dict(title="Rp"), yaxis2=dict(title="Ratio (x)", overlaying="y", side="right"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=420)
_apply_time_axis(fig4, df["trade_date"], start, end, hide_non_trading)
st.plotly_chart(fig4, use_container_width=True)

# Volume vs AVG
if "volume" in df.columns:
    fig5 = go.Figure()
    sub_v  = df[df["volume"].notna()][["trade_date","volume"]]
    fig5.add_bar(x=sub_v["trade_date"], y=sub_v["volume"], name="Volume")
    sub_va = df[df["vol_avg"].notna()][["trade_date","vol_avg"]]
    fig5.add_trace(go.Scatter(x=sub_va["trade_date"], y=sub_va["vol_avg"], name="Vol AVG(20)"))
    fig5.update_layout(title="Volume vs AVG(20)", xaxis_title=None, yaxis_title="Lembar", height=360, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    _apply_time_axis(fig5, df["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(fig5, use_container_width=True)

# Spread (bps)
if show_spread and "spread_bps" in df.columns:
    sub_sp = df[df["spread_bps"].notna()][["trade_date","spread_bps"]]
    if not sub_sp.empty:
        sp_vals = sub_sp["spread_bps"].dropna()
        p75 = float(np.nanpercentile(sp_vals, 75)) if len(sp_vals) else None
        fig6 = px.line(sub_sp, x="trade_date", y="spread_bps", title="Spread (bps)")
        if p75 is not None:
            fig6.add_hline(y=p75, line_dash="dot", line_color="#dc3545", annotation_text=f"P75 ≈ {p75:.1f}")
        fig6.update_layout(height=320, xaxis_title=None, yaxis_title="bps")
        _apply_time_axis(fig6, sub_sp["trade_date"], start, end, hide_non_trading)
        st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ---------- Tabel & Export ----------
with st.expander("Tabel data mentah (siap export)"):
    cols_show = [
        "trade_date","kode_saham","nama_perusahaan",
        "sebelumnya","open_price","first_trade","tertinggi","terendah","penutupan","selisih",
        "nilai","adv","ratio","volume","vol_avg",
        "foreign_buy","foreign_sell","net_foreign_value","spread_bps","close_price","cum_nf",
    ]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show], use_container_width=True, height=360)
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{kode}_foreign_flow_{start}_to_{end}.csv",
        mime="text/csv"
    )

st.caption("💡 Filter utama ada di satu baris (Saham, ADV, Rentang Tanggal) + toggle di kanan. Quick range All/1Y/6M/3M/1M/5D tersedia di bawahnya. Semua chart skip tanggal non-trading saat toggle diaktifkan.")

# ============================
# 🔎 Scanner — MACD Cross + Net Foreign
# ============================
st.divider()
with st.expander("🔎 Scanner — MACD Cross + Net Foreign", expanded=False):
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        scan_all = st.checkbox("Scan semua kode", value=True)
    with c2:
        recency = st.number_input("Cross dalam X hari terakhir", min_value=1, max_value=365, value=15, step=1)
    with c3:
        require_above_zero = st.checkbox("Syarat MACD > 0", value=False)

    d1, d2, d3 = st.columns([1,1,1])
    with d1:
        require_nf = st.checkbox("Syarat NF rolling ≥ 0", value=True)
    with d2:
        nf_window = st.number_input("Window NF (hari)", min_value=1, max_value=30, value=5, step=1)
    with d3:
        preset_txt = ",".join(map(str, get_macd_params()))
        st.caption(f"Preset MACD aktif: **{preset_txt}**")

    watchlist = codes if scan_all else st.multiselect("Watchlist", options=codes, default=[kode] if kode else [])

    # Tombol trigger agar tidak auto-scan & memberi kontrol user
    run_scan = st.button("🚀 Jalankan Scan", type="primary")

    if run_scan:
        with st.spinner("Mengambil data & scanning cepat..."):
            fast, slow, sig = get_macd_params()
            # Warmup minimal agar EMA stabil + buffer dari recency & NF window
            warmup_days = max(slow * 5, 150)
            approx_days = int(warmup_days + recency + int(nf_window) + 30)
            start_scan = max(min_d, end - timedelta(days=approx_days))

            df_bulk = fetch_ohlc_bulk(watchlist, start_scan, end)
            df_scan = scan_universe_fast(
                df_bulk, fast, slow, sig,
                nf_window=int(nf_window), filter_nf=bool(require_nf),
                only_recent_days=int(recency), require_above_zero=bool(require_above_zero)
            )
        if df_scan is None or df_scan.empty:
            st.info("Tidak ada hasil yang memenuhi filter.")
        else:
            order_cols = [
                "qualifies", "days_ago", "last_cross_date", "kode", "last_cross",
                "macd_above_zero", f"NF_sum_{int(nf_window)}d", "close_last"
            ]
            show_cols = [c for c in order_cols if c in df_scan.columns]
            st.dataframe(
                df_scan.sort_values(["qualifies", "days_ago", "last_cross_date"], ascending=[False, True, False])[show_cols],
                use_container_width=True, height=420
            )
            st.download_button(
                "⬇️ Download hasil scan (CSV)",
                data=df_scan.to_csv(index=False).encode("utf-8"),
                file_name=f"scan_macd_{start_scan}_to_{end}.csv",
                mime=\"text/csv\",
            )

            # ===== Ranking + Quick Action (tambahan) =====
            try:
                nf_col = f"NF_sum_{int(nf_window)}d"
                dfv = df_scan.copy()
                nf_vals = pd.to_numeric(dfv.get(nf_col, pd.Series(np.nan)), errors='coerce')
                if nf_vals.notna().any() and (nf_vals.max() - nf_vals.min()) > 0:
                    nf_norm = (nf_vals - nf_vals.min()) / (nf_vals.max() - nf_vals.min())
                else:
                    nf_norm = pd.Series(0.5, index=dfv.index)
                recency = np.exp(-0.35 * pd.to_numeric(dfv['days_ago'], errors='coerce').fillna(999))
                macd_up = dfv.get('macd_above_zero', False).astype(bool).astype(int)
                bull = (dfv.get('last_cross', '') == 'Bullish').astype(int)
                dfv['score'] = (0.4*nf_norm + 0.3*recency + 0.2*macd_up + 0.1*bull).round(3)
                st.caption("
**Ranking kandidat (score = 40% NF + 30% recency + 20% MACD>0 + 10% bullish)**
")
                only_q = st.checkbox("Tampilkan hanya yang qualifies", value=True, key='only_q_rank')
                dfshow = dfv[dfv['qualifies']] if (only_q and 'qualifies' in dfv.columns) else dfv
                order_cols = ['score','qualifies','days_ago','last_cross_date','kode','last_cross','macd_above_zero', nf_col, 'close_last']
                show_cols = [c for c in order_cols if c in dfshow.columns]
                df_ranked = dfshow.sort_values(['qualifies','score','days_ago'], ascending=[False, False, True])[show_cols]
                st.dataframe(df_ranked, use_container_width=True, height=420)
                if not df_ranked.empty:
                    cc1, cc2 = st.columns([3,1])
                    with cc1:
                        pick_code = st.selectbox("Pilih kode untuk dibuka di chart", options=df_ranked['kode'].tolist(), key='pick_code_rank')
                    with cc2:
                        if st.button("Set ke chart", use_container_width=True, key='btn_set_chart'):
                            st.session_state['kode_saham'] = pick_code
                            st.rerun()
            except Exception:
                pass
    else:
        st.caption("Klik **Jalankan Scan** untuk mulai. Scanner menggunakan bulk query + vectorized MACD agar jauh lebih cepat.")

# ============================
# 🧪 Backtest — MACD Rules (simple)
# ============================
st.divider()
with st.expander("🧪 Backtest — MACD Rules (simple)", expanded=False):
    b1, b2, b3 = st.columns([1,1,1])
    with b1:
        idx_default = (codes.index(kode) if (codes and kode in codes) else 0)
        bt_symbol = st.selectbox("Kode saham", options=codes, index=idx_default if idx_default < len(codes) else 0)
    with b2:
        bt_fee = st.number_input("Biaya/Slippage (bps per sisi)", min_value=0, max_value=100, value=0, step=1)
    with b3:
        bt_require_above = st.checkbox("Entry hanya jika MACD > 0", value=False)

    e1, e2 = st.columns([1,1])
    with e1:
        bt_require_nf = st.checkbox("Entry hanya jika NF rolling ≥ 0", value=False)
    with e2:
        bt_nf_window = st.number_input("NF window (hari)", min_value=1, max_value=30, value=5, step=1)

    df_bt = load_series(bt_symbol, start, end)
    if df_bt.empty:
        st.info("Data kosong untuk backtest.")
    else:
        with st.spinner("Menjalankan backtest..."):
            try:
                fast, slow, sig = get_macd_params()
                trades, eq_df, stats = backtest_macd(
                    df_bt, fast, slow, sig,
                    require_above_zero=bool(bt_require_above),
                    nf_window=int(bt_nf_window), require_nf=bool(bt_require_nf),
                    fee_bp=int(bt_fee)
                )
            except Exception as e:
                st.error("Backtest error: " + str(e))
                trades, eq_df, stats = pd.DataFrame(), pd.DataFrame(), {"trades":0,"winrate":np.nan,"profit_factor":np.nan,"max_dd_pct":np.nan,"cagr_pct":np.nan,"total_return_pct":np.nan}

        # --- Metrics ---
        try:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Trades", int(stats.get("trades", 0)))
            winrate_val = stats.get('winrate', np.nan)
            m2.metric("Win Rate", f"{float(winrate_val):.1f}%" if winrate_val==winrate_val else "-")
            pf_val = stats.get("profit_factor", np.nan)
            pf_str = "∞" if (isinstance(pf_val, (float, int, np.floating)) and not math.isfinite(float(pf_val))) else (f"{float(pf_val):.2f}" if pf_val==pf_val else "-")
            m3.metric("Profit Factor", pf_str)
            dd_val = stats.get('max_dd_pct', np.nan)
            m4.metric("Max DD", f"{float(dd_val):.1f}%" if dd_val==dd_val else "-")
            tr_val = stats.get('total_return_pct', np.nan)
            m5.metric("Total Return", f"{float(tr_val):.1f}%" if tr_val==tr_val else "-")
        except Exception as e:
            st.caption("⚠️ Gagal menampilkan ringkasan metrics: " + str(e))

        if not eq_df.empty:
            figEQ = px.line(eq_df, x="date", y="equity", title="Equity Curve (1.0 = awal)")
            _apply_time_axis(figEQ, pd.to_datetime(eq_df["date"]), start, end, hide_non_trading)
            st.plotly_chart(figEQ, use_container_width=True)

        if not trades.empty:
            st.dataframe(trades, use_container_width=True, height=320)
            st.download_button(
                "⬇️ Download trades CSV",
                data=trades.to_csv(index=False).encode("utf-8"),
                file_name=f"trades_{bt_symbol}_{start}_to_{end}.csv",
                mime="text/csv"
            )
        else:
            st.info("Tidak ada trade pada parameter/rule ini.")
