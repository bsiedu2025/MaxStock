# app/pages/13_Foreign_Flow_Detail.py
# Daily Stock Movement – Foreign Flow Focus (HP-friendly + Scanner persist + Backtest polished)
# - 📱 Mobile Mode toggle
# - Scanner: chip filter cepat (≤3D, NF≥p75, MACD>0), Export, "Kirim ke Backtest" (persist hasil)
# - Backtest: UI dirapikan, metrik lengkap; SEMUA perhitungan metrik ADA DI DALAM backtest_macd(...)
# - Python 3.10 friendly, guard kolom fleksibel

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import math
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from db_utils import get_db_connection, get_db_name

THEME = "#3AA6A0"
st.set_page_config(page_title="📈 Pergerakan Harian (Foreign Flow)", page_icon="📈", layout="wide")

# --- Styling & sticky filter bar ---
st.markdown(
    """
    <style>
    section.main > div { padding-top: .5rem; }
    /* radio as segmented pills */
    div[role='radiogroup'] { display: flex; flex-wrap: wrap; gap: .25rem .5rem; }
    div[role='radiogroup'] label { border:1px solid #e5e7eb; padding:6px 12px; border-radius:9999px; background:#fff; }
    .stSelectbox > label, .stDateInput > label, .stRadio > label { font-weight: 600; }
    .checks-row { display:flex; align-items:center; gap: 1.25rem; }
    .checks-row div[data-testid='stCheckbox'] { margin-bottom: 0 !important; }
    .sticky-filter { position: sticky; top: 0; z-index: 1000; background: rgba(255,255,255,.96);
        backdrop-filter: blur(6px); border-bottom: 1px solid #e5e7eb; padding: .5rem .25rem .75rem .25rem;
        box-shadow: 0 8px 16px rgba(0,0,0,.06), 0 1px 0 rgba(0,0,0,.04);
    }
    .chip label { border:1px solid #e5e7eb; padding:6px 10px; border-radius:9999px; background:#fff; }
    /* metric subtle titles */
    .metric-note { color:#64748b; font-size:12px; margin-top:-6px; }
    @media (max-width: 480px) {
      .sticky-filter { padding: .35rem .15rem .5rem .15rem; }
      div[role='radiogroup'] label { padding:6px 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Pergerakan Harian Saham — Fokus Foreign Flow")
st.caption(f"DB aktif: **{get_db_name()}**")

# ---------- DB helpers ----------
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
        return pd.to_datetime(df["trade_date"])
    finally:
        _close(con)

@st.cache_data(ttl=300, show_spinner=False)
def load_series(kode: str, start: date, end: date) -> pd.DataFrame:
    con = get_db_connection(); _alive(con)
    try:
        cols_df = pd.read_sql(
            "SELECT LOWER(column_name) col FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'data_harian'",
            con,
        )
        avail = set(cols_df["col"].tolist())

        base_cols = ["trade_date", "kode_saham"]
        if "nama_perusahaan" in avail:
            base_cols.append("nama_perusahaan")

        price_cols = [
            "sebelumnya","open_price","first_trade","tertinggi","terendah","penutupan","selisih",
        ]
        optional = [
            "nilai","volume","freq","foreign_buy","foreign_sell","net_foreign_value",
            "bid","offer","spread_bps","close_price",
        ]
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

# === rangebreaks ===
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
            sp[(b <= 0) | (o <= 0) | (b.isna()) | (o.isna())] = np.nan
            df["spread_bps"] = sp
        else:
            df["spread_bps"] = np.nan

    for c in [
        "nilai","volume","foreign_buy","foreign_sell","net_foreign_value","close_price",
        "penutupan","sebelumnya","tertinggi","terendah","selisih","open_price","first_trade",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df

# rolling & derived
def add_rolling(df: pd.DataFrame, adv_mode: str) -> pd.DataFrame:
    df = df.sort_values("trade_date").copy()
    nilai = pd.to_numeric(df.get("nilai"), errors="coerce")
    mode_to_win = {"1 Bulan": 20, "3 Bulan": 60, "6 Bulan": 120, "1 Tahun": 252}
    if adv_mode == "All (Default)":
        df["adv"] = nilai.expanding().mean(); adv_label = "All"
    else:
        win = mode_to_win.get(adv_mode, 20)
        df["adv"] = nilai.rolling(win, min_periods=1).mean(); adv_label = adv_mode
    df["vol_avg"] = pd.to_numeric(df.get("volume"), errors="coerce").rolling(20, min_periods=1).mean()
    df["ratio"] = df["nilai"] / df["adv"]
    df["cum_nf"] = df["net_foreign_value"].fillna(0).cumsum()
    df.attrs["adv_label"] = adv_label
    return df

# utils
def idr_short(x: float) -> str:
    try: n = float(x)
    except Exception: return "-"
    a = abs(n)
    for nama, v in [("Triliun", 1e12), ("Miliar", 1e9), ("Juta", 1e6), ("Ribu", 1e3)]:
        if a >= v:
            s = f"{n / v:,.2f}".replace(",", "."); return f"{s} {nama}"
    return f"{n:,.0f}".replace(",", ".")

def fmt_pct(v): return "-" if v is None or pd.isna(v) else f"{v:,.2f}%".replace(",", ".")
def format_money(v, dec=0):
    try: return f"{float(v):,.{dec}f}".replace(",", ".")
    except Exception: return "-"

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
    f = int(st.session_state.get("macd_fast", 12))
    s = int(st.session_state.get("macd_slow", 26))
    g = int(st.session_state.get("macd_signal", 9))
    if f < 1: f = 1
    if s <= f: s = f + 1
    if g < 1: g = 1
    return f, s, g

# --- MACD core ---
def macd_series(close: pd.Series, fast: int, slow: int, sig: int):
    close = pd.to_numeric(close, errors="coerce").dropna()
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig, adjust=False, min_periods=1).mean()
    delta = macd - signal
    return macd, signal, delta

def macd_cross_flags(delta: pd.Series):
    prev = delta.shift(1)
    bull = (prev <= 0) & (delta > 0)
    bear = (prev >= 0) & (delta < 0)
    return bull, bear

# -------- Bulk fetch (scanner) --------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_ohlc_bulk(codes, start, end, chunk_size: int = 200):
    if not codes:
        return pd.DataFrame()
    con = get_db_connection(); _alive(con)
    try:
        cols = pd.read_sql(
            "SELECT LOWER(column_name) AS col FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'data_harian'",
            con,
        )
        have = set(cols["col"].tolist())
        price_candidates = ["penutupan", "close_price", "closing", "close"]
        price_col = next((c for c in price_candidates if c in have), None)
        if price_col is None:
            raise RuntimeError(
                "Tidak ada kolom harga penutupan (butuh salah satu dari penutupan/close_price/closing/close)"
            )
        if "net_foreign_value" in have:
            nf_expr = "net_foreign_value"
        elif {"foreign_buy", "foreign_sell"}.issubset(have):
            nf_expr = "(foreign_buy - foreign_sell)"
        else:
            nf_expr = "NULL"

        parts = []
        start_param = pd.to_datetime(start); end_param = pd.to_datetime(end)
        for i in range(0, len(codes), max(1, int(chunk_size))):
            chunk = codes[i : i + chunk_size]
            if not chunk: continue
            placeholders = ",".join(["%s"] * len(chunk))
            sql = f"""
                SELECT kode_saham, trade_date, {price_col} AS close, {nf_expr} AS net_foreign_value
                FROM data_harian
                WHERE trade_date BETWEEN %s AND %s AND kode_saham IN ({placeholders})
                ORDER BY kode_saham, trade_date
            """
            params = [start_param, end_param] + list(chunk)
            part = pd.read_sql(sql, con, params=params)
            parts.append(part)
        if not parts: return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        if df.empty: return df
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        if "net_foreign_value" in df.columns:
            df["net_foreign_value"] = pd.to_numeric(df["net_foreign_value"], errors="coerce")
        df = df.dropna(subset=["close"])
        return df
    finally:
        _close(con)

def scan_universe_fast(
    df_bulk: pd.DataFrame,
    fast: int,
    slow: int,
    sig: int,
    nf_window: int = 5,
    filter_nf: bool = True,
    only_recent_days: int | None = 15,
    require_above_zero: bool = False,
) -> pd.DataFrame:
    if df_bulk is None or df_bulk.empty:
        return pd.DataFrame()
    out = []
    global_end = pd.to_datetime(df_bulk["trade_date"].max())
    for kd, g in df_bulk.groupby("kode_saham"):
        g = g.sort_values("trade_date").copy()
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
        nf_sum = np.nan
        nf_ok = True
        if filter_nf and "net_foreign_value" in g.columns:
            nf = g["net_foreign_value"].fillna(0.0)
            nf_sum = float(nf.tail(int(nf_window)).sum())
            nf_ok = nf_sum >= 0
        macd_above = bool(macd.iloc[-1] > 0)
        qualifies = True
        if only_recent_days is not None:
            qualifies &= days_ago <= int(only_recent_days)
        if require_above_zero:
            qualifies &= macd_above
        if filter_nf:
            qualifies &= nf_ok
        out.append(
            {
                "kode": kd,
                "last_cross": last_type,
                "last_cross_date": pd.to_datetime(last_date).date(),
                "days_ago": int(days_ago),
                "macd_above_zero": macd_above,
                f"NF_sum_{int(nf_window)}d": nf_sum,
                "qualifies": bool(qualifies),
                "close_last": float(close.iloc[-1]),
            }
        )
    return pd.DataFrame(out)

# -------- Backtest (SEMUA METRIK dihitung di sini) --------
def backtest_macd(
    df_price: pd.DataFrame,
    fast: int,
    slow: int,
    sig: int,
    require_above_zero: bool = False,
    nf_window: int = 0,
    require_nf: bool = False,
    fee_bp: int = 0,
):
    # pilih close
    close = None
    if "penutupan" in df_price.columns and df_price["penutupan"].notna().any():
        close = pd.to_numeric(df_price["penutupan"], errors="coerce")
    elif "close_price" in df_price.columns and df_price["close_price"].notna().any():
        close = pd.to_numeric(df_price["close_price"], errors="coerce")
    if close is None or close.dropna().empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "trades": 0, "winrate": np.nan, "profit_factor": np.nan,
            "max_dd_pct": np.nan, "cagr_pct": np.nan, "total_return_pct": np.nan,
            "avg_trade_pct": np.nan, "median_trade_pct": np.nan, "best_trade_pct": np.nan, "worst_trade_pct": np.nan,
            "avg_hold_days": np.nan, "sharpe": np.nan, "vol_annual_pct": np.nan, "calmar_ratio": np.nan,
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

        if (not in_pos) and bool(bull.iloc[i]) and nf_ok and cond_above:
            in_pos = True; entry_price = float(close_v.iloc[i]); entry_date = d
        elif in_pos and bool(bear.iloc[i]):
            exit_price = float(close_v.iloc[i]); exit_date = d
            ret = (exit_price / entry_price - 1.0) - 2 * fee
            equity *= (1.0 + ret)
            hold_days = int((exit_date.date() - entry_date.date()).days)
            trades.append(
                {
                    "entry_date": entry_date.date(), "entry_price": entry_price,
                    "exit_date": exit_date.date(), "exit_price": exit_price,
                    "ret_pct": ret * 100.0, "hold_days": hold_days,
                }
            )
            in_pos = False; entry_price = None; entry_date = None
        equity_curve.append({"date": d, "equity": equity})

    # close open trade at end
    if in_pos:
        exit_price = float(close_v.iloc[-1]); exit_date = dates.iloc[-1]
        ret = (exit_price / entry_price - 1.0) - 2 * fee
        equity *= (1.0 + ret)
        hold_days = int((exit_date.date() - entry_date.date()).days)
        trades.append(
            {
                "entry_date": entry_date.date(), "entry_price": entry_price,
                "exit_date": exit_date.date(), "exit_price": exit_price,
                "ret_pct": ret * 100.0, "hold_days": hold_days,
            }
        )
        if equity_curve:
            equity_curve[-1]["equity"] = equity

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve)

    # jika tidak ada trade, tetap kembalikan metrik lengkap (NaN)
    if trades_df.empty:
        stats = {
            "trades": 0, "winrate": np.nan, "profit_factor": np.nan,
            "max_dd_pct": np.nan, "cagr_pct": np.nan,
            "total_return_pct": (eq_df["equity"].iloc[-1] - 1.0) * 100.0 if not eq_df.empty else np.nan,
            "avg_trade_pct": np.nan, "median_trade_pct": np.nan, "best_trade_pct": np.nan, "worst_trade_pct": np.nan,
            "avg_hold_days": np.nan, "sharpe": np.nan, "vol_annual_pct": np.nan, "calmar_ratio": np.nan,
        }
        return trades_df, eq_df, stats

    # ---- Metrics (inside function) ----
    wins = trades_df[trades_df["ret_pct"] > 0]
    losses = trades_df[trades_df["ret_pct"] <= 0]
    pf = (wins["ret_pct"].sum() / abs(losses["ret_pct"].sum())) if not losses.empty else float("inf")
    winrate = (len(wins) / len(trades_df)) * 100.0

    # equity stats
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

    # trade stats
    avg_trade = trades_df["ret_pct"].mean()
    med_trade = trades_df["ret_pct"].median()
    best_trade = trades_df["ret_pct"].max()
    worst_trade = trades_df["ret_pct"].min()
    avg_hold = trades_df["hold_days"].mean() if "hold_days" in trades_df.columns else np.nan
    calmar = (cagr / abs(max_dd)) if (not pd.isna(cagr) and not pd.isna(max_dd) and max_dd != 0) else np.nan

    stats = {
        "trades": int(len(trades_df)),
        "winrate": float(winrate),
        "profit_factor": float(pf) if math.isfinite(pf) else float("inf"),
        "max_dd_pct": float(max_dd),
        "cagr_pct": float(cagr),
        "total_return_pct": float(total_ret),
        "avg_trade_pct": float(avg_trade),
        "median_trade_pct": float(med_trade),
        "best_trade_pct": float(best_trade),
        "worst_trade_pct": float(worst_trade),
        "avg_hold_days": float(avg_hold) if not pd.isna(avg_hold) else np.nan,
        "sharpe": float(sharpe) if not pd.isna(sharpe) else np.nan,
        "vol_annual_pct": float(vol_ann) if not pd.isna(vol_ann) else np.nan,
        "calmar_ratio": float(calmar) if not pd.isna(calmar) else np.nan,
    }
    return trades_df, eq_df, stats

# ---------- STATE DEFAULTS ----------
codes = list_codes()
min_d, max_d = date_bounds()

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
if "macd_preset" not in st.session_state:
    st.session_state["macd_preset"] = "12-26-9 (Standard)"
if "macd_fast" not in st.session_state:
    st.session_state["macd_fast"] = 12
if "macd_slow" not in st.session_state:
    st.session_state["macd_slow"] = 26
if "macd_signal" not in st.session_state:
    st.session_state["macd_signal"] = 9
if "mobile_mode" not in st.session_state:
    st.session_state["mobile_mode"] = False
if "bt_symbol" not in st.session_state:
    st.session_state["bt_symbol"] = st.session_state.get("kode_saham")
if "open_backtest" not in st.session_state:
    st.session_state["open_backtest"] = False

# ---------- FILTER BAR + Mobile toggle ----------
with st.container():
    st.markdown("<div class='sticky-filter'>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([1.3, 1.0, 1.6, 0.9])
    with f1:
        st.selectbox("Pilih saham", options=codes, key="kode_saham")
    with f2:
        st.selectbox("ADV window (hari)", ["All (Default)", "1 Tahun", "6 Bulan", "3 Bulan", "1 Bulan"], key="adv_mode")
    with f3:
        dr = st.date_input(
            "Rentang tanggal",
            value=st.session_state.get("date_range", (max(min_d, max_d - relativedelta(years=1)), max_d)),
            min_value=min_d, max_value=max_d,
        )
        if isinstance(dr, tuple) and dr != st.session_state.get("date_range"):
            st.session_state["date_range"] = dr
            st.session_state["range_choice"] = "Custom"
            try: st.rerun()
            except Exception: st.experimental_rerun()
    with f4:
        st.checkbox("📱 Mobile Mode", key="mobile_mode",
                    help="Mode ringkas untuk layar HP (chart lebih pendek, kontrol ditumpuk).")
    st.markdown("</div>", unsafe_allow_html=True)

MOBILE = bool(st.session_state.get("mobile_mode", False))
# tinggi dinamis
H_CANDLE   = 360 if MOBILE else 460
H_MACD     = 260 if MOBILE else 320
H_CLOSE_NF = 340 if MOBILE else 420
H_CUM      = 340 if MOBILE else 420
H_NILAI    = 340 if MOBILE else 420
H_VOLUME   = 300 if MOBILE else 360
H_SPREAD   = 260 if MOBILE else 320
DF_HEIGHT  = 280 if MOBILE else 360
SCAN_H     = 360 if MOBILE else 420
TRADES_H   = 280 if MOBILE else 320

# ---------- Quick range ----------
kode = st.session_state.get("kode_saham")
if not kode: st.stop()

td_series = get_trade_dates(kode)
td_min = td_series.min().date() if not td_series.empty else min_d
td_max = td_series.max().date() if not td_series.empty else max_d

st.radio(
    "Range cepat",
    ["All", "1 Tahun", "6 Bulan", "3 Bulan", "1 Bulan", "5 Hari", "Custom"],
    key="range_choice",
    horizontal=not MOBILE,
)

def _apply_quick(choice: str):
    if choice == "Custom": return
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
        try: st.rerun()
        except Exception: st.experimental_rerun()

_apply_quick(st.session_state.get("range_choice", "1 Tahun"))

# ---------- Toggles ----------
with st.container():
    st.markdown("<div class='checks-row'>", unsafe_allow_html=True)
    cA, cB, cC = st.columns([1, 1, 1])
    with cA: st.checkbox("Tampilkan harga (jika tersedia)", key="show_price")
    with cB: st.checkbox("Tampilkan spread (bps) jika tersedia", key="show_spread")
    with cC: st.checkbox("Hide non-trading days (skip weekend & libur bursa)", key="hide_non_trading")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- MACD preset ----------
m1, m2, m3, m4 = st.columns([1.6, 0.8, 0.8, 0.8])
with m1:
    st.selectbox(
        "MACD preset",
        ["12-26-9 (Standard)", "5-35-5 (Fast)", "8-17-9", "10-30-9", "20-50-9", "Custom"],
        key="macd_preset",
    )
if st.session_state.get("macd_preset") == "Custom":
    with m2: st.number_input("Fast EMA", min_value=1, max_value=200, step=1, key="macd_fast")
    with m3: st.number_input("Slow EMA", min_value=2, max_value=400, step=1, key="macd_slow")
    with m4: st.number_input("Signal",  min_value=1, max_value=100, step=1, key="macd_signal")

# ---------- Data ----------
start, end = st.session_state["date_range"]
adv_mode = st.session_state["adv_mode"]
show_price = st.session_state["show_price"]
show_spread = st.session_state["show_spread"]
hide_non_trading = st.session_state["hide_non_trading"]

df_raw = load_series(kode, start, end)
if df_raw.empty:
    st.warning("Data kosong untuk rentang ini."); st.stop()
df = add_rolling(ensure_metrics(df_raw), adv_mode=adv_mode)

# ---------- KPI ----------
st.divider()
nama = (df["nama_perusahaan"].dropna().iloc[-1]
        if "nama_perusahaan" in df.columns and df["nama_perusahaan"].notna().any() else "-")
st.subheader(f"{kode} — {nama}")

price_series = None
if "penutupan" in df.columns and df["penutupan"].notna().any():
    price_series = pd.to_numeric(df["penutupan"], errors="coerce")
elif "close_price" in df.columns and df["close_price"].notna().any():
    price_series = pd.to_numeric(df["close_price"], errors="coerce")

total_buy = df["foreign_buy"].sum(skipna=True) if "foreign_buy" in df.columns else np.nan
total_sell = df["foreign_sell"].sum(skipna=True) if "foreign_sell" in df.columns else np.nan
total_nf = df["net_foreign_value"].sum(skipna=True)

if MOBILE:
    a, b = st.columns(2)
    with a:
        st.metric("Total Net Foreign", idr_short(total_nf))
        st.metric("Total Foreign Buy", idr_short(total_buy) if pd.notna(total_buy) else "-")
    with b:
        pos_days = (df["net_foreign_value"] > 0).sum()
        pct_pos = 100 * pos_days / len(df)
        st.metric("% Hari Net Buy", f"{pct_pos:.0f}%")
        st.metric("Total Foreign Sell", idr_short(total_sell) if pd.notna(total_sell) else "-")
    st.metric("Max Ratio (nilai/ADV)", f"{df['ratio'].max(skipna=True):.2f}x" if df["ratio"].notna().any() else "-")
else:
    m1, m2, m3, m4, m5 = st.columns([1, 1, 1, 1, 1])
    m1.metric("Total Net Foreign", idr_short(total_nf))
    m2.metric("Total Foreign Buy", idr_short(total_buy) if pd.notna(total_buy) else "-")
    m3.metric("Total Foreign Sell", idr_short(total_sell) if pd.notna(total_sell) else "-")
    pos_days = (df["net_foreign_value"] > 0).sum()
    pct_pos = 100 * pos_days / len(df)
    m4.metric("% Hari Net Buy", f"{pct_pos:.0f}%")
    m5.metric("Max Ratio (nilai/ADV)", f"{df['ratio'].max(skipna=True):.2f}x" if df["ratio"].notna().any() else "-")

if df["net_foreign_value"].notna().any():
    max_buy_day = df.loc[df["net_foreign_value"].idxmax()]
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

    has_ohlc = (
        open_series is not None and close_series is not None and
        high_series is not None and low_series is not None and
        open_series.notna().any() and close_series.notna().any() and
        high_series.notna().any() and low_series.notna().any()
    )

    if has_ohlc:
        mask_ohlc = (open_series.notna() & high_series.notna() & low_series.notna() & close_series.notna())
        df_cdl = df.loc[mask_ohlc].copy()
        open_c, high_c, low_c, close_c = (
            open_series.loc[mask_ohlc], high_series.loc[mask_ohlc],
            low_series.loc[mask_ohlc], close_series.loc[mask_ohlc],
        )
        if not df_cdl.empty:
            ma5  = close_c.rolling(5,  min_periods=1).mean()
            ma20 = close_c.rolling(20, min_periods=1).mean()
            figC = go.Figure()
            figC.add_trace(go.Candlestick(
                x=df_cdl["trade_date"], open=open_c, high=high_c, low=low_c, close=close_c,
                increasing_line_color="#33B766", decreasing_line_color="#DC3545", name="OHLC",
            ))
            figC.add_trace(go.Scatter(x=df_cdl["trade_date"], y=ma5,  name="MA 5",  line=dict(color="#0d6efd", width=1.8)))
            figC.add_trace(go.Scatter(x=df_cdl["trade_date"], y=ma20, name="MA 20", line=dict(color="#6c757d", width=1.8)))
            figC.update_layout(
                title="Candlestick (OHLC) + MA5/MA20",
                xaxis_title=None, yaxis_title="Harga (IDR)",
                xaxis_rangeslider_visible=False, height=H_CANDLE,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40),
            )
            # Panah MACD di candle
            try:
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
                t_cdl = df_cdl["trade_date"].reset_index(drop=True)
                map_low  = pd.Series(low_c.values,  index=t_cdl)
                map_high = pd.Series(high_c.values, index=t_cdl)
                bull_dates = [d for d in trade_m[bull_idx_c] if d in map_low.index]
                bear_dates = [d for d in trade_m[bear_idx_c] if d in map_high.index]
                if len(bull_dates) > 0:
                    yb = [float(map_low[d]) * 0.995 for d in bull_dates]
                    figC.add_trace(go.Scatter(
                        x=bull_dates, y=yb, mode="markers", name="Bullish cross (MACD)",
                        marker=dict(symbol="triangle-up", size=12, color="#22c55e"),
                    ))
                if len(bear_dates) > 0:
                    ya = [float(map_high[d]) * 1.005 for d in bear_dates]
                    figC.add_trace(go.Scatter(
                        x=bear_dates, y=ya, mode="markers", name="Bearish cross (MACD)",
                        marker=dict(symbol="triangle-down", size=12, color="#ef4444"),
                    ))
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
    fast, slow, sig = get_macd_params()
    ema_fast = close_m.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = close_m.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=sig, adjust=False, min_periods=1).mean()
    hist = macd_line - signal_line
    delta = macd_line - signal_line
    prev_delta = delta.shift(1)
    bull_idx = (prev_delta <= 0) & (delta > 0)
    bear_idx = (prev_delta >= 0) & (delta < 0)

    x_bull = sub_m.loc[bull_idx, "trade_date"]; y_bull = macd_line[bull_idx]
    x_bear = sub_m.loc[bear_idx, "trade_date"]; y_bear = macd_line[bear_idx]

    figM = go.Figure()
    colorsM = np.where(hist >= 0, "rgba(51,183,102,0.7)", "rgba(220,53,69,0.7)")
    figM.add_bar(x=sub_m["trade_date"], y=hist, name="Histogram", marker_color=colorsM)
    figM.add_trace(go.Scatter(x=sub_m["trade_date"], y=macd_line, name="MACD",   line=dict(width=2, color="#0d6efd")))
    figM.add_trace(go.Scatter(x=sub_m["trade_date"], y=signal_line, name="Signal", line=dict(width=2, color="#fd7e14")))
    figM.add_trace(go.Scatter(
        x=x_bull, y=y_bull, mode="markers", name="Bullish crossover",
        marker=dict(symbol="triangle-up", size=12, color="#22c55e"),
        hovertemplate="%{x|%Y-%m-%d}<br>MACD: %{y:.2f}<extra>Bullish crossover</extra>",
    ))
    figM.add_trace(go.Scatter(
        x=x_bear, y=y_bear, mode="markers", name="Bearish crossover",
        marker=dict(symbol="triangle-down", size=12, color="#ef4444"),
        hovertemplate="%{x|%Y-%m-%d}<br>MACD: %{y:.2f}<extra>Bearish crossover</extra>",
    ))
    figM.add_hline(y=0, line_color="#adb5bd", line_width=1)
    figM.update_layout(
        title=f"MACD ({fast},{slow},{sig}) + Crossovers", xaxis_title=None, yaxis_title="MACD",
        height=H_MACD, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40),
    )
    _apply_time_axis(figM, sub_m["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(figM, use_container_width=True)

    try:
        delta_val = float(delta.iloc[-1])
        status = "Bullish" if delta_val > 0 else "Bearish"
        last_cross_type, last_cross_date = None, None
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
    sub_nf = df[df["net_foreign_value"].notna()][["trade_date", "net_foreign_value"]]
    colors = np.where(sub_nf["net_foreign_value"] >= 0, "rgba(51,183,102,0.7)", "rgba(220,53,69,0.7)")
    fig1.add_bar(x=sub_nf["trade_date"], y=sub_nf["net_foreign_value"], name="Net Foreign (Rp)", marker_color=colors, yaxis="y2")
    sub_cl = df[price_series.notna()][["trade_date"]].assign(close=price_series.dropna())
    fig1.add_trace(go.Scatter(x=sub_cl["trade_date"], y=sub_cl["close"], name="Close", mode="lines+markers", line=dict(color=THEME, width=2.2)))
    fig1.update_layout(
        title="Close vs Net Foreign Harian",
        xaxis_title=None,
        yaxis=dict(title="Close"),
        yaxis2=dict(title="Net Foreign (Rp)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=H_CLOSE_NF,
        margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40),
    )
    _apply_time_axis(fig1, df["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(fig1, use_container_width=True)
else:
    sub_nf = df[df["net_foreign_value"].notna()]
    fig1b = px.bar(
        sub_nf, x="trade_date", y="net_foreign_value", title="Net Foreign Harian (Rp)",
        color=(sub_nf["net_foreign_value"] >= 0).map({True: "Net Buy", False: "Net Sell"}),
        color_discrete_map={"Net Buy": "#33B766", "Net Sell": "#DC3545"},
    )
    fig1b.update_layout(height=H_CLOSE_NF, xaxis_title=None, yaxis_title="Net Foreign (Rp)", legend_title=None,
                        margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40))
    _apply_time_axis(fig1b, sub_nf["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(fig1b, use_container_width=True)

# Cum Net Foreign vs Close
if show_price and price_series is not None and price_series.notna().any():
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["trade_date"], y=df["cum_nf"], name="Cum. Net Foreign", line=dict(color="#6f42c1", width=2.2)))
    sub_cl = df[price_series.notna()][["trade_date"]].assign(close=price_series.dropna())
    fig2.add_trace(go.Scatter(x=sub_cl["trade_date"], y=sub_cl["close"], name="Close", yaxis="y2", line=dict(color=THEME, width=2), opacity=0.9))
    fig2.update_layout(
        title="Kumulatif Net Foreign vs Close",
        xaxis_title=None,
        yaxis=dict(title="Cum. Net Foreign"),
        yaxis2=dict(title="Close", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=H_CUM,
        margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40),
    )
    _apply_time_axis(fig2, df["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(fig2, use_container_width=True)
else:
    fig2b = px.line(df, x="trade_date", y="cum_nf", title="Kumulatif Net Foreign")
    fig2b.update_layout(height=H_CUM, xaxis_title=None, yaxis_title="Cum. Net Foreign",
                        margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40))
    _apply_time_axis(fig2b, df["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(fig2b, use_container_width=True)

# Buy vs Sell stacked
has_buy_sell_cols = set(["foreign_buy", "foreign_sell"]).issubset(df.columns)
if has_buy_sell_cols:
    sub = df[["trade_date", "foreign_buy", "foreign_sell"]].copy()
    sub["foreign_buy"]  = pd.to_numeric(sub["foreign_buy"],  errors="coerce")
    sub["foreign_sell"] = pd.to_numeric(sub["foreign_sell"], errors="coerce")
    sub = sub[(sub["foreign_buy"].notna()) | (sub["foreign_sell"].notna())]
    if not sub.empty:
        df_bs = sub.melt(id_vars=["trade_date"], value_vars=["foreign_buy", "foreign_sell"], var_name="jenis", value_name="nilai")
        fig3 = px.bar(
            df_bs, x="trade_date", y="nilai", color="jenis", title="Foreign Buy vs Sell (Rp)",
            color_discrete_map={"foreign_buy": "#0d6efd", "foreign_sell": "#DC3545"},
        )
        fig3.update_layout(barmode="stack", height=H_VOLUME, xaxis_title=None, yaxis_title="Rp", legend_title=None,
                           margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40))
        _apply_time_axis(fig3, sub["trade_date"], start, end, hide_non_trading)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Data Foreign Buy/Sell tidak tersedia untuk rentang ini.")
else:
    st.info("Data Foreign Buy/Sell tidak tersedia untuk rentang ini.")

# Nilai vs ADV + Ratio
fig4 = go.Figure()
sub_nilai = df[df["nilai"].notna()][["trade_date", "nilai"]]
fig4.add_bar(x=sub_nilai["trade_date"], y=sub_nilai["nilai"], name="Nilai (Rp)")
sub_adv = df[df["adv"].notna()][["trade_date", "adv"]]
fig4.add_trace(go.Scatter(x=sub_adv["trade_date"], y=sub_adv["adv"], name=f"ADV ({df.attrs.get('adv_label','ADV')})"))
sub_ratio = df[df["ratio"].notna()][["trade_date", "ratio"]]
fig4.add_trace(go.Scatter(x=sub_ratio["trade_date"], y=sub_ratio["ratio"], name="Ratio (Nilai/ADV)", yaxis="y2"))
fig4.update_layout(
    title=f"Nilai vs ADV ({df.attrs.get('adv_label','ADV')}) + Ratio",
    xaxis_title=None, yaxis=dict(title="Rp"),
    yaxis2=dict(title="Ratio (x)", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=H_NILAI,
    margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40),
)
_apply_time_axis(fig4, df["trade_date"], start, end, hide_non_trading)
st.plotly_chart(fig4, use_container_width=True)

# Volume vs AVG
if "volume" in df.columns:
    fig5 = go.Figure()
    sub_v  = df[df["volume"].notna()][["trade_date", "volume"]]
    fig5.add_bar(x=sub_v["trade_date"], y=sub_v["volume"], name="Volume")
    sub_va = df[df["vol_avg"].notna()][["trade_date", "vol_avg"]]
    fig5.add_trace(go.Scatter(x=sub_va["trade_date"], y=sub_va["vol_avg"], name="Vol AVG(20)"))
    fig5.update_layout(
        title="Volume vs AVG(20)", xaxis_title=None, yaxis_title="Lembar",
        height=H_VOLUME, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40),
    )
    _apply_time_axis(fig5, df["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(fig5, use_container_width=True)

# Spread (bps)
if show_spread and "spread_bps" in df.columns:
    sub_sp = df[df["spread_bps"].notna()][["trade_date", "spread_bps"]]
    if not sub_sp.empty:
        sp_vals = sub_sp["spread_bps"].dropna()
        p75 = float(np.nanpercentile(sp_vals, 75)) if len(sp_vals) else None
        fig6 = px.line(sub_sp, x="trade_date", y="spread_bps", title="Spread (bps)")
        if p75 is not None:
            fig6.add_hline(y=p75, line_dash="dot", line_color="#dc3545", annotation_text=f"P75 ≈ {p75:.1f}")
        fig6.update_layout(height=H_SPREAD, xaxis_title=None, yaxis_title="bps",
                           margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40))
        _apply_time_axis(fig6, sub_sp["trade_date"], start, end, hide_non_trading)
        st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ---------- Tabel & Export ----------
with st.expander("Tabel data mentah (siap export)"):
    cols_show = [
        "trade_date","kode_saham","nama_perusahaan","sebelumnya","open_price","first_trade",
        "tertinggi","terendah","penutupan","selisih","nilai","adv","ratio","volume","vol_avg",
        "foreign_buy","foreign_sell","net_foreign_value","spread_bps","close_price","cum_nf",
    ]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show], use_container_width=True, height=DF_HEIGHT)
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{kode}_foreign_flow_{start}_to_{end}.csv",
        mime="text/csv",
    )

st.caption("💡 Aktifkan **📱 Mobile Mode** untuk tampilan ringkas di HP: radio vertikal, chart dipendekkan, margin kecil, KPI ditumpuk.")

# ============================
# 🔎 Scanner — MACD Cross + Net Foreign (persist hasil + chips & export)
# ============================
st.divider()
with st.expander("🔎 Scanner — MACD Cross + Net Foreign", expanded=False):
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1: scan_all = st.checkbox("Scan semua kode", value=True, key="scan_all")
    with c2: recency  = st.number_input("Cross dalam X hari terakhir", min_value=1, max_value=365, value=15, step=1, key="scan_recency")
    with c3: require_above_zero = st.checkbox("Syarat MACD > 0", value=False, key="scan_macdpos")

    d1, d2, d3 = st.columns([1, 1, 1])
    with d1: require_nf = st.checkbox("Syarat NF rolling ≥ 0", value=True, key="scan_reqnf")
    with d2: nf_window  = st.number_input("Window NF (hari)", min_value=1, max_value=30, value=5, step=1, key="scan_nfwin")
    with d3:
        preset_txt = ",".join(map(str, get_macd_params()))
        st.caption(f"Preset MACD aktif: **{preset_txt}**")

    watchlist = codes if scan_all else st.multiselect("Watchlist", options=codes, default=[kode] if kode else [], key="scan_watchlist")
    run_scan = st.button("🚀 Jalankan Scan", type="primary", key="btn_scan")

    # --- Persist / restore hasil scan ---
    df_scan = None
    if run_scan:
        with st.spinner("Mengambil data & scanning cepat..."):
            fast, slow, sig = get_macd_params()
            warmup_days = max(slow * 5, 150)
            approx_days = int(warmup_days + recency + int(nf_window) + 30)
            start_scan = max(min_d, end - timedelta(days=approx_days))
            df_bulk = fetch_ohlc_bulk(watchlist, start_scan, end)
            df_scan = scan_universe_fast(
                df_bulk, fast, slow, sig,
                nf_window=int(nf_window), filter_nf=bool(require_nf),
                only_recent_days=int(recency), require_above_zero=bool(require_above_zero),
            )
        st.session_state["scanner_df"] = df_scan
        st.session_state["scanner_meta"] = {
            "start_scan": start_scan, "end": end, "nf_window": int(nf_window),
            "recency": int(recency), "require_nf": bool(require_nf), "require_above_zero": bool(require_above_zero),
        }
    else:
        df_scan = st.session_state.get("scanner_df", None)

    # Tombol clear
    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("🧹 Clear hasil scan", key="btn_clear_scan"):
            for k in ["scanner_df", "scanner_meta"]:
                if k in st.session_state: del st.session_state[k]
            st.stop()

    if df_scan is None or df_scan.empty:
        st.caption("Klik **Jalankan Scan** untuk mulai. Hasil scan disimpan agar tidak hilang saat klik tombol lain.")
    else:
        # dynamic NF col & p75
        nf_col = None
        for c in df_scan.columns:
            if c.startswith("NF_sum_") and c.endswith("d"):
                nf_col = c; break

        # sort default
        sort_cols, ascending = [], []
        if "qualifies" in df_scan.columns: sort_cols.append("qualifies"); ascending.append(False)
        if "days_ago" in df_scan.columns: sort_cols.append("days_ago"); ascending.append(True)
        if "last_cross_date" in df_scan.columns: sort_cols.append("last_cross_date"); ascending.append(False)
        if nf_col: sort_cols.append(nf_col); ascending.append(False)
        df_scan_sorted = df_scan.sort_values(sort_cols, ascending=ascending) if sort_cols else df_scan.copy()

        # chips
        st.markdown("**Quick filters**")
        ch1, ch2, ch3 = st.columns(3)
        with ch1:
            chip_recent3 = st.checkbox("≤ 3 hari", key="chip_recent3", value=False)
        with ch2:
            if nf_col and df_scan[nf_col].notna().any():
                nf_p75_val = float(np.nanpercentile(df_scan[nf_col], 75))
                chip_nf_p75 = st.checkbox(f"NF ≥ p75 (≈ {nf_p75_val:,.0f})", key="chip_nf_p75", value=False)
            else:
                nf_p75_val = None
                chip_nf_p75 = st.checkbox("NF ≥ p75", key="chip_nf_p75", value=False, disabled=True)
        with ch3:
            chip_macd_pos = st.checkbox("MACD > 0", key="chip_macd_pos", value=False)

        # apply chips
        view = df_scan_sorted.copy()
        if chip_recent3 and "days_ago" in view.columns:
            view = view[view["days_ago"] <= 3]
        if chip_macd_pos and "macd_above_zero" in view.columns:
            view = view[view["macd_above_zero"] == True]
        if chip_nf_p75 and nf_col and nf_p75_val is not None:
            view = view[view[nf_col] >= nf_p75_val]

        # show table
        order_cols = [
            "qualifies","days_ago","last_cross_date","kode","last_cross",
            "macd_above_zero", nf_col if nf_col else None, "close_last",
        ]
        order_cols = [c for c in order_cols if c and c in view.columns]
        st.dataframe(view[order_cols], use_container_width=True, height=SCAN_H)

        # export buttons
        if nf_col and nf_p75_val is not None:
            quick_watch = df_scan_sorted.copy()
            quick_watch = quick_watch[(quick_watch["days_ago"] <= 3) & (quick_watch[nf_col] >= nf_p75_val)]
            st.download_button(
                "⬇️ Export Watchlist (≤3D & NF≥p75)",
                data=quick_watch.to_csv(index=False).encode("utf-8"),
                file_name=f"watchlist_quick_{st.session_state.get('scanner_meta',{}).get('start_scan', start)}_to_{end}.csv",
                mime="text/csv",
            )
        st.download_button(
            "⬇️ Export Current View",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="scan_current_view.csv",
            mime="text/csv",
        )

        # --- Kirim ke Backtest per-ticker ---
        st.markdown("**Kirim ke Backtest (per-ticker)**")
        if view.empty:
            st.caption("— Tidak ada kandidat pada filter yang dipilih.")
        else:
            max_buttons = 50
            show_rows = view.head(max_buttons)
            for i, (_, row) in enumerate(show_rows.iterrows()):
                colA, colB, colC, colD, colE = st.columns([1.2, 1, 0.8, 1.2, 1])
                with colA: st.write(f"**{row.get('kode','-')}**")
                with colB: st.write(row.get("last_cross","-"))
                with colC:
                    dago = row.get("days_ago", np.nan)
                    st.write(f"D{int(dago)}" if pd.notna(dago) else "-")
                with colD:
                    if nf_col:
                        val = row.get(nf_col, np.nan)
                        st.write(f"NF: {val:,.0f}" if pd.notna(val) else "NF: -")
                    else:
                        st.write("NF: -")
                with colE:
                    if st.button("Kirim ke Backtest", key=f"send_bt_{i}_{row.get('kode','?')}"):
                        st.session_state["bt_symbol"] = row.get("kode")
                        st.session_state["open_backtest"] = True
                        # Rerun otomatis oleh mekanisme tombol; hasil scan tetap karena disimpan di session_state

# ============================
# 🧪 Backtest — MACD Rules (simple)  (Tampilan dirapikan)
# ============================
st.divider()
with st.expander("🧪 Backtest — MACD Rules (simple)", expanded=st.session_state.get("open_backtest", False)):
    # --- Controls row (ringkas)
    c1, c2, c3, c4 = st.columns([1.2, 0.9, 0.9, 0.9])
    current_bt_symbol = st.session_state.get("bt_symbol", st.session_state.get("kode_saham"))
    try:
        idx_default = codes.index(current_bt_symbol) if (codes and current_bt_symbol in codes) else 0
    except Exception:
        idx_default = 0
    with c1:
        bt_symbol = st.selectbox("Kode saham", options=codes, index=idx_default if idx_default < len(codes) else 0, key="bt_sym_sel")
        st.session_state["bt_symbol"] = bt_symbol
    with c2:
        bt_fee = st.number_input("Biaya/Slippage (bps per sisi)", min_value=0, max_value=100, value=0, step=1, key="bt_fee")
    with c3:
        bt_require_above = st.checkbox("Entry hanya jika MACD > 0", value=False, key="bt_above")
    with c4:
        bt_require_nf = st.checkbox("Entry hanya jika NF rolling ≥ 0", value=False, key="bt_reqnf")

    c5, _ = st.columns([0.9, 2.1])
    with c5:
        bt_nf_window  = st.number_input("NF window (hari)", min_value=1, max_value=30, value=5, step=1, key="bt_nfwin")

    # --- Run backtest
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
                    nf_window=int(bt_nf_window),
                    require_nf=bool(bt_require_nf),
                    fee_bp=int(bt_fee),
                )
            except Exception as e:
                st.error("Backtest error: " + str(e))
                trades, eq_df, stats = (
                    pd.DataFrame(), pd.DataFrame(),
                    {"trades": 0,"winrate": np.nan,"profit_factor": np.nan,"max_dd_pct": np.nan,
                     "cagr_pct": np.nan,"total_return_pct": np.nan,
                     "avg_trade_pct": np.nan,"median_trade_pct": np.nan,"best_trade_pct": np.nan,"worst_trade_pct": np.nan,
                     "avg_hold_days": np.nan,"sharpe": np.nan,"vol_annual_pct": np.nan,"calmar_ratio": np.nan},
                )

        # --- Metrics grid (2 baris)
        try:
            if not MOBILE:
                r1 = st.columns(6)
                r1[0].metric("Trades", int(stats.get("trades", 0)))
                r1[1].metric("Win Rate", f"{float(stats.get('winrate', np.nan)):.1f}%"
                            if stats.get('winrate', np.nan) == stats.get('winrate', np.nan) else "-")
                pf_val = stats.get("profit_factor", np.nan)
                pf_str = "∞" if (isinstance(pf_val, (float, int, np.floating)) and not math.isfinite(float(pf_val))) else (f"{float(pf_val):.2f}" if pf_val == pf_val else "-")
                r1[2].metric("Profit Factor", pf_str)
                r1[3].metric("Max DD", f"{float(stats.get('max_dd_pct', np.nan)):.1f}%"
                            if stats.get('max_dd_pct', np.nan) == stats.get('max_dd_pct', np.nan) else "-")
                r1[4].metric("Total Return", f"{float(stats.get('total_return_pct', np.nan)):.1f}%"
                            if stats.get('total_return_pct', np.nan) == stats.get('total_return_pct', np.nan) else "-")
                r1[5].metric("CAGR", f"{float(stats.get('cagr_pct', np.nan)):.1f}%"
                            if stats.get('cagr_pct', np.nan) == stats.get('cagr_pct', np.nan) else "-")

                r2 = st.columns(6)
                r2[0].metric("Avg Trade", f"{float(stats.get('avg_trade_pct', np.nan)):.2f}%"
                            if stats.get('avg_trade_pct', np.nan) == stats.get('avg_trade_pct', np.nan) else "-")
                r2[1].metric("Expectancy", f"{float(stats.get('avg_trade_pct', np.nan)):.2f}%")
                r2[2].metric("Median Trade", f"{float(stats.get('median_trade_pct', np.nan)):.2f}%"
                            if stats.get('median_trade_pct', np.nan) == stats.get('median_trade_pct', np.nan) else "-")
                r2[3].metric("Avg Hold", f"{float(stats.get('avg_hold_days', np.nan)):.1f} hari"
                            if stats.get('avg_hold_days', np.nan) == stats.get('avg_hold_days', np.nan) else "-")
                r2[4].metric("Best Trade", f"{float(stats.get('best_trade_pct', np.nan)):.2f}%"
                            if stats.get('best_trade_pct', np.nan) == stats.get('best_trade_pct', np.nan) else "-")
                r2[5].metric("Worst Trade", f"{float(stats.get('worst_trade_pct', np.nan)):.2f}%"
                            if stats.get('worst_trade_pct', np.nan) == stats.get('worst_trade_pct', np.nan) else "-")
                st.markdown(
                    f"<div class='metric-note'>Vol tahunan ≈ {float(stats.get('vol_annual_pct', np.nan)):.1f}% · Sharpe ≈ "
                    f"{float(stats.get('sharpe', np.nan)):.2f} · Calmar ≈ {float(stats.get('calmar_ratio', np.nan)):.2f}</div>",
                    unsafe_allow_html=True,
                )
            else:
                r1c1, r1c2 = st.columns(2)
                r1c1.metric("Trades", int(stats.get("trades", 0)))
                r1c2.metric("Win Rate", f"{float(stats.get('winrate', np.nan)):.1f}%"
                           if stats.get('winrate', np.nan) == stats.get('winrate', np.nan) else "-")
                r2c1, r2c2 = st.columns(2)
                pf_val = stats.get("profit_factor", np.nan)
                pf_str = "∞" if (isinstance(pf_val, (float, int, np.floating)) and not math.isfinite(float(pf_val))) else (f"{float(pf_val):.2f}" if pf_val == pf_val else "-")
                r2c1.metric("Profit Factor", pf_str)
                r2c2.metric("Max DD", f"{float(stats.get('max_dd_pct', np.nan)):.1f}%"
                           if stats.get('max_dd_pct', np.nan) == stats.get('max_dd_pct', np.nan) else "-")
                r3c1, r3c2 = st.columns(2)
                r3c1.metric("Total Return", f"{float(stats.get('total_return_pct', np.nan)):.1f}%"
                           if stats.get('total_return_pct', np.nan) == stats.get('total_return_pct', np.nan) else "-")
                r3c2.metric("CAGR", f"{float(stats.get('cagr_pct', np.nan)):.1f}%"
                           if stats.get('cagr_pct', np.nan) == stats.get('cagr_pct', np.nan) else "-")
                r4c1, r4c2 = st.columns(2)
                r4c1.metric("Avg Trade", f"{float(stats.get('avg_trade_pct', np.nan)):.2f}%"
                           if stats.get('avg_trade_pct', np.nan) == stats.get('avg_trade_pct', np.nan) else "-")
                r4c2.metric("Median Trade", f"{float(stats.get('median_trade_pct', np.nan)):.2f}%"
                           if stats.get('median_trade_pct', np.nan) == stats.get('median_trade_pct', np.nan) else "-")
                r5c1, r5c2 = st.columns(2)
                r5c1.metric("Avg Hold", f"{float(stats.get('avg_hold_days', np.nan)):.1f} hari"
                           if stats.get('avg_hold_days', np.nan) == stats.get('avg_hold_days', np.nan) else "-")
                r5c2.metric("Sharpe", f"{float(stats.get('sharpe', np.nan)):.2f}"
                           if stats.get('sharpe', np.nan) == stats.get('sharpe', np.nan) else "-")
                st.markdown(
                    f"<div class='metric-note'>Best {float(stats.get('best_trade_pct', np.nan)):.2f}% · Worst {float(stats.get('worst_trade_pct', np.nan)):.2f}% "
                    f"· Vol tahunan ≈ {float(stats.get('vol_annual_pct', np.nan)):.1f}%</div>",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.caption("⚠️ Gagal menampilkan ringkasan metrics: " + str(e))

        # --- Equity curve (rapih)
        if not eq_df.empty:
            figEQ = px.line(eq_df, x="date", y="equity", title="Equity Curve (1.0 = awal)")
            figEQ.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8", annotation_text="Start")
            figEQ.update_traces(line={'width': 2.2})
            figEQ.update_layout(height=H_MACD, margin=dict(l=10, r=10, t=40, b=10) if MOBILE else dict(l=40, r=30, t=60, b=40))
            _apply_time_axis(figEQ, pd.to_datetime(eq_df["date"]), start, end, hide_non_trading)
            st.plotly_chart(figEQ, use_container_width=True)
            st.download_button(
                "⬇️ Download Equity Curve CSV",
                data=eq_df.to_csv(index=False).encode("utf-8"),
                file_name=f"equity_curve_{bt_symbol}_{start}_to_{end}.csv",
                mime="text/csv",
            )

        # --- Trades table (format rapi)
        if not trades.empty:
            tv = trades.copy()
            for c in ["entry_price", "exit_price"]:
                if c in tv.columns:
                    tv[c] = tv[c].map(lambda x: format_money(x, 0))
            if "ret_pct" in tv.columns:
                tv["ret_pct"] = tv["ret_pct"].map(lambda x: f"{x:.2f}%")
            if "hold_days" in tv.columns:
                tv["hold_days"] = tv["hold_days"].astype(int)
            order_cols = [c for c in ["entry_date","entry_price","exit_date","exit_price","hold_days","ret_pct"] if c in tv.columns]
            st.dataframe(tv[order_cols], use_container_width=True, height=TRADES_H)
            st.download_button(
                "⬇️ Download Trades CSV",
                data=trades.to_csv(index=False).encode("utf-8"),
                file_name=f"trades_{bt_symbol}_{start}_to_{end}.csv",
                mime="text/csv",
            )
        else:
            st.info("Tidak ada trade pada parameter/rule ini.")
