# app/pages/13_Foreign_Flow_Detail.py — rebuilt safe for Python 3.10
# Daily Stock Movement — Fokus Foreign Flow
# Features: sticky filter bar + shadow, non‑trading skip, quick range,
# candlestick + MA5/MA20 + MACD cross markers, MACD panel w/ presets,
# scanner (bulk, chunked, ranked) + quick set-to-chart, simple MACD backtest.

from datetime import date, timedelta
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from db_utils import get_db_connection, get_db_name

# -------------------- Page setup --------------------
THEME = "#3AA6A0"
st.set_page_config(
    page_title="📈 Pergerakan Harian (Foreign Flow)",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Pergerakan Harian Saham — Fokus Foreign Flow")

# DB badge
try:
    st.caption("DB aktif: " + (get_db_name() or "-"))
except Exception:
    st.caption("DB aktif: defaultdb")

# -------------------- Utilities --------------------

@st.cache_data(ttl=600, show_spinner=False)
def _get_codes():
    con = get_db_connection()
    try:
        df = pd.read_sql("SELECT DISTINCT kode_saham FROM data_harian ORDER BY 1", con)
        codes = df["kode_saham"].dropna().astype(str).tolist()
        return codes
    finally:
        try:
            con.close()
        except Exception:
            pass


def _alive(con):
    try:
        _ = pd.read_sql("SELECT 1", con)
    except Exception:
        pass


def _close(con):
    try:
        con.close()
    except Exception:
        pass


@st.cache_data(ttl=600, show_spinner=False)
def load_series(kode, start, end):
    """Load OHLC + NF subset for a single code."""
    con = get_db_connection(); _alive(con)
    try:
        # Detect columns to be resilient
        cols = pd.read_sql(
            """
            SELECT LOWER(column_name) AS col
            FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = 'data_harian'
            """,
            con,
        )
        have = set(cols["col"].tolist())

        price_candidates = ["penutupan", "close_price", "closing", "close"]
        close_col = next((c for c in price_candidates if c in have), None)
        open_col = "pembukaan" if "pembukaan" in have else ("open_price" if "open_price" in have else None)
        high_col = "tertinggi" if "tertinggi" in have else ("high_price" if "high_price" in have else None)
        low_col = "terendah" if "terendah" in have else ("low_price" if "low_price" in have else None)

        nf_expr = None
        if "net_foreign_value" in have:
            nf_expr = "net_foreign_value"
        elif {"foreign_buy", "foreign_sell"}.issubset(have):
            nf_expr = "(foreign_buy - foreign_sell)"
        else:
            nf_expr = "NULL"

        select_cols = ["kode_saham", "trade_date"]
        if open_col: select_cols.append(open_col + " AS open")
        if high_col: select_cols.append(high_col + " AS high")
        if low_col: select_cols.append(low_col + " AS low")
        if close_col: select_cols.append(close_col + " AS close")
        select_cols.append(nf_expr + " AS net_foreign_value")
        sql = (
            "SELECT " + ", ".join(select_cols) +
            " FROM data_harian WHERE kode_saham = %s AND trade_date BETWEEN %s AND %s ORDER BY trade_date"
        )
        df = pd.read_sql(sql, con, params=[kode, pd.to_datetime(start), pd.to_datetime(end)])
        if df.empty:
            return df
        df["trade_date"] = pd.to_datetime(df["trade_date"])  # normalize
        for c in ["open", "high", "low", "close", "net_foreign_value"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    finally:
        _close(con)


def _missing_business_days(dates):
    """Return list of business days missing between min and max (exclude weekend)."""
    if len(dates) == 0:
        return []
    idx = pd.date_range(pd.to_datetime(dates.min()).date(), pd.to_datetime(dates.max()).date(), freq="B")
    dset = set(pd.to_datetime(dates).date)
    # Build list of dates present
    present = set(pd.to_datetime(dates).date)
    missing = [pd.to_datetime(d).to_pydatetime() for d in idx.date if d not in present]
    return missing


def _apply_time_axis(fig, x_dates, start, end, hide_non_trading):
    if hide_non_trading:
        # Skip weekends + holidays (missing business days)
        rb = [dict(bounds=[6, 1], pattern="day of week")]  # 6=Sat .. 1=Mon
        miss = _missing_business_days(pd.to_datetime(x_dates))
        if len(miss) > 0:
            rb.append(dict(values=miss))
        fig.update_xaxes(rangebreaks=rb)
    fig.update_xaxes(range=[pd.to_datetime(start), pd.to_datetime(end)])


# ------- MACD helpers (no type hints for 3.10 safety) -------

def macd_series(close, fast, slow, sig):
    close = pd.to_numeric(pd.Series(close), errors="coerce").dropna()
    ema_fast = close.ewm(span=int(fast), adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=int(slow), adjust=False, min_periods=1).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=int(sig), adjust=False, min_periods=1).mean()
    delta = macd - signal
    return macd, signal, delta


def macd_cross_flags(delta):
    prev = delta.shift(1)
    bull = (prev <= 0) & (delta > 0)
    bear = (prev >= 0) & (delta < 0)
    return bull, bear


# -------- Bulk fetch for fast scanner --------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_ohlc_bulk(codes, start, end, chunk_size=200):
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
        price_col = None
        for c in ["penutupan", "close_price", "closing", "close"]:
            if c in have:
                price_col = c; break
        if price_col is None:
            raise RuntimeError("Tidak ada kolom harga penutupan (penutupan/close_price/closing/close)")
        if "net_foreign_value" in have:
            nf_expr = "net_foreign_value"
        elif {"foreign_buy", "foreign_sell"}.issubset(have):
            nf_expr = "(foreign_buy - foreign_sell)"
        else:
            nf_expr = "NULL"

        parts = []
        start_param = pd.to_datetime(start)
        end_param = pd.to_datetime(end)
        for i in range(0, len(codes), max(1, int(chunk_size))):
            chunk = codes[i:i+chunk_size]
            placeholders = ",".join(["%s"] * len(chunk))
            sql = f"""
                SELECT kode_saham, trade_date,
                       {price_col} AS close,
                       {nf_expr} AS net_foreign_value
                FROM data_harian
                WHERE trade_date BETWEEN %s AND %s
                  AND kode_saham IN ({placeholders})
                ORDER BY kode_saham, trade_date
            """
            params = [start_param, end_param] + list(chunk)
            part = pd.read_sql(sql, con, params=params)
            parts.append(part)
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        if df.empty:
            return df
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        if "net_foreign_value" in df.columns:
            df["net_foreign_value"] = pd.to_numeric(df["net_foreign_value"], errors="coerce")
        df = df.dropna(subset=["close"])
        return df
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
            last_bear = g.loc[bear, "trade_date"].iloc[-1]
            if last_date is None or pd.to_datetime(last_bear) > pd.to_datetime(last_date):
                last_type, last_date = "Bearish", last_bear
        if last_date is None:
            continue
        days_ago = int((global_end.normalize() - pd.to_datetime(last_date).normalize()).days)
        nf_sum = np.nan
        nf_ok = True
        if filter_nf and "net_foreign_value" in g.columns:
            nf_sum = float(g["net_foreign_value"].fillna(0.0).tail(int(nf_window)).sum())
            nf_ok = nf_sum >= 0
        macd_above = bool(macd.iloc[-1] > 0)
        qualifies = True
        if only_recent_days is not None:
            qualifies = qualifies and (days_ago <= int(only_recent_days))
        if require_above_zero:
            qualifies = qualifies and macd_above
        if filter_nf:
            qualifies = qualifies and nf_ok
        out.append({
            "kode": kd,
            "last_cross": last_type,
            "last_cross_date": pd.to_datetime(last_date).date(),
            "days_ago": days_ago,
            "macd_above_zero": macd_above,
            f"NF_sum_{int(nf_window)}d": nf_sum,
            "qualifies": bool(qualifies),
            "close_last": float(close.iloc[-1]),
        })
    return pd.DataFrame(out)


# -------------------- Session defaults --------------------
if "kode_saham" not in st.session_state:
    codes_default = _get_codes()
    st.session_state["kode_saham"] = codes_default[0] if codes_default else ""
if "date_range" not in st.session_state:
    today = date.today()
    st.session_state["date_range"] = (today - timedelta(days=365), today)
if "range_choice" not in st.session_state:
    st.session_state["range_choice"] = "1 Tahun"
if "hide_non_trading" not in st.session_state:
    st.session_state["hide_non_trading"] = True
if "show_price" not in st.session_state:
    st.session_state["show_price"] = True
if "show_spread" not in st.session_state:
    st.session_state["show_spread"] = True
# MACD params
if "macd_preset" not in st.session_state:
    st.session_state["macd_preset"] = "12-26-9 (Standard)"
if "macd_fast" not in st.session_state:
    st.session_state["macd_fast"] = 12
if "macd_slow" not in st.session_state:
    st.session_state["macd_slow"] = 26
if "macd_signal" not in st.session_state:
    st.session_state["macd_signal"] = 9


# -------------------- Styling --------------------
st.markdown(
    """
    <style>
    section.main > div { padding-top: .5rem; }
    .sticky-filter { position: sticky; top: 0; z-index: 1000; background: rgba(255,255,255,.96);
                     backdrop-filter: blur(6px); border-bottom: 1px solid #e5e7eb; padding: .5rem .25rem .75rem .25rem;
                     box-shadow: 0 8px 16px rgba(0,0,0,.06), 0 1px 0 rgba(0,0,0,.04); }
    div[role='radiogroup'] { display:flex; flex-wrap:wrap; gap:.5rem; }
    div[role='radiogroup'] label { border:1px solid #e5e7eb; padding:6px 12px; border-radius:9999px; background:#fff; }
    .checks-row { display:flex; align-items:center; gap: 1.25rem; }
    .checks-row div[data-testid='stCheckbox'] { margin-bottom: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Filter bar (sticky) --------------------
codes = _get_codes()
min_d = date(2018, 1, 1)
max_d = date.today()

with st.container():
    st.markdown("<div class='sticky-filter'>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1.3, 1.0, 1.6])
    with f1:
        kode = st.selectbox("Pilih saham", options=codes, key="kode_saham")
    with f2:
        st.selectbox("ADV window (hari)", ["All (Default)", "1 Tahun", "6 Bulan", "3 Bulan", "1 Bulan"], key="adv_mode")
    with f3:
        dr = st.date_input(
            "Rentang tanggal",
            value=st.session_state.get("date_range", (max(min_d, max_d - timedelta(days=365)), max_d)),
            min_value=min_d,
            max_value=max_d,
        )
        if isinstance(dr, tuple) and dr != st.session_state.get("date_range"):
            st.session_state["date_range"] = dr
            st.session_state["range_choice"] = "Custom"
            st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Quick range
st.markdown("**Range cepat**")
qr_cols = st.columns(6)
qr_labels = ["All", "1 Tahun", "6 Bulan", "3 Bulan", "1 Bulan", "5 Hari"]
for i, lbl in enumerate(qr_labels):
    with qr_cols[i]:
        if st.radio("", options=[lbl], key=f"qr_{i}") == lbl:
            pass
# Klik detection via on_change is tricky; we use small buttons-like radios
# We'll read which radio has a value (the one just set remains True once per rerun).
for i, lbl in enumerate(qr_labels):
    val = st.session_state.get(f"qr_{i}")
    if val == lbl and st.session_state.get("range_choice") != lbl:
        end = date.today()
        if lbl == "All":
            start = min_d
        elif lbl == "1 Tahun":
            start = end - timedelta(days=365)
        elif lbl == "6 Bulan":
            start = end - timedelta(days=182)
        elif lbl == "3 Bulan":
            start = end - timedelta(days=91)
        elif lbl == "1 Bulan":
            start = end - timedelta(days=30)
        else:
            start = end - timedelta(days=7)
        st.session_state["date_range"] = (start, end)
        st.session_state["range_choice"] = lbl
        st.experimental_rerun()

# Checkboxes row
with st.container():
    st.markdown("<div class='checks-row'>", unsafe_allow_html=True)
    cA, cB, cC = st.columns([1, 1, 1])
    with cA:
        st.checkbox("Tampilkan harga (jika tersedia)", key="show_price")
    with cB:
        st.checkbox("Tampilkan spread (bps) jika tersedia", key="show_spread")
    with cC:
        st.checkbox("Hide non-trading days (skip weekend & libur bursa)", key="hide_non_trading")
    st.markdown("</div>", unsafe_allow_html=True)

start, end = st.session_state["date_range"]
hide_non_trading = bool(st.session_state.get("hide_non_trading", True))

# -------------------- Data fetch --------------------
df = load_series(kode, start, end)
if df.empty:
    st.info("Data tidak tersedia untuk rentang ini.")
    st.stop()

# Choose available price series
close_series = None
for c in ["close", "penutupan", "close_price", "closing"]:
    if c in df.columns and df[c].notna().any():
        close_series = pd.to_numeric(df[c], errors="coerce")
        break

# -------------------- Candlestick --------------------
if {"open", "high", "low", "close"}.issubset(df.columns):
    df_cdl = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df_cdl["MA5"] = df_cdl["close"].rolling(5).mean()
    df_cdl["MA20"] = df_cdl["close"].rolling(20).mean()

    figC = go.Figure()
    figC.add_trace(go.Candlestick(
        x=df_cdl["trade_date"], open=df_cdl["open"], high=df_cdl["high"], low=df_cdl["low"], close=df_cdl["close"],
        name="OHLC"
    ))
    figC.add_trace(go.Scatter(x=df_cdl["trade_date"], y=df_cdl["MA5"], name="MA5", line=dict(width=1.5)))
    figC.add_trace(go.Scatter(x=df_cdl["trade_date"], y=df_cdl["MA20"], name="MA20", line=dict(width=1.5)))

    figC.update_layout(
        title="Candlestick (OHLC) + MA5/MA20",
        xaxis_title=None,
        yaxis_title="Harga (IDR)",
        xaxis_rangeslider_visible=False,
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # MACD cross markers on candlestick (recomputed from available close)
    try:
        # MACD params from preset/custom
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

        fast, slow, sig = get_macd_params()
        close_m = pd.to_numeric(df_cdl["close"], errors="coerce").dropna()
        trade_m = df_cdl.loc[close_m.index, "trade_date"].reset_index(drop=True)
        macd, signal, delta = macd_series(close_m.reset_index(drop=True), fast, slow, sig)
        prev_delta = delta.shift(1)
        bull_idx = (prev_delta <= 0) & (delta > 0)
        bear_idx = (prev_delta >= 0) & (delta < 0)

        t_cdl = df_cdl["trade_date"].reset_index(drop=True)
        map_low = pd.Series(df_cdl["low"].values, index=t_cdl)
        map_high = pd.Series(df_cdl["high"].values, index=t_cdl)
        bull_dates = [d for d in trade_m[bull_idx] if d in map_low.index]
        bear_dates = [d for d in trade_m[bear_idx] if d in map_high.index]
        if len(bull_dates) > 0:
            yb = [float(map_low[d]) * 0.995 for d in bull_dates]
            figC.add_trace(go.Scatter(x=bull_dates, y=yb, mode="markers", name="Bullish cross (MACD)",
                                      marker=dict(symbol="triangle-up", size=12)))
        if len(bear_dates) > 0:
            ya = [float(map_high[d]) * 1.005 for d in bear_dates]
            figC.add_trace(go.Scatter(x=bear_dates, y=ya, mode="markers", name="Bearish cross (MACD)",
                                      marker=dict(symbol="triangle-down", size=12)))
    except Exception:
        pass

    _apply_time_axis(figC, df_cdl["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(figC, use_container_width=True)

# -------------------- MACD Panel --------------------
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

# Compute MACD for panel
if close_series is not None and close_series.notna().any():
    def get_macd_params2():
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

    f_, s_, g_ = get_macd_params2()
    valid_idx = close_series.notna()
    sub_m = df.loc[valid_idx, ["trade_date"]].copy()
    close_m = pd.to_numeric(close_series[valid_idx], errors="coerce").dropna()
    macd_line, signal_line, delta = macd_series(close_m, f_, s_, g_)
    prev_delta = delta.shift(1)
    bull_idx = (prev_delta <= 0) & (delta > 0)
    bear_idx = (prev_delta >= 0) & (delta < 0)
    hist = macd_line - signal_line

    figM = go.Figure()
    colorsM = np.where(hist >= 0, "rgba(51,183,102,0.7)", "rgba(220,53,69,0.7)")
    figM.add_bar(x=sub_m["trade_date"], y=hist, name="Histogram", marker_color=colorsM)
    figM.add_trace(go.Scatter(x=sub_m["trade_date"], y=macd_line, name="MACD", line=dict(width=2)))
    figM.add_trace(go.Scatter(x=sub_m["trade_date"], y=signal_line, name="Signal", line=dict(width=2)))
    figM.add_trace(go.Scatter(x=sub_m.loc[bull_idx, "trade_date"], y=macd_line[bull_idx], mode="markers", name="Bullish crossover", marker=dict(symbol="triangle-up", size=12)))
    figM.add_trace(go.Scatter(x=sub_m.loc[bear_idx, "trade_date"], y=macd_line[bear_idx], mode="markers", name="Bearish crossover", marker=dict(symbol="triangle-down", size=12)))
    figM.add_hline(y=0, line_width=1)
    figM.update_layout(title=f"MACD ({f_},{s_},{g_}) + Crossovers", xaxis_title=None, yaxis_title="MACD", height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    _apply_time_axis(figM, sub_m["trade_date"], start, end, hide_non_trading)
    st.plotly_chart(figM, use_container_width=True)

    # Alert text
    try:
        last_type, last_date = None, None
        if bull_idx.any():
            last_type, last_date = "Bullish", sub_m.loc[bull_idx, "trade_date"].iloc[-1]
        if bear_idx.any():
            last_bear_date = sub_m.loc[bear_idx, "trade_date"].iloc[-1]
            if last_date is None or pd.to_datetime(last_bear_date) > pd.to_datetime(last_date):
                last_type, last_date = "Bearish", last_bear_date
        if last_date is not None:
            days_ago = (pd.to_datetime(df["trade_date"].max()).date() - pd.to_datetime(last_date).date()).days
            st.caption("⚡️ Cross terbaru: " + str(last_type) + " pada " + str(pd.to_datetime(last_date).date()) + " · " + str(days_ago) + " hari lalu.")
        delta_val = float((macd_line - signal_line).iloc[-1])
        st.caption("📉 MACD status saat ini: " + ("Bullish" if delta_val > 0 else "Bearish") + " (MACD - Signal = " + f"{delta_val:.2f}" + ").")
    except Exception:
        pass

# -------------------- Scanner --------------------
st.divider()
with st.expander("🔎 Scanner — MACD Cross + Net Foreign", expanded=False):
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        scan_all = st.checkbox("Scan semua kode", value=True)
    with c2:
        recency = st.number_input("Cross dalam X hari terakhir", min_value=1, max_value=365, value=15, step=1)
    with c3:
        require_above_zero = st.checkbox("Syarat MACD > 0", value=False)

    d1, d2, d3 = st.columns([1, 1, 1])
    with d1:
        require_nf = st.checkbox("Syarat NF rolling ≥ 0", value=True)
    with d2:
        nf_window = st.number_input("Window NF (hari)", min_value=1, max_value=30, value=5, step=1)
    with d3:
        ftmp, stmp, gtmp = 12, 26, 9
        preset_now = st.session_state.get("macd_preset", "12-26-9 (Standard)")
        if preset_now == "Custom":
            ftmp = int(st.session_state.get("macd_fast", 12))
            stmp = int(st.session_state.get("macd_slow", 26))
            gtmp = int(st.session_state.get("macd_signal", 9))
        elif preset_now == "5-35-5 (Fast)":
            ftmp, stmp, gtmp = 5, 35, 5
        elif preset_now == "8-17-9":
            ftmp, stmp, gtmp = 8, 17, 9
        elif preset_now == "10-30-9":
            ftmp, stmp, gtmp = 10, 30, 9
        elif preset_now == "20-50-9":
            ftmp, stmp, gtmp = 20, 50, 9
        st.caption("Preset MACD aktif: " + ",".join(map(str, [ftmp, stmp, gtmp])))

    watchlist = codes if scan_all else st.multiselect("Watchlist", options=codes, default=[kode] if kode else [])
    run_scan = st.button("🚀 Jalankan Scan", type="primary")

    if run_scan:
        with st.spinner("Mengambil data & scanning cepat..."):
            # Warm-up days for EMA stability
            warmup_days = max(stmp * 5, 150)
            approx_days = int(warmup_days + int(recency) + int(nf_window) + 30)
            start_scan = max(start, end - timedelta(days=approx_days))
            df_bulk = fetch_ohlc_bulk(watchlist, start_scan, end)
            df_scan = scan_universe_fast(
                df_bulk, ftmp, stmp, gtmp,
                nf_window=int(nf_window), filter_nf=bool(require_nf),
                only_recent_days=int(recency), require_above_zero=bool(require_above_zero),
            )
        if df_scan is None or df_scan.empty:
            st.info("Tidak ada hasil yang memenuhi filter.")
        else:
            # Ranking score: 40% NF, 30% recency, 20% MACD>0, 10% Bullish
            try:
                nf_col = "NF_sum_" + str(int(nf_window)) + "d"
                dfv = df_scan.copy()
                nf_vals = pd.to_numeric(dfv.get(nf_col, pd.Series(np.nan)), errors="coerce")
                if nf_vals.notna().any() and (nf_vals.max() - nf_vals.min()) > 0:
                    nf_norm = (nf_vals - nf_vals.min()) / (nf_vals.max() - nf_vals.min())
                else:
                    nf_norm = pd.Series(0.5, index=dfv.index)
                rec = np.exp(-0.35 * pd.to_numeric(dfv["days_ago"], errors="coerce").fillna(999))
                macdup = dfv.get("macd_above_zero", False).astype(bool).astype(int)
                bull = (dfv.get("last_cross", "") == "Bullish").astype(int)
                dfv["score"] = (0.4 * nf_norm + 0.3 * rec + 0.2 * macdup + 0.1 * bull).round(3)
            except Exception:
                dfv = df_scan.copy(); dfv["score"] = np.nan

            only_q = st.checkbox("Tampilkan hanya yang qualifies", value=True)
            dfshow = dfv[dfv["qualifies"]] if (only_q and "qualifies" in dfv.columns) else dfv
            order_cols = ["score", "qualifies", "days_ago", "last_cross_date", "kode", "last_cross", "macd_above_zero", nf_col, "close_last"]
            show_cols = [c for c in order_cols if c in dfshow.columns]
            df_ranked = dfshow.sort_values(["qualifies", "score", "days_ago"], ascending=[False, False, True])[show_cols]
            st.dataframe(df_ranked, use_container_width=True, height=440)
            if not df_ranked.empty:
                cc1, cc2 = st.columns([3, 1])
                with cc1:
                    pick_code = st.selectbox("Pilih kode untuk dibuka di chart", options=df_ranked["kode"].tolist())
                with cc2:
                    if st.button("Set ke chart", use_container_width=True):
                        st.session_state["kode_saham"] = pick_code
                        st.experimental_rerun()
            st.download_button(
                "⬇️ Download hasil scan (CSV)",
                data=dfv.to_csv(index=False).encode("utf-8"),
                file_name="scan_macd_" + str(start_scan) + "_to_" + str(end) + ".csv",
                mime="text/csv",
            )
    else:
        st.caption("Klik \"Jalankan Scan\" untuk mulai. Scanner menggunakan bulk query + vectorized MACD agar cepat.")

# -------------------- Backtest simple --------------------
st.divider()
with st.expander("🧪 Backtest — MACD Rules (simple)", expanded=False):
    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        idx_default = codes.index(kode) if (codes and kode in codes) else 0
        if idx_default >= len(codes): idx_default = 0
        bt_symbol = st.selectbox("Kode saham", options=codes, index=idx_default)
    with b2:
        bt_fee = st.number_input("Biaya/Slippage (bps per sisi)", min_value=0, max_value=100, value=0, step=1)
    with b3:
        bt_require_above = st.checkbox("Entry hanya jika MACD > 0", value=False)

    e1, e2 = st.columns([1, 1])
    with e1:
        bt_require_nf = st.checkbox("Entry hanya jika NF rolling ≥ 0", value=False)
    with e2:
        bt_nf_window = st.number_input("NF window (hari)", min_value=1, max_value=30, value=5, step=1)

    def backtest_macd(df_price, fast, slow, sig, require_above_zero=False, nf_window=0, require_nf=False, fee_bp=0):
        # choose close
        close = None
        for c in ["penutupan", "close_price", "closing", "close"]:
            if c in df_price.columns and df_price[c].notna().any():
                close = pd.to_numeric(df_price[c], errors="coerce"); break
        if close is None or close.dropna().empty:
            return pd.DataFrame(), pd.DataFrame(), {"trades":0, "winrate":np.nan, "profit_factor":np.nan, "max_dd_pct":np.nan, "cagr_pct":np.nan, "total_return_pct":np.nan}
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
                trades.append({"entry_date": entry_date.date(), "entry_price": entry_price, "exit_date": exit_date.date(), "exit_price": exit_price, "ret_pct": ret * 100.0})
                in_pos = False; entry_price = None; entry_date = None
            equity_curve.append({"date": d, "equity": equity})
        if in_pos:
            exit_price = float(close_v.iloc[-1]); exit_date = dates.iloc[-1]
            ret = (exit_price / entry_price - 1.0) - 2 * fee
            equity *= (1.0 + ret)
            trades.append({"entry_date": entry_date.date(), "entry_price": entry_price, "exit_date": exit_date.date(), "exit_price": exit_price, "ret_pct": ret * 100.0})
            if equity_curve:
                equity_curve[-1]["equity"] = equity
        trades_df = pd.DataFrame(trades)
        eq_df = pd.DataFrame(equity_curve)
        if trades_df.empty:
            stats = {"trades":0, "winrate":np.nan, "profit_factor":np.nan, "max_dd_pct":np.nan, "cagr_pct":np.nan, "total_return_pct": (eq_df["equity"].iloc[-1] - 1.0) * 100.0 if not eq_df.empty else np.nan}
            return trades_df, eq_df, stats
        wins = trades_df[trades_df["ret_pct"] > 0]
        losses = trades_df[trades_df["ret_pct"] <= 0]
        pf = (wins["ret_pct"].sum() / abs(losses["ret_pct"].sum())) if not losses.empty else float("inf")
        winrate = (len(wins) / len(trades_df)) * 100.0
        if not eq_df.empty:
            ec = eq_df.set_index("date")["equity"]
            roll_max = ec.cummax()
            max_dd = ((ec / roll_max) - 1.0).min() * 100.0
            days_span = (ec.index.max().date() - ec.index.min().date()).days if len(ec) > 1 else 0
            cagr = ((ec.iloc[-1]) ** (365.0 / days_span) - 1.0) * 100.0 if days_span > 0 else np.nan
            total_ret = (ec.iloc[-1] - 1.0) * 100.0
        else:
            max_dd = np.nan; cagr = np.nan; total_ret = np.nan
        stats = {"trades": int(len(trades_df)), "winrate": float(winrate), "profit_factor": float(pf), "max_dd_pct": float(max_dd), "cagr_pct": float(cagr), "total_return_pct": float(total_ret)}
        return trades_df, eq_df, stats

    df_bt = load_series(bt_symbol, start, end)
    if df_bt.empty:
        st.info("Data kosong untuk backtest.")
    else:
        with st.spinner("Menjalankan backtest..."):
            try:
                # Use current preset
                fbt, sbt, gbt = 12, 26, 9
                pnow = st.session_state.get("macd_preset", "12-26-9 (Standard)")
                if pnow == "Custom":
                    fbt = int(st.session_state.get("macd_fast", 12)); sbt = int(st.session_state.get("macd_slow", 26)); gbt = int(st.session_state.get("macd_signal", 9))
                elif pnow == "5-35-5 (Fast)":
                    fbt, sbt, gbt = 5, 35, 5
                elif pnow == "8-17-9":
                    fbt, sbt, gbt = 8, 17, 9
                elif pnow == "10-30-9":
                    fbt, sbt, gbt = 10, 30, 9
                elif pnow == "20-50-9":
                    fbt, sbt, gbt = 20, 50, 9
                trades, eq_df, stats = backtest_macd(df_bt, fbt, sbt, gbt, require_above_zero=bool(bt_require_above), nf_window=int(bt_nf_window), require_nf=bool(bt_require_nf), fee_bp=int(bt_fee))
            except Exception as e:
                st.error("Backtest error: " + str(e))
                trades, eq_df, stats = pd.DataFrame(), pd.DataFrame(), {"trades":0, "winrate":np.nan, "profit_factor":np.nan, "max_dd_pct":np.nan, "cagr_pct":np.nan, "total_return_pct":np.nan}

        # Metrics
        try:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Trades", int(stats.get("trades", 0)))
            wv = stats.get("winrate", np.nan)
            m2.metric("Win Rate", ("{:.1f}%".format(float(wv))) if wv == wv else "-")
            pf_val = stats.get("profit_factor", np.nan)
            pf_str = "∞" if (isinstance(pf_val, (float, int)) and not math.isfinite(float(pf_val))) else (("{:.2f}".format(float(pf_val))) if pf_val == pf_val else "-")
            m3.metric("Profit Factor", pf_str)
            ddv = stats.get("max_dd_pct", np.nan)
            m4.metric("Max DD", ("{:.1f}%".format(float(ddv))) if ddv == ddv else "-")
            trv = stats.get("total_return_pct", np.nan)
            m5.metric("Total Return", ("{:.1f}%".format(float(trv))) if trv == trv else "-")
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
                file_name="trades_" + str(bt_symbol) + "_" + str(start) + "_to_" + str(end) + ".csv",
                mime="text/csv",
            )
        else:
            st.info("Tidak ada trade pada parameter/rule ini.")
