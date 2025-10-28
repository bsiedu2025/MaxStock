# app/pages/13_Foreign_Flow_Detail.py
# Daily Stock Movement – Foreign Flow Focus
# v3.4 UIX: layout rapi & profesional
# - Range tanggal pakai satu widget (date range)
# - Quick range pakai radio horizontal (pills)
# - Hilangkan warning Session State (jangan set value & key bersamaan)
# - Toggle Hide non-trading days tetap ada, semua chart pakai rangebreaks saat ON

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from db_utils import get_db_connection, get_db_name

THEME = "#3AA6A0"
st.set_page_config(page_title="📈 Pergerakan Harian (Foreign Flow)", page_icon="📈", layout="wide")

# --- Minimal styling (pills & spacing) ---
st.markdown(
    """
    <style>
    /* tighten top spacing */
    section.main > div { padding-top: .5rem; }
    /* radio as segmented pills */
    div[role='radiogroup'] { display: flex; flex-wrap: wrap; gap: .25rem .5rem; }
    div[role='radiogroup'] label { border:1px solid #e5e7eb; padding:6px 12px; border-radius:9999px; background:#fff; }
    /* smaller captions */
    .small-note { color:#6b7280; font-size: .85rem; }
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

# ---------- UI (rapi) ----------
codes = list_codes()
min_d, max_d = date_bounds()

# defaults session state
if "date_range" not in st.session_state:
    st.session_state["date_range"] = (max(min_d, max_d - relativedelta(years=1)), max_d)
if "range_choice" not in st.session_state:
    st.session_state["range_choice"] = "1 Tahun"
if "hide_non_trading" not in st.session_state:
    st.session_state["hide_non_trading"] = True

# === FILTER BAR ===
with st.container():
    f1, f2 = st.columns([2,1])
    with f1:
        kode = st.selectbox("Pilih saham", options=codes, index=0 if codes else None)
    with f2:
        st.checkbox("Hide non-trading days (skip weekend & libur bursa)", key="hide_non_trading")

if not kode:
    st.stop()

# quick range (radio pills)
with st.container():
    td_series = get_trade_dates(kode)
    td_min = td_series.min().date() if not td_series.empty else min_d
    td_max = td_series.max().date() if not td_series.empty else max_d

    st.radio(
        "Range cepat",
        ["All", "1 Tahun", "6 Bulan", "3 Bulan", "1 Bulan", "5 Hari", "Custom"],
        key="range_choice",
        horizontal=True,
    )

    # apply quick range only when not Custom
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
        # update only if different
        if st.session_state.get("date_range") != (start, end):
            st.session_state["date_range"] = (start, end)
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    _apply_quick(st.session_state.get("range_choice", "1 Tahun"))

# date range input (single widget)
with st.container():
    dr = st.date_input(
        "Rentang tanggal",
        key="date_range",
        min_value=min_d,
        max_value=max_d,
    )
    # Jika user edit manual, set quick range ke Custom
    if isinstance(dr, tuple) and dr == st.session_state.get("date_range") and st.session_state.get("range_choice") != "Custom":
        pass
    else:
        st.session_state["range_choice"] = "Custom"

start, end = st.session_state["date_range"]

# ADV, show flags
with st.container():
    c4, c5, c6 = st.columns([1,1,1])
    with c4:
        adv_mode = st.selectbox("ADV window (hari)", ["All (Default)", "1 Tahun", "6 Bulan", "3 Bulan", "1 Bulan"], index=0)
    with c5:
        show_price = st.checkbox("Tampilkan harga (jika tersedia)", value=True)
    with c6:
        show_spread = st.checkbox("Tampilkan spread (bps) jika tersedia", value=True)

# ---------- Data ----------
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

# derive harga
last_close = price_series.dropna().iloc[-1] if (price_series is not None and price_series.notna().any()) else np.nan
if "selisih" in df.columns and df["selisih"].notna().any():
    change_val = pd.to_numeric(df["selisih"], errors="coerce").dropna().iloc[-1]
    if "sebelumnya" in df.columns and pd.notna(df["sebelumnya"].iloc[-1]) and df["sebelumnya"].iloc[-1] != 0:
        change_pct = change_val / float(df["sebelumnya"].iloc[-1]) * 100
    else:
        change_pct = np.nan
else:
    if "penutupan" in df.columns and "sebelumnya" in df.columns:
        prev = pd.to_numeric(df["sebelumnya"], errors="coerce").dropna().iloc[-1] if df["sebelumnya"].notna().any() else np.nan
        cls  = pd.to_numeric(df["penutupan"], errors="coerce").dropna().iloc[-1] if df["penutupan"].notna().any() else np.nan
        change_val = cls - prev if pd.notna(cls) and pd.notna(prev) else np.nan
        change_pct = (change_val / prev * 100) if pd.notna(change_val) and prev not in [0, np.nan] else np.nan
    else:
        change_val = np.nan
        change_pct = np.nan

high_p = pd.to_numeric(df.get("tertinggi"), errors="coerce").max(skipna=True) if "tertinggi" in df.columns else np.nan
low_p  = pd.to_numeric(df.get("terendah"),  errors="coerce").min(skipna=True) if "terendah"  in df.columns else np.nan
sma5   = price_series.rolling(5, min_periods=1).mean().dropna().iloc[-1] if (price_series is not None and price_series.notna().any()) else np.nan

st.markdown("### Ringkasan Metrik")
a,b,c = st.columns(3)
with a:
    st.metric("Harga Terakhir (IDR)", format_money(last_close) if pd.notna(last_close) else "-")
    st.metric("Perubahan Harga", format_money(change_val) if pd.notna(change_val) else "-", delta=(fmt_pct(change_pct) if pd.notna(change_pct) else None))
with b:
    st.metric("Tertinggi (Periode Filter)", format_money(high_p) if pd.notna(high_p) else "-")
    st.metric("Terendah (Periode Filter)",  format_money(low_p)  if pd.notna(low_p)  else "-")
with c:
    st.metric("SMA 5 Hari (IDR)", format_money(sma5) if pd.notna(sma5) else "-")

st.divider()

# ---------- Charts ----------
hide_non_trading = st.session_state.get("hide_non_trading", True)

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
            figC.update_layout(title="Candlestick (OHLC) + MA5/MA20", xaxis_title=None, yaxis_title="Harga (IDR)", xaxis_rangeslider_visible=False, height=460, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            _apply_time_axis(figC, df_cdl["trade_date"], start, end, hide_non_trading)
            st.plotly_chart(figC, use_container_width=True)
    else:
        st.info("Candlestick belum bisa ditampilkan karena kolom OHLC tidak lengkap pada rentang ini.")

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
fig4.add_bar(x=sub_nilai["trade_date"], y=sub_nilai["nilai"], name="Nilai (Rp)", marker_color="rgba(13,110,253,.6)")
sub_adv = df[df["adv"].notna()][["trade_date","adv"]]
fig4.add_trace(go.Scatter(x=sub_adv["trade_date"], y=sub_adv["adv"],   name=f"ADV ({df.attrs.get('adv_label','ADV')})", line=dict(color="#495057", width=2)))
sub_ratio = df[df["ratio"].notna()][["trade_date","ratio"]]
fig4.add_trace(go.Scatter(x=sub_ratio["trade_date"], y=sub_ratio["ratio"], name="Ratio (Nilai/ADV)", yaxis="y2", line=dict(color="#20c997", width=2)))
fig4.update_layout(title=f"Nilai vs ADV ({df.attrs.get('adv_label','ADV')}) + Ratio", xaxis_title=None, yaxis=dict(title="Rp"), yaxis2=dict(title="Ratio (x)", overlaying="y", side="right"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=420)
_apply_time_axis(fig4, df["trade_date"], start, end, hide_non_trading)
st.plotly_chart(fig4, use_container_width=True)

# Volume vs AVG
if "volume" in df.columns:
    fig5 = go.Figure()
    sub_v  = df[df["volume"].notna()][["trade_date","volume"]]
    fig5.add_bar(x=sub_v["trade_date"], y=sub_v["volume"], name="Volume", marker_color="rgba(32,201,151,.6)")
    sub_va = df[df["vol_avg"].notna()][["trade_date","vol_avg"]]
    fig5.add_trace(go.Scatter(x=sub_va["trade_date"], y=sub_va["vol_avg"], name="Vol AVG(20)", line=dict(color="#198754", width=2)))
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

st.caption("💡 Quick range = All/1Y/6M/3M/1M/5D. Rentang tanggal bisa diubah manual (otomatis switch ke *Custom*). Toggle untuk menyembunyikan hari non-trading. Semua chart otomatis skip tanggal tanpa data.")
