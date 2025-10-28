# pages/13_Foreign_Flow_Detail.py
# ==========================================================
# Pergerakan Harian Saham — Fokus Foreign Flow
# Update:
#  - Skip tanggal tanpa data (rangebreaks weekend + missing dates)
#  - Quick range selector (1Y default) untuk rentang harga
#  - Expander "Cara baca & Ringkasan" per chart (default hidden)
#  - Default rentang = 1 tahun
# ==========================================================

import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from functools import lru_cache
from sqlalchemy import create_engine

# ----------------------------------------------------------
# DB & Data helpers
# ----------------------------------------------------------
@lru_cache(maxsize=64)
def get_engine():
    # sesuaikan kredensialmu
    # contoh: mysql+pymysql://user:password@host:3306/dbname
    url = st.secrets["connections"]["mysql"]["url"]  # atau hardcode url DSN-mu
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)

@st.cache_data(show_spinner=False, ttl=600)
def load_series(kode: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Ambil time-series harga & foreign flow dari table data_harian.
    Kolom yang dipakai:
      trade_date, kode_saham,
      penutupan AS close_price, open_price, tertinggi, terendah, volume,
      foreign_buy, foreign_sell
    """
    q = """
    SELECT
        trade_date,
        kode_saham,
        penutupan    AS close_price,
        open_price,
        tertinggi    AS high,
        terendah     AS low,
        volume,
        foreign_buy,
        foreign_sell
    FROM data_harian
    WHERE kode_saham = %(kode)s
      AND trade_date BETWEEN %(start)s AND %(end)s
    ORDER BY trade_date ASC;
    """
    df = pd.read_sql(q, get_engine(), params={"kode": kode, "start": start, "end": end})
    if df.empty:
        return df

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["net_foreign"] = (df["foreign_buy"] - df["foreign_sell"]).fillna(0.0)
    # jadikan index untuk memudahkan kalkulasi rolling
    df = df.set_index("trade_date").sort_index()
    # MA (boleh disimpan untuk dipakai di candle)
    df["sma5"] = df["close_price"].rolling(5).mean()
    df["sma20"] = df["close_price"].rolling(20).mean()
    return df

# ----------------------------------------------------------
# UI helpers
# ----------------------------------------------------------
RANGE_OPTS = (
    "1 Tahun (Default)",   # default index=0
    "6 Bulan",
    "3 Bulan",
    "1 Bulan",
    "All",
)

def _infer_start_from_range(df: pd.DataFrame, opt: str) -> pd.Timestamp:
    """Hitung tanggal mulai dari pilihan quick range."""
    if df.empty:
        return pd.Timestamp.today() - pd.Timedelta(days=365)

    last = df.index.max()
    if opt == "All":
        return df.index.min()
    span = {
        "1 Tahun (Default)": 365,
        "6 Bulan": 182,
        "3 Bulan": 90,
        "1 Bulan": 30,
    }.get(opt, 365)
    return (last - pd.Timedelta(days=span)).normalize()

def _missing_dates_index(df: pd.DataFrame) -> list:
    """Kumpulkan semua tanggal kalender dalam rentang – tanggal yang tdk ada datanya (libur/no trade)."""
    if df.empty:
        return []
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    miss = full.difference(df.index)
    # plotly butuh list-of-str / pandas Timestamp; kita kirim list Timestamp
    return list(miss)

def _fmt_idr_short(x: float) -> str:
    """Format angka pendek (Juta/Miliar/Triliun) gaya Indonesia."""
    if x is None or np.isnan(x):
        return "-"
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1_000_000_000_000:
        return f"{sign}{a/1_000_000_000_000:.2f} T"
    if a >= 1_000_000_000:
        return f"{sign}{a/1_000_000_000:.2f} M"
    if a >= 1_000_000:
        return f"{sign}{a/1_000_000:.2f} Jt"
    if a >= 1_000:
        return f"{sign}{a/1_000:.2f} Rb"
    return f"{sign}{a:,.0f}"

# ----------------------------------------------------------
# Chart builders
# ----------------------------------------------------------
def make_candlestick(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> go.Figure:
    d = df.loc[(df.index >= start) & (df.index <= end)].copy()
    if d.empty:
        fig = go.Figure()
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        fig.add_annotation(text="Tidak ada data pada rentang ini",
                           showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=d.index,
        open=d["open_price"],
        high=d["high"],
        low=d["low"],
        close=d["close_price"],
        name="OHLC"
    ))
    # SMA 5 & SMA 20
    fig.add_trace(go.Scatter(x=d.index, y=d["sma5"], name="SMA 5", mode="lines",
                             line=dict(color="#1f77b4", width=1.8)))
    fig.add_trace(go.Scatter(x=d.index, y=d["sma20"], name="SMA 20", mode="lines",
                             line=dict(color="#7f7f7f", width=1.8)))

    # Hilangkan tanggal tanpa data (weekend & missing trading days)
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),                 # skip akhir pekan
            dict(values=_missing_dates_index(d)),        # skip libur/no trade
        ]
    )

    fig.update_layout(
        height=420,
        margin=dict(l=15, r=15, t=35, b=10),
        yaxis_title="Harga (IDR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        title="Candlestick + SMA (5/20)"
    )
    return fig

def make_close_vs_netforeign(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> go.Figure:
    d = df.loc[(df.index >= start) & (df.index <= end)].copy()
    if d.empty:
        fig = go.Figure()
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        fig.add_annotation(text="Tidak ada data pada rentang ini",
                           showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    fig = go.Figure()

    # Net foreign bars (secondary axis)
    colors = np.where(d["net_foreign"] >= 0, "#2ca02c", "#d62728")
    fig.add_trace(go.Bar(
        x=d.index, y=d["net_foreign"],
        name="Net Foreign (Rp)",
        marker_color=colors,
        yaxis="y2"
    ))

    # Close line (primary axis)
    fig.add_trace(go.Scatter(
        x=d.index, y=d["close_price"],
        name="Close",
        mode="lines+markers",
        line=dict(color="#17becf", width=2),
        marker=dict(size=3)
    ))

    # Hilangkan tanggal tanpa data
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(values=_missing_dates_index(d)),
        ]
    )

    fig.update_layout(
        height=420,
        margin=dict(l=15, r=15, t=35, b=10),
        template="plotly_white",
        title="Close vs Net Foreign Harian",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title="Close"),
        yaxis2=dict(title="Net Foreign (Rp)", overlaying="y", side="right", showgrid=False),
        barmode="relative"
    )
    return fig

# ----------------------------------------------------------
# Summary text helpers
# ----------------------------------------------------------
def make_price_summary(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> str:
    d = df.loc[(df.index >= start) & (df.index <= end)]
    if d.empty:
        return "Tidak ada data pada rentang ini."
    close_last = d["close_price"].iloc[-1]
    close_first = d["close_price"].iloc[0]
    chg = close_last - close_first
    chg_pct = (chg / close_first) * 100 if close_first else np.nan
    hi = d["high"].max()
    lo = d["low"].min()
    return (
        f"- **Close terakhir**: {_fmt_idr_short(close_last)}\n"
        f"- **Perubahan**: {_fmt_idr_short(chg)} ({chg_pct:.2f}%)\n"
        f"- **Tertinggi periode**: {_fmt_idr_short(hi)}  •  "
        f"**Terendah periode**: {_fmt_idr_short(lo)}\n"
        f"- **Rata-rata volume**: {_fmt_idr_short(d['volume'].mean())}\n"
    )

def make_nf_summary(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> str:
    d = df.loc[(df.index >= start) & (df.index <= end)]
    if d.empty:
        return "Tidak ada data pada rentang ini."
    nf_total = d["net_foreign"].sum()
    max_buy_dt = d["net_foreign"].idxmax()
    max_buy = d["net_foreign"].max()
    max_sell_dt = d["net_foreign"].idxmin()
    max_sell = d["net_foreign"].min()
    return (
        f"- **Akumulasi Net Foreign**: {_fmt_idr_short(nf_total)}\n"
        f"- **Hari Net Buy terbesar**: {max_buy_dt.date()} "
        f"({_fmt_idr_short(max_buy)})\n"
        f"- **Hari Net Sell terbesar**: {max_sell_dt.date()} "
        f"({_fmt_idr_short(max_sell)})\n"
        f"- Interpretasi cepat: bar hijau (net buy) yang **konsisten** "
        f"bersama **harga uptrend** → akumulasi; bar merah + harga melemah → distribusi.\n"
    )

def help_price():
    return (
        "**Cara membaca:**\n"
        "- Candlestick menunjukkan **OHLC** harian. Garis **SMA5** dan **SMA20** "
        "membantu melihat momentum jangka pendek.\n"
        "- Break di atas/di bawah MA sering dipakai sebagai konfirmasi arah.\n"
        "- Rangebreaks menghilangkan tanggal tanpa transaksi (weekend/libur).\n"
    )

def help_nf():
    return (
        "**Cara membaca:**\n"
        "- Bar **Net Foreign** hijau = net buy; merah = net sell. Garis menunjukkan **Close**.\n"
        "- Kumpulkan konteks: bila bar hijau dominan **dan** close naik → akumulasi; sebaliknya distribusi.\n"
        "- Perhatikan juga ukuran bar vs historinya untuk deteksi **spike** yang tidak biasa.\n"
    )

# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
st.set_page_config(page_title="Pergerakan Harian Saham — Foreign Flow", layout="wide")

st.title("Pergerakan Harian Saham — Fokus Foreign Flow")

col0, col1, col2 = st.columns([2, 1.2, 1.2])

# --- Input utama
with col0:
    kode = st.selectbox("Pilih saham", options=st.session_state.get("DAFTAR_KODE", ["TLKM", "BBCA", "BBRI", "ASII"]), index=0)

with col1:
    # default 1 tahun kebelakang; end default = hari ini
    end_date = st.date_input("Sampai tanggal", value=dt.date.today())
with col2:
    start_date = st.date_input("Dari tanggal", value=(dt.date.today() - dt.timedelta(days=365)))

# Ambil data (ambil lebih panjang; nanti dipotong oleh quick-range)
raw = load_series(kode, start_date, end_date)

# --- Quick range untuk harga (default 1 tahun)
st.markdown("#### Rentang Harga (Quick)")
opt_range = st.selectbox("Pilih rentang", options=RANGE_OPTS, index=0, label_visibility="collapsed")
if raw.empty:
    st.warning("Data kosong untuk kode & rentang tanggal ini.")
    st.stop()

# Hitung start range dari pilihan
start_q = _infer_start_from_range(raw, opt_range)
end_q = raw.index.max()

# ----------------------------------------------------------
# Header ringkasan singkat
# ----------------------------------------------------------
st.markdown(f"### {kode}")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
d_q = raw.loc[(raw.index >= start_q) & (raw.index <= end_q)]
with kpi1:
    st.metric("Close terakhir", _fmt_idr_short(d_q["close_price"].iloc[-1]) if not d_q.empty else "-")
with kpi2:
    chg = (d_q["close_price"].iloc[-1] - d_q["close_price"].iloc[0]) if len(d_q) > 1 else 0
    chg_pct = (chg / d_q["close_price"].iloc[0] * 100) if len(d_q) > 1 else 0
    st.metric("Perubahan", _fmt_idr_short(chg), f"{chg_pct:.2f}%")
with kpi3:
    st.metric("Tertinggi (periode)", _fmt_idr_short(d_q["high"].max()) if not d_q.empty else "-")
with kpi4:
    st.metric("Terendah (periode)", _fmt_idr_short(d_q["low"].min()) if not d_q.empty else "-")
with kpi5:
    st.metric("Akumulasi Net Foreign", _fmt_idr_short(d_q["net_foreign"].sum()) if not d_q.empty else "-")

st.divider()

# ----------------------------------------------------------
# CHART 1: Candlestick
# ----------------------------------------------------------
st.subheader("Candlestick Harga Harian")
fig_price = make_candlestick(raw, start_q, end_q)
st.plotly_chart(fig_price, use_container_width=True)

with st.expander("Cara membaca & ringkasan (hide/show)", expanded=False):
    st.markdown(help_price())
    st.markdown("**Ringkasan periode:**")
    st.markdown(make_price_summary(raw, start_q, end_q))

st.divider()

# ----------------------------------------------------------
# CHART 2: Close vs Net Foreign Harian
# ----------------------------------------------------------
st.subheader("Close vs Net Foreign Harian")
fig_nf = make_close_vs_netforeign(raw, start_q, end_q)
st.plotly_chart(fig_nf, use_container_width=True)

with st.expander("Cara membaca & ringkasan (hide/show)", expanded=False):
    st.markdown(help_nf())
    st.markdown("**Ringkasan periode:**")
    st.markdown(make_nf_summary(raw, start_q, end_q))

st.caption(
    "Catatan: sumbu waktu menggunakan *rangebreaks* untuk menghilangkan weekend/ tanggal tanpa transaksi. "
    "Rentang default 1 tahun; ubah via menu **Rentang Harga (Quick)**."
)
