# app/pages/13_Foreign_Flow_Detail.py
# Daily Stock Movement – Foreign Flow Focus
# v3.0: + Candlestick (OHLC) + MA5/MA20, Ringkasan Metrik dari kolom harga

from datetime import date, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from db_utils import get_db_connection, get_db_name

THEME = "#3AA6A0"
st.set_page_config(page_title="📈 Pergerakan Harian (Foreign Flow)", page_icon="📈", layout="wide")


# ------------------------------ #
# Utilities koneksi MySQL
# ------------------------------ #
def _alive(con):
    try:
        if hasattr(con, "is_connected") and not con.is_connected():
            con.reconnect()
    except Exception:
        try:
            con.ping(reconnect=True, attempts=1, delay=0)
        except Exception:
            pass

def _close(con):
    try:
        if hasattr(con, "is_connected"):
            if con.is_connected():
                con.close()
        else:
            con.close()
    except Exception:
        pass


# ------------------------------ #
# Cached loaders
# ------------------------------ #
@st.cache_data(ttl=300, show_spinner=False)
def list_kode() -> pd.DataFrame:
    con = get_db_connection(); _alive(con)
    try:
        df = pd.read_sql(
            "SELECT kode_saham, nama_perusahaan FROM master_saham ORDER BY kode_saham",
            con,
        )
        return df
    finally:
        _close(con)

@st.cache_data(ttl=300, show_spinner=False)
def get_minmax_date(kode: str):
    con = get_db_connection(); _alive(con)
    try:
        r = pd.read_sql(
            """
            SELECT MIN(trade_date) as mn, MAX(trade_date) as mx
            FROM data_harian WHERE kode_saham=%s
            """,
            con, params=[kode]
        ).iloc[0]
        mn, mx = r["mn"], r["mx"]
        if pd.isna(mn) or pd.isna(mx):
            today = date.today()
            return (today - timedelta(days=60), today)
        return (pd.to_datetime(mn).date(), pd.to_datetime(mx).date())
    finally:
        _close(con)

@st.cache_data(ttl=300, show_spinner=False)
def load_series(kode: str, start: date, end: date) -> pd.DataFrame:
    """Ambil kolom yang tersedia; yang tidak ada diisi NaN agar UI tetap stabil."""
    con = get_db_connection(); _alive(con)
    try:
        cols_df = pd.read_sql(
            "SELECT LOWER(column_name) col FROM information_schema.columns "
            "WHERE table_schema=DATABASE() AND table_name='data_harian'",
            con
        )
        cols = set(cols_df["col"].tolist())
        # kolom yang mungkin tersedia
        need_cols = [
            "trade_date", "kode_saham", "nama_perusahaan",
            "nilai", "foreign_buy", "foreign_sell",
            "volume", "sebelumnya", "open_price", "first_trade",
            "tertinggi", "terendah", "penutupan", "selisih",
            "bid", "offer"
        ]
        sel = [c for c in need_cols if c in cols]
        sel_clause = ", ".join(sel)

        df = pd.read_sql(
            f"""
            SELECT {sel_clause}
            FROM data_harian
            WHERE kode_saham=%s AND trade_date BETWEEN %s AND %s
            ORDER BY trade_date
            """,
            con, params=[kode, start, end]
        )

        # normalisasi nama kolom -> lower
        df.columns = [c.lower() for c in df.columns]
        # pastikan semua kolom penting ada dengan NaN jika tak tersedia
        for c in need_cols:
            c2 = c.lower()
            if c2 not in df.columns:
                df[c2] = np.nan

        # nilai spread (bps) jika bid/offer tersedia
        if {"bid", "offer"}.issubset(set(df.columns)):
            with np.errstate(divide='ignore', invalid='ignore'):
                df["spread_bps"] = (df["offer"] - df["bid"]) / ((df["offer"] + df["bid"]) / 2) * 10000.0
        else:
            df["spread_bps"] = np.nan

        # Hitung Net Foreign Harian
        if {"foreign_buy", "foreign_sell"}.issubset(set(df.columns)):
            df["net_foreign"] = df["foreign_buy"] - df["foreign_sell"]
        else:
            df["net_foreign"] = np.nan

        # ADV harian dari window default 1 tahun (di chart lain akan ada opsi)
        # Untuk halaman ini kita sediakan rolling 20D & 60D supaya cepat
        if "nilai" in df.columns:
            df["adv20"] = df["nilai"].rolling(20, min_periods=1).mean()
            df["adv60"] = df["nilai"].rolling(60, min_periods=1).mean()
        else:
            df["adv20"] = np.nan
            df["adv60"] = np.nan

        return df
    finally:
        _close(con)


# ------------------------------ #
# Helper presentasi
# ------------------------------ #
def fmt_idr(x):
    if pd.isna(x):
        return "-"
    try:
        # format ribuan / jutaan / milyar
        v = float(x)
        abs_v = abs(v)
        if abs_v >= 1_000_000_000_000:
            return f"{v/1_000_000_000_000:.2f} T"
        if abs_v >= 1_000_000_000:
            return f"{v/1_000_000_000:.2f} M"
        if abs_v >= 1_000_000:
            return f"{v/1_000_000:.2f} Juta"
        if abs_v >= 1_000:
            return f"{v/1_000:.2f} Rb"
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        return str(x)

def pct_str(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return "-"
        return f"{(a-b)/b*100:.2f}%"
    except Exception:
        return "-"


# ------------------------------ #
# UI – Control
# ------------------------------ #
st.title("Pergerakan Harian Saham — Fokus Foreign Flow")
st.caption(f"DB aktif: **{get_db_name()}**")

kode_df = list_kode()
kode_list = kode_df["kode_saham"].tolist()
kode = st.selectbox("Pilih saham", options=kode_list, index=0)

min_d, max_d = get_minmax_date(kode)
col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    start = st.date_input("Dari tanggal", value=max(min_d, date.today()-timedelta(days=365)), min_value=min_d, max_value=max_d)
with col_b:
    end = st.date_input("Sampai tanggal", value=max_d, min_value=min_d, max_value=max_d)
with col_c:
    adv_mode = st.selectbox("ADV window (hari)",
                            options=["All (Default)", "1 Tahun", "6 Bulan", "3 Bulan", "1 Bulan"],
                            index=0)

show_price = st.checkbox(" ✅ Tampilkan harga (jika ada close_price)", value=True)
show_spread = st.checkbox(" ✅ Tampilkan spread (bps) jika tersedia", value=True)

df = load_series(kode, start, end)

# nama saham
nama = df["nama_perusahaan"].dropna().unique()
judul = f"{kode}"
if len(nama) > 0:
    judul += f" — {nama[0]}"
st.subheader(judul)

# ------------------------------ #
# KPI Ringkas (Total Foreign)
# ------------------------------ #
k1, k2, k3, k4, k5 = st.columns(5)

total_net = df["net_foreign"].sum(skipna=True)
total_buy = df["foreign_buy"].sum(skipna=True) if "foreign_buy" in df.columns else np.nan
total_sell = df["foreign_sell"].sum(skipna=True) if "foreign_sell" in df.columns else np.nan

# % hari net buy
if "net_foreign" in df.columns:
    pct_hari_buy = (df["net_foreign"] > 0).sum() / max(1, df["net_foreign"].notna().sum()) * 100.0
else:
    pct_hari_buy = np.nan

# Max ratio (nilai/ADV)
if "nilai" in df.columns and "adv20" in df.columns:
    ratio_1y = (df["nilai"] / df["adv20"]).replace([np.inf, -np.inf], np.nan)
    max_ratio = ratio_1y.max(skipna=True)
else:
    max_ratio = np.nan

with k1:
    st.metric("Total Net Foreign", fmt_idr(total_net))
with k2:
    st.metric("Total Foreign Buy", fmt_idr(total_buy))
with k3:
    st.metric("Total Foreign Sell", fmt_idr(total_sell))
with k4:
    st.metric("% Hari Net Buy", f"{pct_hari_buy:.0f}%" if not pd.isna(pct_hari_buy) else "-")
with k5:
    st.metric("Max Ratio (nilai/ADV)", f"{max_ratio:.2f}x" if not pd.isna(max_ratio) else "-")

# hari net buy/sell terbesar
c1, c2 = st.columns(2)
with c1:
    if "net_foreign" in df.columns and "trade_date" in df.columns:
        idx = df["net_foreign"].idxmax()
        if idx is not None and not pd.isna(idx):
            v = df.loc[idx, "net_foreign"]
            t = df.loc[idx, "trade_date"]
            st.caption(f"🔎 Hari Net Buy terbesar: {pd.to_datetime(t).date()} ({fmt_idr(v)})")
with c2:
    if "net_foreign" in df.columns and "trade_date" in df.columns:
        idx = df["net_foreign"].idxmin()
        if idx is not None and not pd.isna(idx):
            v = df.loc[idx, "net_foreign"]
            t = df.loc[idx, "trade_date"]
            st.caption(f"🔎 Hari Net Sell terbesar: {pd.to_datetime(t).date()} ({fmt_idr(v)})")

st.markdown("---")

# ------------------------------ #
# Ringkasan Metrik Harga
# ------------------------------ #
st.subheader("Ringkasan Metrik")
m1, m2, m3 = st.columns(3)

# Ambil close price terbaik (urutan prioritas: penutupan, first_trade, open_price, sebelumnya)
h_close = None
for c in ["penutupan", "first_trade", "open_price", "sebelumnya"]:
    if c in df.columns and df[c].notna().any():
        h_close = c; break

# SMA 5D
if h_close:
    df["_sma5"] = df[h_close].rolling(5, min_periods=1).mean()
else:
    df["_sma5"] = np.nan

# Perubahan harga
if h_close:
    delta = df[h_close].iloc[-1] - df[h_close].iloc[0]
    pct = pct_str(df[h_close].iloc[-1], df[h_close].iloc[0])
else:
    delta = np.nan; pct = "-"

with m1:
    st.metric("Harga Terakhir (IDR)", fmt_idr(df[h_close].iloc[-1]) if h_close else "-")
    st.metric("Perubahan Harga", fmt_idr(delta) if not pd.isna(delta) else "-", pct)

with m2:
    hi = df["tertinggi"].max(skipna=True) if "tertinggi" in df.columns else np.nan
    lo = df["terendah"].min(skipna=True) if "terendah" in df.columns else np.nan
    st.metric("Tertinggi (Periode Filter)", fmt_idr(hi))
    st.metric("Terendah (Periode Filter)", fmt_idr(lo))

with m3:
    v = df["volume"].iloc[-1] if "volume" in df.columns and df["volume"].notna().any() else np.nan
    st.metric("Vol. Hari Terakhir (Jt Lbr)", f"{v/1e6:.2f}" if not pd.isna(v) else "-")
    st.metric("SMA 5 Hari (IDR)", fmt_idr(df["_sma5"].iloc[-1]) if df["_sma5"].notna().any() else "-")

st.markdown("---")

# ------------------------------ #
# Candlestick + MA5/MA20
# ------------------------------ #
st.subheader("Candlestick (OHLC) + MA5/MA20")
# Cari kolom OHLC
c_open  = "open_price" if "open_price" in df.columns else None
c_high  = "tertinggi"  if "tertinggi"  in df.columns else None
c_low   = "terendah"   if "terendah"   in df.columns else None
c_close = None
for cc in ["penutupan", "first_trade", "open_price", "sebelumnya"]:
    if cc in df.columns and df[cc].notna().any():
        c_close = cc; break

if all([c_open, c_high, c_low, c_close]):
    plot_df = df[["trade_date", c_open, c_high, c_low, c_close]].copy()
    plot_df.columns = ["trade_date", "open", "high", "low", "close"]
    plot_df["ma5"]  = plot_df["close"].rolling(5, min_periods=1).mean()
    plot_df["ma20"] = plot_df["close"].rolling(20, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=plot_df["trade_date"], open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"], close=plot_df["close"], name="OHLC"))
    fig.add_trace(go.Scatter(
        x=plot_df["trade_date"], y=plot_df["ma5"], name="MA5", line=dict(color="#2E86DE")))
    fig.add_trace(go.Scatter(
        x=plot_df["trade_date"], y=plot_df["ma20"], name="MA20", line=dict(color="#636E72")))
    fig.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Cara membaca & ringkasan chart (Candlestick)", expanded=False):
        st.markdown("""
- **Candlestick** memperlihatkan pergerakan harga (Open–High–Low–Close) setiap hari.
- **MA5/MA20** membantu melihat tren pendek vs menengah (potensi _cross_).
- Perhatikan area konsolidasi dan _breakout_ dibantu MA.
        """)
else:
    st.info("Kolom OHLC tidak lengkap, candlestick tidak dapat ditampilkan.")

st.markdown("---")

# ------------------------------ #
# Chart 1: Net Foreign Harian (Bar) + Close (Line)
# ------------------------------ #
st.subheader("Close vs Net Foreign Harian")
if h_close and "net_foreign" in df.columns:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df["trade_date"], y=df["net_foreign"], name="Net Foreign (Rp)",
        marker_color=np.where(df["net_foreign"]>=0, "#4CAF50", "#E74C3C")
    ))
    fig2.add_trace(go.Scatter(
        x=df["trade_date"], y=df[h_close], name="Close", yaxis="y2",
        line=dict(color="#16A085")
    ))
    fig2.update_layout(
        height=380, margin=dict(l=10,r=10,t=10,b=10),
        yaxis=dict(title="Rp"),
        yaxis2=dict(overlaying="y", side="right", title="Close")
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("ℹ️ Cara membaca & ringkasan chart (Close vs Net Foreign)", expanded=False):
        st.markdown("""
- **Bar hijau/merah** adalah _Net Foreign_ harian (hijau: **Net Buy**, merah: **Net Sell**).
- **Garis Close** menunjukkan penutupan harga harian.
- Kenaikan harga yang dibarengi Net Buy tinggi sering mengindikasikan dorongan akumulasi asing.
        """)
else:
    st.info("Data Net Foreign dan/atau Close tidak tersedia.")

st.markdown("---")

# ------------------------------ #
# Chart 2: Nilai vs ADV + Ratio (Nilai/ADV)
# ------------------------------ #
st.subheader("Nilai vs ADV + Ratio")
if "nilai" in df.columns:
    # window ADV yang dipilih
    if adv_mode == "1 Tahun":
        win = 252
    elif adv_mode == "6 Bulan":
        win = 126
    elif adv_mode == "3 Bulan":
        win = 63
    elif adv_mode == "1 Bulan":
        win = 21
    else:
        win = 252  # default 1 tahun

    df["_adv"] = df["nilai"].rolling(win, min_periods=1).mean()
    df["_ratio"] = (df["nilai"] / df["_adv"]).replace([np.inf, -np.inf], np.nan)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=df["trade_date"], y=df["nilai"], name="Nilai (Rp)", marker_color="#5DADE2"))
    fig3.add_trace(go.Scatter(x=df["trade_date"], y=df["_adv"], name=f"ADV ({win} hari)", line=dict(color="#2C3E50")))
    fig3.add_trace(go.Scatter(x=df["trade_date"], y=df["_ratio"], name="Ratio (Nilai/ADV)", yaxis="y2", line=dict(color="#1ABC9C")))
    fig3.update_layout(
        height=380, margin=dict(l=10,r=10,t=10,b=10),
        yaxis=dict(title="Rp"),
        yaxis2=dict(overlaying="y", side="right", title="Ratio (x)")
    )
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("ℹ️ Cara membaca & ringkasan chart (Nilai vs ADV + Ratio)", expanded=False):
        st.markdown(f"""
- **Nilai (bar)** adalah total nilai transaksi per hari.
- **ADV ({win} hari)** adalah rata-rata nilai transaksi {win} hari terakhir.
- **Ratio (Nilai/ADV)** di atas 1× berarti aktivitas hari itu **di atas rata-rata**; makin besar → makin tidak biasa.
        """)
else:
    st.info("Kolom **nilai** tidak ditemukan.")

st.markdown("---")

# ------------------------------ #
# Chart 3: Spread (bps)
# ------------------------------ #
if show_spread and "spread_bps" in df.columns:
    st.subheader("Spread (bps)")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df["trade_date"], y=df["spread_bps"], name="Spread (bps)"))
    # garis P75 untuk baseline
    p75 = np.nanpercentile(df["spread_bps"], 75) if df["spread_bps"].notna().any() else np.nan
    if not pd.isna(p75):
        fig4.add_hline(y=p75, line_dash="dot", line_color="#E74C3C", annotation_text=f"P75 ≈ {p75:.1f}", annotation_position="top left")
    fig4.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="bps")
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("ℹ️ Cara membaca & ringkasan chart (Spread)", expanded=False):
        st.markdown("""
- **Spread** (dalam bps) dihitung dari `(Offer − Bid) / ((Offer + Bid)/2) × 10.000)`.
- Spread tinggi menandakan friksi likuiditas lebih besar; risk/reward eksekusi order meningkat.
- Garis **P75** membantu mengidentifikasi kapan spread termasuk tinggi dibanding sejarah periode filter.
        """)

# ------------------------------ #
# Ekspor data
# ------------------------------ #
with st.expander("⬇️ Export data", expanded=False):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=f"{kode}_{start}_{end}.csv", mime="text/csv")
