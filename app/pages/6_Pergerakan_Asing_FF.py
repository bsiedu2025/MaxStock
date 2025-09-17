\
import os
import re
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import mysql.connector
import altair as alt
import numpy as np

st.set_page_config(page_title="Pergerakan Asing (FF)", page_icon="🧭", layout="wide")
st.title("🧭 Analisa Pergerakan Asing (FF)")

st.caption("""
Interpretasi: baris **Ticker berakhiran ' FF'** pada `eod_prices_raw` merepresentasikan *foreign flow* (FF).
- **Volume > 0** → **BELI** asing (net buy)
- **Volume < 0** → **JUAL** asing (net sell)
Tips: gunakan rentang tanggal 30–90 hari untuk melihat tren akumulasi.
""")

# --- DB utils
@st.cache_resource(show_spinner=False)
def get_conn():
    host = os.getenv("DB_HOST", st.secrets.get("DB_HOST", ""))
    port = int(os.getenv("DB_PORT", st.secrets.get("DB_PORT", 3306)))
    database = os.getenv("DB_NAME", st.secrets.get("DB_NAME", ""))
    user = os.getenv("DB_USER", st.secrets.get("DB_USER", ""))
    password = os.getenv("DB_PASSWORD", st.secrets.get("DB_PASSWORD", ""))
    ssl_ca = os.getenv("DB_SSL_CA", st.secrets.get("DB_SSL_CA", ""))

    ssl_args = {}
    if ssl_ca and "BEGIN CERTIFICATE" in ssl_ca:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
        tmp.write(ssl_ca.encode("utf-8"))
        tmp.flush()
        ssl_args = {"ssl_ca": tmp.name, "ssl_disabled": False}
    else:
        ssl_args = {"ssl_disabled": True}

    conn = mysql.connector.connect(
        host=host, port=port, database=database,
        user=user, password=password, autocommit=True, **ssl_args
    )
    return conn

def load_ff(conn, start_d: date, end_d: date) -> pd.DataFrame:
    sql = """
    SELECT
      Ticker,
      Tanggal,
      Volume,
      `Close`
    FROM eod_prices_raw
    WHERE Ticker LIKE '% FF'
      AND Tanggal BETWEEN %s AND %s
    """
    df = pd.read_sql(sql, con=conn, params=(start_d, end_d))
    if df.empty:
        return df
    # Base ticker: remove trailing ' FF'
    df["BaseTicker"] = df["Ticker"].str.replace(r"\\s*FF$", "", regex=True).str.strip()
    df["FF_Net"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    df["Aksi"] = df["FF_Net"].apply(lambda v: "BELI" if v > 0 else ("JUAL" if v < 0 else "NETRAL"))
    return df

def load_price(conn, tickers, start_d: date, end_d: date) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["Ticker","Tanggal","Close"])
    # Simple IN clause; parameterize safely
    placeholders = ",".join(["%s"] * len(tickers))
    sql = f"""
    SELECT Ticker, Tanggal, `Close`
    FROM eod_prices_raw
    WHERE Ticker IN ({placeholders})
      AND Tanggal BETWEEN %s AND %s
    """
    params = list(tickers) + [start_d, end_d]
    pr = pd.read_sql(sql, con=conn, params=params)
    return pr

# --- Filters
today = date.today()
default_from = today - timedelta(days=60)
with st.sidebar:
    st.header("Filter")
    d1 = st.date_input("Dari tanggal", value=default_from)
    d2 = st.date_input("Sampai tanggal", value=today)
    min_abs = st.number_input("Min |FF| untuk ranking", min_value=0, value=1_000_000, step=100_000)
    include_ihsg = st.toggle("Tampilkan IHSG FF", value=True, help="Centang untuk menyertakan baris 'IHSG FF' di tabel & chart.")
    ma_window = st.slider("Smoothing (Rolling MA, hari)", min_value=1, max_value=20, value=1, help="Set >1 untuk menghaluskan FF cumulative & harga.")

conn = get_conn()
df = load_ff(conn, d1, d2)

if df.empty:
    st.warning("Tidak ada data FF pada rentang tanggal ini.")
    st.stop()

# Pisahkan IHSG bila perlu
if not include_ihsg:
    df = df[df["BaseTicker"].str.upper() != "IHSG"]

# --- Ringkasan Harian (Top Movers) + Download
st.subheader("Top Movers FF (per Tanggal)")
daily = (
    df.assign(AbsFF=df["FF_Net"].abs())
      .sort_values(["Tanggal","AbsFF"], ascending=[True, False])
)
topN = st.slider("Tampilkan Top N per hari", 3, 50, 10)
top_daily = daily.groupby("Tanggal").head(topN)
st.dataframe(top_daily[["Tanggal","BaseTicker","FF_Net","Aksi","Close"]], use_container_width=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("⬇️ Unduh Top Movers (CSV)",
                       data=top_daily.to_csv(index=False).encode("utf-8"),
                       file_name="ff_top_movers.csv",
                       mime="text/csv")
with c2:
    st.download_button("⬇️ Unduh Data FF Mentah (CSV)",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name="ff_raw.csv",
                       mime="text/csv")

# --- Agregat per Ticker (periode) + Download
st.subheader("Agregat Periode")
agg = (df.groupby("BaseTicker", as_index=False)
         .agg(FF_Net=("FF_Net","sum"),
              Hari=("FF_Net","size"),
              LastClose=("Close","last")))
agg["AbsFF"] = agg["FF_Net"].abs()
agg = agg.query("AbsFF >= @min_abs").sort_values("AbsFF", ascending=False)

colA, colB = st.columns(2)
with colA:
    st.markdown("**Top Net BUY**")
    buy_tbl = agg[agg["FF_Net"]>0][["BaseTicker","FF_Net","Hari","LastClose"]].head(30)
    st.dataframe(buy_tbl, use_container_width=True)
with colB:
    st.markdown("**Top Net SELL**")
    sell_tbl = agg[agg["FF_Net"]<0][["BaseTicker","FF_Net","Hari","LastClose"]].head(30)
    st.dataframe(sell_tbl, use_container_width=True)

st.download_button("⬇️ Unduh Agregat Periode (CSV)",
                   data=agg.to_csv(index=False).encode("utf-8"),
                   file_name="ff_aggregate.csv",
                   mime="text/csv")

# --- CHARTS: Cumulative FF per Ticker (+ MA)
st.subheader("📈 Akumulasi FF (Cumulative) per Ticker")

# Ambil kandidat ticker (Top by AbsFF)
candidates = agg["BaseTicker"].tolist()
default_top = candidates[:5] if candidates else []

selected = st.multiselect("Pilih ticker untuk plot (default: Top 5 AbsFF)",
                          options=candidates, default=default_top)

df_base = (
    df[["Tanggal","BaseTicker","FF_Net","Close"]]
      .sort_values(["BaseTicker","Tanggal"])
      .copy()
)

# Cumulative per ticker
df_base["FF_Cum"] = df_base.groupby("BaseTicker")["FF_Net"].cumsum()

# Optional smoothing (MA)
if ma_window and ma_window > 1:
    df_base["FF_Cum_S"] = df_base.groupby("BaseTicker")["FF_Cum"].transform(lambda s: s.rolling(ma_window, min_periods=1).mean())
    df_base["Close_S"] = df_base.groupby("BaseTicker")["Close"].transform(lambda s: s.rolling(ma_window, min_periods=1).mean())
else:
    df_base["FF_Cum_S"] = df_base["FF_Cum"]
    df_base["Close_S"] = df_base["Close"]

plot_pool = selected if selected else default_top
plot_df = df_base[df_base["BaseTicker"].isin(plot_pool)]

st.download_button("⬇️ Unduh Data Chart (CSV)",
                   data=plot_df.to_csv(index=False).encode("utf-8"),
                   file_name="ff_chart_data.csv",
                   mime="text/csv")

if plot_df.empty:
    st.info("Tidak ada data untuk ditampilkan pada chart akumulasi.")
else:
    line = (
        alt.Chart(plot_df, title=f"Cumulative FF by Ticker (MA={ma_window})")
        .mark_line()
        .encode(
            x=alt.X("Tanggal:T", title="Tanggal"),
            y=alt.Y("FF_Cum_S:Q", title="Cumulative FF"),
            color=alt.Color("BaseTicker:N", title="Ticker"),
            tooltip=[
                alt.Tooltip("Tanggal:T", title="Tanggal"),
                alt.Tooltip("BaseTicker:N", title="Ticker"),
                alt.Tooltip("FF_Cum_S:Q", title="Cum FF (S)", format=",d"),
                alt.Tooltip("FF_Net:Q", title="FF Harian", format=",d"),
            ],
        )
        .interactive()
    )
    st.altair_chart(line, use_container_width=True)

# --- Harga vs Akumulasi FF: Merge & Correlation
st.subheader("🔗 Korelasi Harga vs Akumulasi FF")

# Ambil harga non-FF untuk ticker terpilih
price_tickers = [t for t in plot_pool if t]  # nama base ticker persis sama di tabel non-FF
price_df = load_price(conn, price_tickers, d1, d2).rename(columns={"Ticker":"BaseTicker"})
price_df = price_df.sort_values(["BaseTicker","Tanggal"])

# Join dengan FF cumulative
merged = pd.merge(plot_df[["Tanggal","BaseTicker","FF_Net","FF_Cum_S","Close_S"]],
                  price_df[["BaseTicker","Tanggal","Close"]],
                  on=["BaseTicker","Tanggal"], how="left", suffixes=("_ff","_px"))

# Fallback: kalau Close non-FF kosong, gunakan Close dari baris FF
merged["Close_px"] = merged["Close_px"].fillna(merged["Close_S"])

# Normalisasi untuk chart gabungan (opsional)
def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return s * 0
    return (s - mu) / sd

merged["Z_FF_Cum"] = merged.groupby("BaseTicker")["FF_Cum_S"].transform(zscore)
merged["Z_Close"] = merged.groupby("BaseTicker")["Close_px"].transform(zscore)

# Chart layered: FF_Cum vs Close (y independent)
if not merged.empty:
    base = alt.Chart(merged).encode(x=alt.X("Tanggal:T", title="Tanggal"))
    ff_line = base.mark_line().encode(
        y=alt.Y("FF_Cum_S:Q", title="Cumulative FF"),
        color=alt.value("#1f77b4")
    )
    px_line = base.mark_line().encode(
        y=alt.Y("Close_px:Q", title="Harga Penutupan"),
        color=alt.value("#ff7f0e")
    )
    layered = alt.layer(ff_line, px_line, data=merged).resolve_scale(y='independent').properties(
        title=f"Harga vs Cumulative FF (MA={ma_window})"
    )
    st.altair_chart(layered, use_container_width=True)

    # Tabel korelasi
    rows = []
    for tkr, sub in merged.groupby("BaseTicker"):
        sub = sub.dropna(subset=["FF_Cum_S","Close_px"]).copy()
        if len(sub) >= 2:
            corr = np.corrcoef(sub["FF_Cum_S"], sub["Close_px"])[0,1]
        else:
            corr = np.nan
        rows.append({"Ticker": tkr, "Corr(FF_Cum, Close)": corr})
    corr_df = pd.DataFrame(rows).sort_values("Corr(FF_Cum, Close)", ascending=False)
    st.markdown("**Korelasi Pearson per Ticker** (mendekati +1 = searah, -1 = berlawanan)")
    st.dataframe(corr_df, use_container_width=True)
    st.download_button("⬇️ Unduh Korelasi (CSV)",
                       data=corr_df.to_csv(index=False).encode("utf-8"),
                       file_name="ff_price_correlation.csv",
                       mime="text/csv")
else:
    st.info("Belum ada data gabungan FF & harga untuk ditampilkan.")
