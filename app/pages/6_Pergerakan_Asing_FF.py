\
import os
import re
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import mysql.connector

st.set_page_config(page_title="Pergerakan Asing (FF)", page_icon="🧭", layout="wide")
st.title("🧭 Analisa Pergerakan Asing (FF)")

st.caption("""
Interpretasi: baris **Ticker berakhiran ' FF'** pada `eod_prices_raw` merepresentasikan *foreign flow* (FF).
- **Volume > 0** → **BELI** asing (net buy)
- **Volume < 0** → **JUAL** asing (net sell)
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
    df["BaseTicker"] = df["Ticker"].str.replace(r"\s*FF$", "", regex=True).str.strip()
    df["FF_Net"] = df["Volume"].astype("float")
    df["Aksi"] = df["FF_Net"].apply(lambda v: "BELI" if v > 0 else ("JUAL" if v < 0 else "NETRAL"))
    return df

# --- Filters
today = date.today()
default_from = today - timedelta(days=14)
col1, col2, col3 = st.columns([1,1,2])
with col1:
    d1 = st.date_input("Dari tanggal", value=default_from)
with col2:
    d2 = st.date_input("Sampai tanggal", value=today)
with col3:
    min_abs = st.number_input("Min |FF| untuk ranking", min_value=0, value=1_000_000, step=100_000)

conn = get_conn()
df = load_ff(conn, d1, d2)

if df.empty:
    st.warning("Tidak ada data FF pada rentang tanggal ini.")
    st.stop()

# --- Ringkasan Harian (Top Movers)
st.subheader("Top Movers FF (per Tanggal)")
daily = (
    df.assign(AbsFF=df["FF_Net"].abs())
      .sort_values(["Tanggal","AbsFF"], ascending=[True, False])
)
topN = st.slider("Tampilkan Top N per hari", 3, 50, 10)
top_daily = daily.groupby("Tanggal").head(topN)
st.dataframe(top_daily[["Tanggal","BaseTicker","FF_Net","Aksi","Close"]], use_container_width=True)

# --- Agregat per Ticker (periode)
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
    st.dataframe(agg[agg["FF_Net"]>0][["BaseTicker","FF_Net","Hari","LastClose"]].head(30), use_container_width=True)
with colB:
    st.markdown("**Top Net SELL**")
    st.dataframe(agg[agg["FF_Net"]<0][["BaseTicker","FF_Net","Hari","LastClose"]].head(30), use_container_width=True)

# --- Link ke halaman Harga Saham
st.subheader("Quick Links")
def _link(tkr: str) -> str:
    from urllib.parse import quote
    return f"[{tkr}](./Harga_Saham?ticker={quote(tkr)})"

st.write("Klik ticker untuk membuka halaman Harga Saham (jika halaman tersebut aktif di app).")
links_df = pd.DataFrame({
    "Ticker": agg["BaseTicker"],
    "Open Chart": agg["BaseTicker"].apply(_link)
}).head(100)
st.dataframe(links_df, hide_index=True, use_container_width=True)
