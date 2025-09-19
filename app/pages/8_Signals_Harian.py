# -*- coding: utf-8 -*-
# app/pages/8_Signals_Harian.py
# Viewer untuk tabel signals_daily

import os
import io
import tempfile
from urllib.parse import quote_plus
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import plotly.graph_objects as go

st.set_page_config(page_title="🔔 Sinyal Harian (FF)", page_icon="🔔", layout="wide")
st.title("🔔 Sinyal Harian (Foreign Flow)")
st.caption("Sinyal berbasis FF_intensity p95 ± filter MA20. Sumber tabel: `signals_daily`.")

def _build_engine():
    host = os.getenv("DB_HOST", st.secrets.get("DB_HOST", ""))
    port = int(os.getenv("DB_PORT", st.secrets.get("DB_PORT", 3306)))
    database = os.getenv("DB_NAME", st.secrets.get("DB_NAME", ""))
    user = os.getenv("DB_USER", st.secrets.get("DB_USER", ""))
    password = os.getenv("DB_PASSWORD", st.secrets.get("DB_PASSWORD", ""))
    ssl_ca = os.getenv("DB_SSL_CA", st.secrets.get("DB_SSL_CA", ""))

    pwd = quote_plus(str(password))
    connect_args = {}
    if ssl_ca and "BEGIN CERTIFICATE" in ssl_ca:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
        tmp.write(ssl_ca.encode("utf-8")); tmp.flush()
        connect_args["ssl_ca"] = tmp.name

    url = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{database}"
    return create_engine(url, connect_args=connect_args, pool_recycle=300, pool_pre_ping=True)

engine = _build_engine()

# Load latest N days
N_DAYS = st.slider("Ambil data (hari terbaru)", 1, 30, 5, 1)
sig_types = st.multiselect("Filter Signal", ["FF_BUY","FF_SELL","NEUTRAL"], default=["FF_BUY","FF_SELL"])

# Sector mapping (optional)
sector_map = None
up = st.file_uploader("Opsional: Mapping sektor (CSV kolom: symbol, sector)", type=["csv"])
if up is not None:
    try:
        dfm = pd.read_csv(up)
        cols = [c.lower().strip() for c in dfm.columns]
        dfm.columns = cols
        if "symbol" in cols and "sector" in cols:
            sector_map = dfm.rename(columns={"symbol":"base_symbol"})
            sector_map["base_symbol"] = sector_map["base_symbol"].astype(str).str.upper()
        else:
            st.error("CSV harus punya kolom: symbol, sector")
    except Exception as e:
        st.error(f"Gagal baca CSV: {e}")

sql = f"""
    SELECT *
    FROM signals_daily
    WHERE trade_date >= CURDATE() - INTERVAL :n DAY
"""
with engine.connect() as con:
    df = pd.read_sql(text(sql), con, params={"n": int(N_DAYS)})

if df.empty:
    st.info("Belum ada data di `signals_daily`. Jalankan generator (GitHub Actions) lebih dulu.")
    st.stop()

df["trade_date"] = pd.to_datetime(df["trade_date"])
df = df.sort_values(["trade_date","base_symbol"])
df["base_symbol"] = df["base_symbol"].astype(str).str.upper()

if sector_map is not None:
    df = df.merge(sector_map, on="base_symbol", how="left")
else:
    df["sector"] = None

# Filter signal
if sig_types:
    df = df[df["signal"].isin(sig_types)]
if df.empty:
    st.warning("Tidak ada baris sesuai filter.")
    st.stop()

# Metrics ringkas
c1, c2, c3 = st.columns(3)
c1.metric("Total FF_BUY", int((df["signal"]=="FF_BUY").sum()))
c2.metric("Total FF_SELL", int((df["signal"]=="FF_SELL").sum()))
c3.metric("Total NEUTRAL", int((df["signal"]=="NEUTRAL").sum()))

# Distribusi per tanggal
dist = (df.groupby(["trade_date","signal"])["base_symbol"]
          .nunique().reset_index(name="count"))
pivot = dist.pivot(index="trade_date", columns="signal", values="count").fillna(0)
fig = go.Figure()
for sig in ["FF_BUY","FF_SELL","NEUTRAL"]:
    if sig in pivot.columns:
        fig.add_bar(x=pivot.index, y=pivot[sig], name=sig)
fig.update_layout(barmode="stack", height=380, margin=dict(l=40,r=40,t=40,b=40), hovermode="x unified",
                  title="Distribusi Sinyal per Hari (n emiten)")
st.plotly_chart(fig, use_container_width=True)

# Tabel utama
show_cols = ["trade_date","base_symbol","signal","ff_intensity","threshold_p95","foreign_net","adv20","close","ma20","reason","sector"]
st.dataframe(df[show_cols], use_container_width=True)

# Download CSV
csv = df[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", data=csv, file_name="signals_daily_latest.csv", mime="text/csv")