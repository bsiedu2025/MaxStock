# -*- coding: utf-8 -*-
"""
Streamlit Page — Historis Emas (USD/oz) & USD/IDR, dengan perbaikan menyeluruh:
- Toggle sumber emas: Stooq atau Yahoo (XAUUSD=X / GC=F)
- Kurs USD/IDR utama dari Google Sheets (gviz CSV), fallback otomatis ke Yahoo
- Simpan ke MySQL/MariaDB di dua tabel: gold_data & idr_data (PRIMARY KEY trade_date)
- Upload FULL (drop-create) dan GAP (append only)
- Join aman untuk MySQL (tanpa FULL JOIN), ffill untuk weekend/libur, agregasi D/W/M/Y
- Chart Plotly + metrik + ekspor CSV

Dependensi: streamlit, pandas, numpy, requests, SQLAlchemy, mysql-connector-python, plotly
"""

from __future__ import annotations
import os
import io
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine
import plotly.express as px
import plotly.graph_objects as go

# ────────────────────────────────────────────────────────────────────────────────
# Page setup
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="💰 Historis Emas & Rupiah", page_icon="🪙", layout="wide")
st.title("💰 Historis Emas & Nilai Tukar Rupiah — versi fixed")
st.caption("Sumber emas bisa dipilih (Stooq/Yahoo). Kurs dari Google Sheets atau fallback Yahoo. Data opsional disimpan ke MySQL/MariaDB.")

# ────────────────────────────────────────────────────────────────────────────────
# Const & helpers
# ────────────────────────────────────────────────────────────────────────────────
GRAM_PER_TROY_OZ = 31.1034768
YAHOO_CSV = (
    "https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={p1}&period2={p2}&interval=1d&events=history&includeAdjustedClose=true"
)
STOOQ_XAUUSD = "https://stooq.com/q/d/l/?s=xauusd&i=d"

@st.cache_data(show_spinner=False)
def _unix_range(start: date, end: date) -> Tuple[int, int]:
    epoch = date(1970, 1, 1)
    p1 = int((start - epoch).days * 86400)
    p2 = int(((end + timedelta(days=1)) - epoch).days * 86400)  # period2 eksklusif → +1 hari
    return p1, p2

# ────────────────────────────────────────────────────────────────────────────────
# DB config
# ────────────────────────────────────────────────────────────────────────────────

def _read_db_cfg() -> Optional[dict]:
    try:
        if "db" in st.secrets:
            s = st.secrets["db"]
            return {
                "host": s.get("host"),
                "port": int(s.get("port", 3306)),
                "database": s.get("name") or s.get("database"),
                "user": s.get("user"),
                "password": s.get("password"),
                "ssl_ca": s.get("ssl_ca"),
            }
    except Exception:
        pass
    return {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "ssl_ca": os.getenv("DB_SSL_CA"),
    }

@st.cache_resource(show_spinner=False)
def get_engine() -> Optional[Engine]:
    cfg = _read_db_cfg()
    if not cfg or not cfg.get("host") or not cfg.get("database"):
        return None

    connect_args = {}
    if cfg.get("ssl_ca"):
        # Terima path atau isi PEM langsung
        if os.path.exists(str(cfg["ssl_ca"])):
            connect_args["ssl_ca"] = cfg["ssl_ca"]
        else:
            pem_path = os.path.join(os.getcwd(), "_mysql_ca.pem")
            try:
                with open(pem_path, "w", encoding="utf-8") as f:
                    f.write(str(cfg["ssl_ca"]))
                connect_args["ssl_ca"] = pem_path
            except Exception:
                pass

    url = f"mysql+mysqlconnector://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True, pool_recycle=300)

# ────────────────────────────────────────────────────────────────────────────────
# Fetchers (emas & kurs)
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_gold_stooq() -> pd.DataFrame:
    df = pd.read_csv(STOOQ_XAUUSD)
    df = df.rename(columns={"Date": "trade_date", "Close": "Gold_USD"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df["Gold_USD"] = pd.to_numeric(df["Gold_USD"], errors="coerce")
    return df.dropna(subset=["trade_date", "Gold_USD"]).sort_values("trade_date").reset_index(drop=True)

@st.cache_data(ttl=3600)
def fetch_gold_yahoo(symbol: str, start: date, end: date) -> pd.DataFrame:
    p1, p2 = _unix_range(start, end)
    url = YAHOO_CSV.format(symbol=symbol, p1=p1, p2=p2)
    df = pd.read_csv(url)
    df = df.rename(columns={"Date": "trade_date", "Close": "Gold_USD"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df["Gold_USD"] = pd.to_numeric(df["Gold_USD"], errors="coerce")
    return df.dropna(subset=["trade_date", "Gold_USD"]).sort_values("trade_date").reset_index(drop=True)

@st.cache_data(ttl=1800)
def fetch_idr_sheets(sheet_id: str, gid: int = 0) -> pd.DataFrame:
    if not sheet_id:
        return pd.DataFrame(columns=["trade_date", "IDR_USD"])
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
    txt = requests.get(url, timeout=30).text
    df = pd.read_csv(io.StringIO(txt))
    if df.shape[1] < 2:
        return pd.DataFrame(columns=["trade_date", "IDR_USD"])
    df = df.iloc[:, :2]
    df.columns = ["trade_date", "IDR_USD"]
    df = df[~df["trade_date"].astype(str).str.contains("Date", case=False, na=False)]
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df["IDR_USD"] = pd.to_numeric(df["IDR_USD"].astype(str).str.replace(",", "").str.replace(" ", ""), errors="coerce")
    return df.dropna(subset=["trade_date", "IDR_USD"]).sort_values("trade_date").reset_index(drop=True)

@st.cache_data(ttl=3600)
def fetch_idr_yahoo(start: date, end: date) -> pd.DataFrame:
    p1, p2 = _unix_range(start, end)
    url = YAHOO_CSV.format(symbol="USDIDR=X", p1=p1, p2=p2)
    df = pd.read_csv(url)
    df = df.rename(columns={"Date": "trade_date", "Close": "IDR_USD"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df["IDR_USD"] = pd.to_numeric(df["IDR_USD"], errors="coerce")
    return df.dropna(subset=["trade_date", "IDR_USD"]).sort_values("trade_date").reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────────────────
# DB schema & uploaders
# ────────────────────────────────────────────────────────────────────────────────
DDL_GOLD = """
CREATE TABLE IF NOT EXISTS gold_data (
  trade_date DATE PRIMARY KEY,
  Gold_USD   DOUBLE
) ENGINE=InnoDB
"""

DDL_IDR = """
CREATE TABLE IF NOT EXISTS idr_data (
  trade_date DATE PRIMARY KEY,
  IDR_USD    DOUBLE
) ENGINE=InnoDB
"""

@st.cache_data(ttl=600)
def table_exists(engine: Engine, name: str) -> bool:
    try:
        with engine.connect() as con:
            con.execute(sql_text(f"SELECT COUNT(*) FROM {name} LIMIT 1"))
        return True
    except Exception:
        return False

@st.cache_data(ttl=60)
def max_trade_date(engine: Engine, name: str) -> Optional[date]:
    try:
        with engine.connect() as con:
            d = con.execute(sql_text(f"SELECT MAX(trade_date) FROM {name}")).scalar()
            if isinstance(d, datetime):
                return d.date()
            return d
    except Exception:
        return None


def recreate_tables(engine: Engine):
    with engine.begin() as con:
        con.execute(sql_text("DROP TABLE IF EXISTS gold_data"))
        con.execute(sql_text("DROP TABLE IF EXISTS idr_data"))
        con.execute(sql_text(DDL_GOLD))
        con.execute(sql_text(DDL_IDR))


def ensure_tables(engine: Engine):
    with engine.begin() as con:
        con.execute(sql_text(DDL_GOLD))
        con.execute(sql_text(DDL_IDR))


def upload_full(engine: Optional[Engine], gold_df: pd.DataFrame, idr_df: pd.DataFrame):
    if engine is None:
        st.warning("DB belum dikonfigurasi → skip upload ke MySQL (in-memory saja).")
        return
    with st.status("Membuat ulang tabel & upload penuh…", expanded=True) as s:
        recreate_tables(engine)
        gold_df.to_sql("gold_data", engine, if_exists="append", index=False)
        idr_df.to_sql("idr_data", engine, if_exists="append", index=False)
        s.update(label=f"Selesai. gold_data={len(gold_df):,}, idr_data={len(idr_df):,}", state="complete")


def upload_gap(engine: Optional[Engine], gold_df: pd.DataFrame, idr_df: pd.DataFrame):
    if engine is None:
        st.warning("DB belum dikonfigurasi → skip upload gap.")
        return
    ensure_tables(engine)
    with st.status("Mengunggah GAP data…", expanded=True) as s:
        dmax_g = max_trade_date(engine, "gold_data")
        dmax_i = max_trade_date(engine, "idr_data")
        g_gap = gold_df if dmax_g is None else gold_df[gold_df["trade_date"] > dmax_g]
        i_gap = idr_df if dmax_i is None else idr_df[idr_df["trade_date"] > dmax_i]
        if not g_gap.empty:
            g_gap.to_sql("gold_data", engine, if_exists="append", index=False)
            st.write(f"gold_data +{len(g_gap):,}")
        if not i_gap.empty:
            i_gap.to_sql("idr_data", engine, if_exists="append", index=False)
            st.write(f"idr_data +{len(i_gap):,}")
        if g_gap.empty and i_gap.empty:
            st.write("Tidak ada gap baru.")
        s.update(label="GAP upload selesai", state="complete")

# ────────────────────────────────────────────────────────────────────────────────
# Reader (join aman untuk MySQL)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def read_joined(engine: Optional[Engine], start: date, end: date) -> pd.DataFrame:
    if engine is None:
        return pd.DataFrame()

    # Emulasi FULL OUTER JOIN dengan UNION di MySQL
    q = sql_text(
        """
        SELECT t.trade_date, t.Gold_USD, t.IDR_USD FROM (
          SELECT g.trade_date, g.Gold_USD, i.IDR_USD
          FROM gold_data g LEFT JOIN idr_data i ON i.trade_date = g.trade_date
          WHERE g.trade_date BETWEEN :s AND :e
          UNION
          SELECT i.trade_date, g.Gold_USD, i.IDR_USD
          FROM idr_data i LEFT JOIN gold_data g ON g.trade_date = i.trade_date
          WHERE i.trade_date BETWEEN :s AND :e
        ) t
        ORDER BY t.trade_date
        """
    )
    with engine.connect() as con:
        df = pd.read_sql(q, con, params={"s": start, "e": end})

    # Build daily index & ffill agar mulus (tidak bolong saat weekend)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    idx = pd.date_range(start, end, freq="D")
    df = df.set_index(pd.to_datetime(df["trade_date"]))
    df = df.reindex(idx).sort_index()
    df["Gold_USD"] = df["Gold_USD"].ffill()
    df["IDR_USD"] = df["IDR_USD"].ffill()
    df["trade_date"] = df.index.date
    return df.reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan Data")
    gold_source = st.radio("Sumber Emas", ["Stooq", "Yahoo"], index=0)
    yahoo_symbol = st.selectbox("Symbol Yahoo", ["XAUUSD=X", "GC=F"], index=0)

    st.subheader("Sumber Kurs USD/IDR")
    sheet_id = st.text_input("Spreadsheet ID (Sheets)", value=os.getenv("FX_SHEET_ID", ""), help="Share: Anyone with link. Kolom A=Date, B=Rate")
    gid = st.number_input("GID", min_value=0, value=int(os.getenv("FX_SHEET_GID", "0")))

    st.subheader("Rentang & Agregasi")
    today = date.today()
    start_date = st.date_input("Mulai", value=today - timedelta(days=365*5))
    end_date = st.date_input("Selesai", value=today)
    freq_label = st.selectbox("Agregasi", ["Harian", "Mingguan", "Bulanan", "Tahunan"], index=0)
    freq_map = {"Harian": "D", "Mingguan": "W", "Bulanan": "M", "Tahunan": "Y"}
    freq = freq_map[freq_label]

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        btn_full = st.button("📥 Full Replace ke DB")
    with c2:
        btn_gap = st.button("⚡ Upload GAP")

# ────────────────────────────────────────────────────────────────────────────────
# Ambil data sumber untuk tampilan (in-memory)
# ────────────────────────────────────────────────────────────────────────────────
try:
    if gold_source == "Stooq":
        gold_src = fetch_gold_stooq()
        gold_show = gold_src[(gold_src["trade_date"] >= start_date) & (gold_src["trade_date"] <= end_date)]
    else:
        gold_src = fetch_gold_yahoo(yahoo_symbol, start_date, end_date)
        gold_show = gold_src.copy()
except Exception as e:
    st.error(f"Gagal mengambil emas dari {gold_source}: {e}")
    gold_src = pd.DataFrame(columns=["trade_date", "Gold_USD"]) ; gold_show = gold_src

# Kurs: Sheets utamanya, fallback Yahoo kalau gagal/kosong
try:
    idr_src = fetch_idr_sheets(sheet_id, gid)
    idr_show = idr_src[(idr_src["trade_date"] >= start_date) & (idr_src["trade_date"] <= end_date)] if not idr_src.empty else idr_src
    if idr_show.empty:
        raise RuntimeError("Sheets kosong / gagal")
except Exception:
    st.warning("Sheets gagal/ kosong → fallback Yahoo…")
    idr_src = fetch_idr_yahoo(start_date, end_date)
    idr_show = idr_src.copy()

# ────────────────────────────────────────────────────────────────────────────────
# Upload ke DB sesuai tombol
# ────────────────────────────────────────────────────────────────────────────────
engine = get_engine()
if btn_full:
    upload_full(engine, gold_src, idr_src)
if btn_gap:
    upload_gap(engine, gold_src, idr_src)

# ────────────────────────────────────────────────────────────────────────────────
# Gabung untuk tampilan (pakai DB kalau ada; jika tidak, in-memory)
# ────────────────────────────────────────────────────────────────────────────────
if engine is not None and table_exists(engine, "gold_data") and table_exists(engine, "idr_data"):
    try:
        merged = read_joined(engine, start_date, end_date)
    except Exception as e:
        st.warning(f"Baca dari DB gagal ({e}) → pakai in-memory.")
        merged = pd.merge(gold_show, idr_show, on="trade_date", how="outer").sort_values("trade_date")
else:
    merged = pd.merge(gold_show, idr_show, on="trade_date", how="outer").sort_values("trade_date")

# Ffill harian untuk tampilan
if not merged.empty:
    idx = pd.date_range(start_date, end_date, freq="D")
    merged = merged.set_index(pd.to_datetime(merged["trade_date"]))
    merged = merged.reindex(idx).sort_index()
    merged["Gold_USD"], merged["IDR_USD"] = merged["Gold_USD"].ffill(), merged["IDR_USD"].ffill()
    merged["trade_date"] = merged.index.date
    merged = merged.reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────────────────
# Tabel, metrik, chart, ekspor
# ────────────────────────────────────────────────────────────────────────────────
if merged.empty:
    st.warning("Data gabungan kosong. Cek sumber/ tanggal.")
    st.stop()

merged["Gold_IDR_per_gram"] = merged["Gold_USD"] * merged["IDR_USD"] / GRAM_PER_TROY_OZ
st.subheader("📄 Data Gabungan (Harian)")
st.dataframe(merged.tail(30), use_container_width=True)

# Metrik
last = merged.dropna().iloc[-1]
prev = merged.dropna().iloc[-2] if len(merged.dropna()) >= 2 else last
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Gold USD/oz (terkini)", f"${last['Gold_USD']:.2f}", delta=f"{(last['Gold_USD']-prev['Gold_USD']):+.2f}")
with m2:
    st.metric("USD/IDR (terkini)", f"Rp{last['IDR_USD']:,.0f}", delta=f"{(last['IDR_USD']-prev['IDR_USD']):+,.0f}")
with m3:
    st.metric("Emas IDR/gram (terkini)", f"Rp{last['Gold_IDR_per_gram']:,.0f}")

# Agregasi
def _aggregate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    dt = pd.to_datetime(df["trade_date"]) ; g = pd.Series(df["Gold_USD"].values, index=dt) ; r = pd.Series(df["IDR_USD"].values, index=dt)
    g, r = g.resample(freq).last(), r.resample(freq).last()
    out = pd.DataFrame({"trade_date": g.index, "Gold_USD": g.values, "IDR_USD": r.values})
    out["Gold_IDR_per_gram"] = out["Gold_USD"] * out["IDR_USD"] / GRAM_PER_TROY_OZ
    out["Gold_%"] = out["Gold_USD"].pct_change() * 100
    out["IDR_%"] = out["IDR_USD"].pct_change() * 100
    return out.dropna(how="all")

freq = freq_map[freq_label]
agg = _aggregate(merged, freq)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.line(merged, x="trade_date", y="Gold_USD", title="Gold USD/oz (Harian)"), use_container_width=True)
with c2:
    st.plotly_chart(px.line(merged, x="trade_date", y="IDR_USD", title="USD/IDR (Harian)"), use_container_width=True)

st.subheader(f"📊 Agregasi {freq_label}")
st.dataframe(agg.tail(12), use_container_width=True)

fig = go.Figure()
fig.add_bar(x=agg["trade_date"], y=agg["Gold_%"], name="Emas %")
fig.add_trace(go.Scatter(x=agg["trade_date"], y=agg["IDR_%"], name="Rupiah %", mode="lines"))
fig.update_layout(title="Perubahan Persentase (Agregasi)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Ekspor
st.download_button("⬇️ Unduh Gabungan (Harian) CSV", data=merged.to_csv(index=False).encode(), file_name="gold_idr_daily.csv", mime="text/csv")
st.download_button("⬇️ Unduh Agregasi CSV", data=agg.to_csv(index=False).encode(), file_name=f"gold_idr_{freq_label.lower()}.csv", mime="text/csv")

st.caption("Sumber emas: Stooq atau Yahoo (XAUUSD=X/GC=F). Kurs: Google Sheets atau fallback Yahoo. Tabel: gold_data & idr_data.")
