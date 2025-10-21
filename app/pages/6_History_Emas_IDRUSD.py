# -*- coding: utf-8 -*-
"""
Streamlit page: History Emas (USD/oz) & Kurs USD/IDR
- Sumber emas bisa dipilih: Stooq atau Yahoo Finance (toggle di sidebar)
- Kurs USD/IDR diambil dari Google Sheets (CSV via gviz) ATAU bisa juga Yahoo (opsi cadangan)
- Simpan ke MySQL: dua tabel (gold_data, idr_data)
- Join, forward-fill di akhir pekan/libur, agregasi (D/W/M/Y), dan chart Plotly

Catatan:
- App tetap bisa jalan tanpa DB (akan tampilkan data in-memory) kalau kredensial DB tidak tersedia.
- Pastikan pip deps: streamlit, pandas, numpy, SQLAlchemy, mysql-connector-python, plotly, requests, python-dateutil
"""

from __future__ import annotations
import os
import io
import sys
import math
import json
import time
import textwrap
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine
import plotly.express as px

# ------------------------------
# Page Setup
# ------------------------------
st.set_page_config(
    page_title="History Emas & USD/IDR",
    page_icon="🪙",
    layout="wide",
)

st.title("🪙 History Emas & USD/IDR — dengan Toggle Sumber (Stooq/Yahoo)")
st.caption(
    "Sumber emas bisa dipilih di sidebar. Data disimpan di MySQL (jika tersedia), lalu digabung dan divisualisasikan."
)

# ------------------------------
# Helpers
# ------------------------------
YAHOO_GOLD_SYMBOL_SPOT = "XAUUSD=X"  # spot gold USD/oz
YAHOO_GOLD_SYMBOL_FUT = "GC=F"       # alternative futures continuous
YAHOO_CSV_TEMPLATE = (
    "https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={p1}"
    "&period2={p2}&interval=1d&events=history&includeAdjustedClose=true"
)
STOOQ_GOLD_XAUUSD_CSV = "https://stooq.com/q/d/l/?s=xauusd&i=d"
GRAM_PER_TROY_OZ = 31.1034768

@st.cache_data(show_spinner=False)
def _unix_range(start: date, end: date) -> Tuple[int, int]:
    """Return (period1, period2) as seconds since epoch; period2 exclusive (add 1 day)."""
    epoch = date(1970, 1, 1)
    p1 = int((start - epoch).total_seconds())
    p2 = int(((end + timedelta(days=1)) - epoch).total_seconds())
    return p1, p2

# ------------------------------
# Database
# ------------------------------

def _read_db_config() -> Optional[dict]:
    # Prefer st.secrets then env vars
    cfg = {}
    try:
        if "db" in st.secrets:
            s = st.secrets["db"]
            cfg = {
                "host": s.get("host"),
                "port": int(s.get("port", 3306)),
                "database": s.get("name") or s.get("database"),
                "user": s.get("user"),
                "password": s.get("password"),
                "ssl_ca": s.get("ssl_ca"),
            }
        else:
            # env fallback
            cfg = {
                "host": os.getenv("DB_HOST"),
                "port": int(os.getenv("DB_PORT", "3306")),
                "database": os.getenv("DB_NAME"),
                "user": os.getenv("DB_USER"),
                "password": os.getenv("DB_PASSWORD"),
                "ssl_ca": os.getenv("DB_SSL_CA"),
            }
    except Exception:
        pass

    if not cfg or not cfg.get("host") or not cfg.get("database"):
        return None
    return cfg

@st.cache_resource(show_spinner=False)
def get_engine() -> Optional[Engine]:
    cfg = _read_db_config()
    if not cfg:
        return None

    # Optional SSL CA file handling
    connect_args = {}
    if cfg.get("ssl_ca"):
        # Write CA to temp file for mysql-connector-python
        ca_path = os.path.join(os.getcwd(), "_mysql_ca.pem")
        try:
            with open(ca_path, "w", encoding="utf-8") as f:
                f.write(cfg["ssl_ca"])  # assume PEM content
            connect_args["ssl_ca"] = ca_path
        except Exception:
            # Accept path if it's a path, not content
            if os.path.exists(cfg["ssl_ca"]):
                connect_args["ssl_ca"] = cfg["ssl_ca"]

    url = (
        f"mysql+mysqlconnector://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
    return engine

# ------------------------------
# Fetchers
# ------------------------------

@st.cache_data(ttl=3600, show_spinner=True)
def fetch_gold_from_stooq() -> pd.DataFrame:
    df = pd.read_csv(STOOQ_GOLD_XAUUSD_CSV)
    # Expected columns: Date,Open,High,Low,Close,Volume
    df = df.rename(columns={"Date": "trade_date", "Close": "Gold_USD"})
    # Coerce types
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df["Gold_USD"] = pd.to_numeric(df["Gold_USD"], errors="coerce")
    df = df.dropna(subset=["trade_date", "Gold_USD"]).sort_values("trade_date").reset_index(drop=True)
    return df

@st.cache_data(ttl=3600, show_spinner=True)
def fetch_gold_from_yahoo(symbol: str = YAHOO_GOLD_SYMBOL_SPOT,
                          start: Optional[date] = None,
                          end: Optional[date] = None) -> pd.DataFrame:
    if start is None:
        start = date(1970, 1, 1)
    if end is None:
        end = date.today()
    p1, p2 = _unix_range(start, end)
    url = YAHOO_CSV_TEMPLATE.format(symbol=symbol, p1=p1, p2=p2)
    df = pd.read_csv(url)
    # Columns: Date,Open,High,Low,Close,Adj Close,Volume
    df = df.rename(columns={"Date": "trade_date", "Close": "Gold_USD"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df["Gold_USD"] = pd.to_numeric(df["Gold_USD"], errors="coerce")
    df = df.dropna(subset=["trade_date", "Gold_USD"]).sort_values("trade_date").reset_index(drop=True)
    return df

@st.cache_data(ttl=1800, show_spinner=True)
def fetch_idr_from_sheets(spreadsheet_id: str, gid: int = 0) -> pd.DataFrame:
    """Ambil kurs USD/IDR dari Google Sheets (tab pertama/gid=0) via CSV gviz.
    Wajib: kolom pertama = tanggal, kolom kedua = nilai kurs (IDR per USD).
    Sheet harus public Viewer.
    """
    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
    df_raw = pd.read_csv(url)

    # Cari dua kolom pertama yang valid
    # Deteksi kolom tanggal & nilai by position, lalu coerce
    df = df_raw.copy()
    if df.shape[1] < 2:
        raise ValueError("CSV dari Google Sheets minimal harus punya 2 kolom (tanggal; nilai kurs)")

    df = df.iloc[:, :2]
    df.columns = ["trade_date", "IDR_USD"]
    # Buang header duplikat seperti baris 'Date' dsb.
    df = df[~df["trade_date"].astype(str).str.contains("Date", case=False, na=False)]

    # Coerce
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    # Hapus pemisah ribuan koma/titik lalu konversi
    df["IDR_USD"] = (
        df["IDR_USD"].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    )
    df["IDR_USD"] = pd.to_numeric(df["IDR_USD"], errors="coerce")

    df = df.dropna(subset=["trade_date", "IDR_USD"]).sort_values("trade_date").reset_index(drop=True)
    return df

# Cadangan: USD/IDR dari Yahoo jika butuh penuh otomatis
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_idr_from_yahoo(start: Optional[date] = None, end: Optional[date] = None) -> pd.DataFrame:
    if start is None:
        start = date(1970, 1, 1)
    if end is None:
        end = date.today()
    p1, p2 = _unix_range(start, end)
    url = YAHOO_CSV_TEMPLATE.format(symbol="USDIDR=X", p1=p1, p2=p2)
    df = pd.read_csv(url)
    df = df.rename(columns={"Date": "trade_date", "Close": "IDR_USD"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df["IDR_USD"] = pd.to_numeric(df["IDR_USD"], errors="coerce")
    df = df.dropna(subset=["trade_date", "IDR_USD"]).sort_values("trade_date").reset_index(drop=True)
    return df

# ------------------------------
# DB schema helpers
# ------------------------------
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


def recreate_tables(engine: Engine):
    with engine.begin() as conn:
        conn.execute(sql_text("DROP TABLE IF EXISTS gold_data"))
        conn.execute(sql_text("DROP TABLE IF EXISTS idr_data"))
        conn.execute(sql_text(DDL_GOLD))
        conn.execute(sql_text(DDL_IDR))


def ensure_tables(engine: Engine):
    with engine.begin() as conn:
        conn.execute(sql_text(DDL_GOLD))
        conn.execute(sql_text(DDL_IDR))


# ------------------------------
# Uploaders (full & gap)
# ------------------------------

def upload_full(engine: Optional[Engine], gold_df: pd.DataFrame, idr_df: pd.DataFrame):
    if engine is None:
        st.warning("DB tidak terkonfigurasi. Melewatkan penyimpanan ke MySQL (in-memory saja).")
        return
    with st.status("Mempersiapkan tabel & mengunggah (full replace)…", expanded=True) as status:
        recreate_tables(engine)
        st.write("✔ Tabel dibuat ulang")
        gold_df.to_sql("gold_data", con=engine, if_exists="append", index=False)
        idr_df.to_sql("idr_data", con=engine, if_exists="append", index=False)
        st.write(f"✔ Upload gold_data: {len(gold_df):,} baris")
        st.write(f"✔ Upload idr_data : {len(idr_df):,} baris")
        status.update(label="Selesai full replace ✅", state="complete")


def _max_date(engine: Engine, table: str) -> Optional[date]:
    q = f"SELECT MAX(trade_date) AS d FROM {table}"
    with engine.connect() as conn:
        r = conn.execute(sql_text(q)).scalar()
        if r is None:
            return None
        if isinstance(r, datetime):
            return r.date()
        return r


def upload_gap(engine: Optional[Engine], gold_df: pd.DataFrame, idr_df: pd.DataFrame):
    if engine is None:
        st.warning("DB tidak terkonfigurasi. Melewatkan penyimpanan gap ke MySQL (in-memory saja).")
        return

    ensure_tables(engine)

    with st.status("Mengunggah gap data…", expanded=True) as status:
        # GOLD
        dmax_gold = _max_date(engine, "gold_data")
        if dmax_gold is None:
            st.write("gold_data kosong → mengunggah penuh…")
            gold_to_up = gold_df
        else:
            gold_to_up = gold_df[gold_df["trade_date"] > dmax_gold]
        if not gold_to_up.empty:
            gold_to_up.to_sql("gold_data", con=engine, if_exists="append", index=False)
            st.write(f"✔ gold_data +{len(gold_to_up):,} baris")
        else:
            st.write("gold_data tidak ada gap")

        # IDR
        dmax_idr = _max_date(engine, "idr_data")
        if dmax_idr is None:
            st.write("idr_data kosong → mengunggah penuh…")
            idr_to_up = idr_df
        else:
            idr_to_up = idr_df[idr_df["trade_date"] > dmax_idr]
        if not idr_to_up.empty:
            idr_to_up.to_sql("idr_data", con=engine, if_exists="append", index=False)
            st.write(f"✔ idr_data +{len(idr_to_up):,} baris")
        else:
            st.write("idr_data tidak ada gap")

        status.update(label="Selesai unggah gap ✅", state="complete")


# ------------------------------
# Query merged & aggregate
# ------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def read_joined(engine: Optional[Engine], start: date, end: date) -> pd.DataFrame:
    if engine is None:
        return pd.DataFrame()
    q = sql_text(
        """
        SELECT g.trade_date,
               g.Gold_USD,
               i.IDR_USD
        FROM gold_data g
        FULL JOIN idr_data i ON i.trade_date = g.trade_date
        WHERE (g.trade_date BETWEEN :s AND :e) OR (i.trade_date BETWEEN :s AND :e)
        ORDER BY 1
        """
    )
    # Note: FULL JOIN not supported by MySQL; use UNION to emulate
    if engine.dialect.name.lower().startswith("mysql"):
        q = sql_text(
            """
            SELECT t.trade_date, t.Gold_USD, t.IDR_USD FROM (
              SELECT g.trade_date, g.Gold_USD, i.IDR_USD
              FROM gold_data g
              LEFT JOIN idr_data i ON i.trade_date = g.trade_date
              WHERE g.trade_date BETWEEN :s AND :e
              UNION
              SELECT i.trade_date, g.Gold_USD, i.IDR_USD
              FROM idr_data i
              LEFT JOIN gold_data g ON g.trade_date = i.trade_date
              WHERE i.trade_date BETWEEN :s AND :e
            ) t
            ORDER BY t.trade_date
            """
        )

    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"s": start, "e": end})
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    # Forward-fill to cover weekend/libur
    df = df.set_index(pd.to_datetime(df["trade_date"])).sort_index()
    idx = pd.date_range(start, end, freq="D")
    df = df.reindex(idx)
    df["Gold_USD"] = df["Gold_USD"].ffill()
    df["IDR_USD"] = df["IDR_USD"].ffill()
    df["trade_date"] = df.index.date
    df = df[["trade_date", "Gold_USD", "IDR_USD"]].reset_index(drop=True)
    return df


def aggregate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty:
        return df
    s = pd.to_datetime(df["trade_date"])  # index for resample
    g = pd.Series(df["Gold_USD"].values, index=s)
    r = pd.Series(df["IDR_USD"].values, index=s)

    # Resample ke last value tiap periode
    g_res = g.resample(freq).last()
    r_res = r.resample(freq).last()

    out = pd.DataFrame({
        "trade_date": g_res.index.date,
        "Gold_USD": g_res.values,
        "IDR_USD": r_res.values,
    })
    out["Gold_IDR_per_gram"] = out["Gold_USD"] * out["IDR_USD"] / GRAM_PER_TROY_OZ

    # %Change terhadap periode sebelumnya
    out["Gold_USD_%chg"] = out["Gold_USD"].pct_change() * 100.0
    out["IDR_USD_%chg"] = out["IDR_USD"].pct_change() * 100.0
    return out.dropna(how="all")

# ------------------------------
# Sidebar Controls
# ------------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan Data")

    gold_source = st.radio(
        "Sumber Emas (USD/oz)",
        options=["Stooq", "Yahoo Finance"],
        index=0,
        help="Pilih sumber data emas. Jika salah satu gagal, gunakan yang lain."
    )

    yahoo_symbol = st.selectbox(
        "Symbol Yahoo (jika Yahoo dipilih)",
        options=[YAHOO_GOLD_SYMBOL_SPOT, YAHOO_GOLD_SYMBOL_FUT],
        index=0,
        help="XAUUSD=X = spot; GC=F = futures continuous"
    )

    st.markdown("---")
    st.subheader("Kurs USD/IDR dari Google Sheets")
    sheet_id = st.text_input(
        "Spreadsheet ID",
        value=os.getenv("FX_SHEET_ID", ""),
        placeholder="1x_your_google_sheet_id_here",
        help="Ambil bagian di URL di antara /d/ dan /edit. Share: Anyone with link (Viewer). Kolom1=tanggal, Kolom2=kurs."
    )
    gid = st.number_input("GID tab (opsional)", min_value=0, value=int(os.getenv("FX_SHEET_GID", "0")))

    st.markdown("---")
    st.subheader("Rentang Tanggal & Agregasi")
    today = date.today()
    default_start = today - timedelta(days=365)
    start_date = st.date_input("Mulai", value=default_start)
    end_date = st.date_input("Selesai", value=today)

    freq_label = st.selectbox("Agregasi", ["Harian", "Mingguan", "Bulanan", "Tahunan"], index=0)
    freq_map = {"Harian": "D", "Mingguan": "W", "Bulanan": "M", "Tahunan": "Y"}
    freq = freq_map[freq_label]

    st.markdown("---")
    colb1, colb2 = st.columns(2)
    with colb1:
        do_full = st.button("📥 Ambil & Buat Tabel (Full Replace)")
    with colb2:
        do_gap = st.button("⚡ Lengkapi Gap Data")

# ------------------------------
# Fetch source data (gold & fx)
# ------------------------------
# Ambil emas sesuai toggle (untuk tampilan &/atau unggah ke DB)
try:
    if gold_source == "Stooq":
        gold_df_all = fetch_gold_from_stooq()
        gold_df_range = gold_df_all[(gold_df_all["trade_date"] >= start_date) & (gold_df_all["trade_date"] <= end_date)]
    else:
        gold_df_all = fetch_gold_from_yahoo(symbol=yahoo_symbol, start=start_date, end=end_date)
        gold_df_range = gold_df_all.copy()
except Exception as e:
    st.error(f"Gagal mengambil data emas dari {gold_source}: {e}")
    gold_df_all = pd.DataFrame(columns=["trade_date", "Gold_USD"])  # empty fallback
    gold_df_range = gold_df_all

# Ambil kurs IDR/USD
idr_df_range = pd.DataFrame(columns=["trade_date", "IDR_USD"])  # init
fx_err = None
if sheet_id:
    try:
        idr_full = fetch_idr_from_sheets(sheet_id, gid=gid)
        idr_df_range = idr_full[(idr_full["trade_date"] >= start_date) & (idr_full["trade_date"] <= end_date)]
    except Exception as e:
        fx_err = f"Google Sheets gagal: {e}"
else:
    fx_err = "Spreadsheet ID belum diisi."

# Jika Google Sheets gagal, tawarkan Yahoo sebagai fallback cepat (read-only)
if fx_err:
    st.warning(f"{fx_err} → mencoba fallback Yahoo (read-only) …")
    try:
        idr_df_range = fetch_idr_from_yahoo(start=start_date, end=end_date)
    except Exception as e:
        st.error(f"Fallback Yahoo juga gagal: {e}")
        idr_df_range = pd.DataFrame(columns=["trade_date", "IDR_USD"])  # keep empty

# ------------------------------
# Upload ke DB (opsional)
# ------------------------------
engine = get_engine()

if do_full:
    try:
        # Untuk full replace, gunakan seluruh histori agar tabel lengkap
        gold_for_upload = gold_df_all
        if sheet_id:
            idr_for_upload = fetch_idr_from_sheets(sheet_id, gid=gid)
        else:
            idr_for_upload = fetch_idr_from_yahoo()
        upload_full(engine, gold_for_upload, idr_for_upload)
    except Exception as e:
        st.exception(e)

if do_gap:
    try:
        # Gap upload memakai seluruh histori sumber agar bisa difilter > MAX(trade_date)
        gold_for_upload = gold_df_all
        if sheet_id:
            idr_for_upload = fetch_idr_from_sheets(sheet_id, gid=gid)
        else:
            idr_for_upload = fetch_idr_from_yahoo()
        upload_gap(engine, gold_for_upload, idr_for_upload)
    except Exception as e:
        st.exception(e)

# ------------------------------
# Join untuk tampilan (ambil dari DB jika ada; kalau tidak, gabung in-memory)
# ------------------------------
if engine is not None:
    try:
        merged = read_joined(engine, start=start_date, end=end_date)
    except Exception as e:
        st.warning(f"Gagal baca dari DB, gunakan data in-memory: {e}")
        engine = None
        merged = pd.DataFrame()
else:
    merged = pd.DataFrame()

if engine is None:
    # Gabung in-memory (outer join), lalu ffill
    merged = pd.merge(
        gold_df_range, idr_df_range, on="trade_date", how="outer"
    ).sort_values("trade_date")
    if not merged.empty:
        tmp_idx = pd.to_datetime(merged["trade_date"])  # build daily index for ffill
        idx = pd.date_range(start_date, end_date, freq="D")
        merged = merged.set_index(tmp_idx).reindex(idx)
        merged["Gold_USD"] = merged["Gold_USD"].ffill()
        merged["IDR_USD"] = merged["IDR_USD"].ffill()
        merged["trade_date"] = merged.index.date
        merged = merged.reset_index(drop=True)

# ------------------------------
# Output Tables & Charts
# ------------------------------

st.subheader("📄 Data Gabungan (Harian)")
if merged.empty:
    st.info("Data gabungan kosong. Periksa sumber atau tanggal.")
else:
    merged["Gold_IDR_per_gram"] = merged["Gold_USD"] * merged["IDR_USD"] / GRAM_PER_TROY_OZ
    st.dataframe(merged.tail(20), use_container_width=True)

    # Metrics
    last_row = merged.dropna().iloc[-1]
    st.metric(
        label="Gold USD/oz (terkini)",
        value=f"{last_row['Gold_USD']:.2f}",
    )
    st.metric(
        label="USD/IDR (terkini)",
        value=f"{last_row['IDR_USD']:.2f}",
    )
    st.metric(
        label="Emas IDR/gram (terkini)",
        value=f"{last_row['Gold_IDR_per_gram']:.0f}",
    )

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.line(merged, x="trade_date", y="Gold_USD", title="Gold USD/oz (Harian)")
        fig1.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.line(merged, x="trade_date", y="IDR_USD", title="USD/IDR (Harian)")
        fig2.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Aggregation
    st.subheader(f"📊 Agregasi ({freq_label})")
    agg = aggregate(merged, freq=freq)
    if not agg.empty:
        st.dataframe(agg.tail(12), use_container_width=True)
        fig3 = px.bar(agg, x="trade_date", y="Gold_USD_%chg", title="% Perubahan Gold USD/oz")
        fig3.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.line(agg, x="trade_date", y="Gold_IDR_per_gram", title="Emas (IDR/gram)")
        fig4.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig4, use_container_width=True)

# ------------------------------
# Footer
# ------------------------------
st.caption(
    "Sumber emas: Stooq (xauusd) atau Yahoo (XAUUSD=X / GC=F). Kurs: Google Sheets (gviz CSV) atau fallback Yahoo."
)
