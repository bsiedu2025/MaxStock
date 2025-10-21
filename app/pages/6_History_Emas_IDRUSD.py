# -*- coding: utf-8 -*-
# app/pages/6_History_Emas_IDRUSD.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import os
import tempfile
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
import requests
from io import StringIO

# ────────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="💰 Historis Emas & Rupiah", page_icon="📈", layout="wide")
st.title("💰 Historis Emas & Nilai Tukar Rupiah")
st.caption(
    "Menampilkan data historis harga **Emas (riil dari Stooq)** dan **Nilai Tukar Rupiah (riil dari Google Sheets)** "
    "yang tersimpan di tabel terpisah (`gold_data` & `idr_data`) database Anda."
)

# ────────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS (tidak set None agar aman untuk date_input)
# ────────────────────────────────────────────────────────────────────────────────
if "sheet_id_input" not in st.session_state:
    st.session_state.sheet_id_input = "13tvBjRlF_BDAfg2sApGG9jW-KI6A8Fdl97FlaHWwjMY"
if "sheet_gid_input" not in st.session_state:
    st.session_state.sheet_gid_input = "0"  # fleksibel; default 0
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False
if "start_date_filter" not in st.session_state:
    st.session_state.start_date_filter = None
if "end_date_filter" not in st.session_state:
    st.session_state.end_date_filter = None
if "min_date_db" not in st.session_state:
    st.session_state.min_date_db = None

# ────────────────────────────────────────────────────────────────────────────────
# DB CONNECTION
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _build_engine():
    host = os.getenv("DB_HOST", st.secrets.get("DB_HOST", ""))
    port = int(os.getenv("DB_PORT", st.secrets.get("DB_PORT", 3306)))
    database = os.getenv("DB_NAME", st.secrets.get("DB_NAME", ""))
    user = os.getenv("DB_USER", st.secrets.get("DB_USER", ""))
    password = os.getenv("DB_PASSWORD", st.secrets.get("DB_PASSWORD", ""))
    ssl_ca = os.getenv("DB_SSL_CA", st.secrets.get("DB_SSL_CA", ""))

    pwd = quote_plus(str(password))
    connect_args = {}

    try:
        if ssl_ca and "BEGIN CERTIFICATE" in ssl_ca:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
            tmp.write(ssl_ca.encode("utf-8")); tmp.flush()
            connect_args["ssl_ca"] = tmp.name
    except Exception as e:
        st.warning(f"Error saat menyiapkan SSL CA: {e}")

    url = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{database}"
    return create_engine(url, connect_args=connect_args, pool_recycle=300, pool_pre_ping=True)

engine = _build_engine()
GOLD_TABLE = "gold_data"
IDR_TABLE = "idr_data"

@st.cache_data(ttl=120)
def _table_exists(name: str) -> bool:
    """Cek tabel via information_schema (lebih cepat & jelas)."""
    try:
        with engine.connect() as con:
            q = text("""
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = :t
                LIMIT 1
            """)
            return con.execute(q, {"t": name}).scalar() is not None
    except Exception:
        return False

@st.cache_data(ttl=60)
def get_latest_trade_date(table_name: str) -> datetime.date:
    """Tanggal terakhir di tabel tertentu (default fallback 1990-01-01)."""
    if not _table_exists(table_name):
        return datetime(1990, 1, 1).date()
    try:
        with engine.connect() as con:
            q = text(f"SELECT MAX(trade_date) FROM {table_name}")
            d = con.execute(q).scalar()
            return d if d else datetime(1990, 1, 1).date()
    except Exception:
        return datetime(1990, 1, 1).date()

# ────────────────────────────────────────────────────────────────────────────────
# FETCHERS (Stooq & Google Sheets)
# ────────────────────────────────────────────────────────────────────────────────
def fetch_gold_from_stooq() -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    try:
        df = pd.read_csv(url, index_col="Date", parse_dates=True)
        df = df[["Close"]].rename(columns={"Close": "Gold_USD"}).sort_index()
        df = df.reset_index().rename(columns={"Date": "trade_date"})
        df["trade_date"] = df["trade_date"].dt.date
        df.dropna(inplace=True)
        return df[["trade_date", "Gold_USD"]]
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data Emas Stooq: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_idr_from_sheets(sheet_id: str, gid: str = "0") -> pd.DataFrame:
    """Ambil kurs IDR/USD dari Google Sheets (kolom A: tanggal, kolom B: nilai)."""
    if not sheet_id:
        st.error("Spreadsheet ID tidak boleh kosong.")
        return pd.DataFrame()

    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
    try:
        resp = requests.get(csv_url, timeout=30)
        resp.raise_for_status()
        df_raw = pd.read_csv(StringIO(resp.text), header=None)
        df_raw.dropna(how="all", inplace=True)

        # Cari baris pertama yang tampak seperti tanggal
        first_idx = df_raw[df_raw.iloc[:, 0].astype(str).str.contains(r"\d{1,2}/\d{1,2}/\d{4}")].index.min()
        if pd.isna(first_idx):
            st.error("Gagal menemukan baris data Rupiah. Pastikan kolom A berisi tanggal & sharing 'Anyone with the link (Viewer)'.")
            return pd.DataFrame()

        df = df_raw.iloc[first_idx:, [0, 1]].copy()
        df.columns = ["trade_date_raw", "IDR_USD"]

        # Bersihkan angka
        df["IDR_USD"] = (
            df["IDR_USD"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
        )
        df["IDR_USD"] = pd.to_numeric(df["IDR_USD"], errors="coerce")

        # Parse tanggal (coba 2 cara)
        td = pd.to_datetime(df["trade_date_raw"], errors="coerce")
        if td.isna().mean() > 0.5:
            td = pd.to_datetime(df["trade_date_raw"], errors="coerce", dayfirst=True)

        df["trade_date"] = td.dt.date
        df.dropna(subset=["trade_date", "IDR_USD"], inplace=True)

        return df[["trade_date", "IDR_USD"]]
    except requests.exceptions.HTTPError as he:
        st.error(f"Gagal mengambil data Rupiah dari Sheets. Cek Share setting. Error: {he}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat parsing Rupiah: {e}")
        return pd.DataFrame()

# ────────────────────────────────────────────────────────────────────────────────
# WRITE HELPERS
# ────────────────────────────────────────────────────────────────────────────────
def _create_and_upload_table(df_data: pd.DataFrame, table_name: str, value_col: str, replace: bool = False):
    """Create (kalau perlu) dan append data. replace=True akan drop & recreate."""
    total_rows = len(df_data)
    eng = _build_engine()

    if replace:
        with st.status(f"1. Menyiapkan Tabel {table_name} (REPLACE)...", expanded=False) as s:
            with eng.connect() as con:
                con.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                con.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        trade_date DATE NOT NULL,
                        {value_col} FLOAT,
                        PRIMARY KEY (trade_date)
                    ) ENGINE=InnoDB;
                """))
                con.commit()
            s.update(label=f"✅ Tabel {table_name} siap.", state="complete")

    with st.status(f"2. Mengunggah {total_rows} baris ke {table_name}...", expanded=False) as s:
        try:
            df_tmp = df_data.set_index("trade_date")
            df_tmp.to_sql(name=table_name, con=eng, if_exists="append", index=True, chunksize=5000)
            s.update(label=f"✅ Upload {table_name} selesai ({total_rows} baris).", state="complete")
            return True
        except Exception as e:
            st.error(f"❌ Gagal unggah {table_name}! Error: {e}")
            return False

def upload_full_data_to_db_two_tables(sheet_id: str, gid: str):
    st.session_state.is_loading = True
    st.cache_data.clear()
    st.cache_resource.clear()

    df_gold = fetch_gold_from_stooq()
    df_idr = fetch_idr_from_sheets(sheet_id, gid)

    if df_gold.empty:
        st.error("Data Emas tidak ditemukan dari Stooq.")
        st.session_state.is_loading = False
        return
    if df_idr.empty:
        st.error("Data Rupiah tidak ditemukan dari Sheets.")
        st.session_state.is_loading = False
        return

    st.subheader("Proses Upload Emas (`gold_data`)")
    ok1 = _create_and_upload_table(df_gold[["trade_date", "Gold_USD"]], GOLD_TABLE, "Gold_USD", replace=True)

    st.subheader("Proses Upload Rupiah (`idr_data`)")
    ok2 = _create_and_upload_table(df_idr[["trade_date", "IDR_USD"]], IDR_TABLE, "IDR_USD", replace=True)

    st.session_state.is_loading = False
    if ok1 and ok2:
        st.success("🎉 Kedua tabel berhasil dibuat & diisi.")
        st.rerun()
    else:
        st.error("⚠️ Sebagian gagal. Cek log di atas.")

def delete_all_tables():
    try:
        with engine.connect() as con:
            con.execute(text(f"DROP TABLE IF EXISTS {GOLD_TABLE}"))
            con.execute(text(f"DROP TABLE IF EXISTS {IDR_TABLE}"))
            con.commit()
        st.success(f"Kedua tabel (`{GOLD_TABLE}` dan `{IDR_TABLE}`) dihapus.")
        st.cache_data.clear(); st.cache_resource.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Gagal menghapus tabel: {e}")

def upload_gap_data_two_tables(sheet_id: str, gid: str):
    st.session_state.is_loading = True
    last_gold = get_latest_trade_date(GOLD_TABLE)
    last_idr = get_latest_trade_date(IDR_TABLE)

    df_full_gold = fetch_gold_from_stooq()
    df_full_idr = fetch_idr_from_sheets(sheet_id, gid)

    if df_full_gold.empty or df_full_idr.empty:
        st.error("Gagal mengambil data sumber (emas/kurs).")
        st.session_state.is_loading = False
        return

    df_gold_gap = df_full_gold[df_full_gold["trade_date"] > last_gold].copy()
    df_idr_gap  = df_full_idr[df_full_idr["trade_date"]  > last_idr].copy()

    if df_gold_gap.empty and df_idr_gap.empty:
        st.info("Tidak ada data baru. Sudah up-to-date.")
        st.session_state.is_loading = False
        return

    if not df_gold_gap.empty:
        with st.status(f"Mengisi gap {GOLD_TABLE} ({len(df_gold_gap)} baris)...", expanded=True) as s:
            df_gold_gap.set_index("trade_date").to_sql(GOLD_TABLE, con=engine, if_exists="append", index=True, chunksize=500)
            s.update(label=f"✅ Gap {GOLD_TABLE} selesai.", state="complete", expanded=False)

    if not df_idr_gap.empty:
        with st.status(f"Mengisi gap {IDR_TABLE} ({len(df_idr_gap)} baris)...", expanded=True) as s:
            df_idr_gap.set_index("trade_date").to_sql(IDR_TABLE, con=engine, if_exists="append", index=True, chunksize=500)
            s.update(label=f"✅ Gap {IDR_TABLE} selesai.", state="complete", expanded=False)

    st.cache_data.clear()
    st.success("🎉 Kedua tabel ter-update!")
    st.session_state.is_loading = False
    st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# READ & MERGE
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def fetch_and_merge_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    if not (_table_exists(GOLD_TABLE) and _table_exists(IDR_TABLE)):
        return pd.DataFrame()

    sql_gold = f"SELECT trade_date, Gold_USD FROM {GOLD_TABLE} WHERE trade_date BETWEEN :s AND :e"
    sql_idr  = f"SELECT trade_date, IDR_USD  FROM {IDR_TABLE}  WHERE trade_date BETWEEN :s AND :e"
    params = {"s": start_date, "e": end_date}
    try:
        with engine.connect() as con:
            df_gold = pd.read_sql(text(sql_gold), con, params=params)
            df_idr  = pd.read_sql(text(sql_idr),  con, params=params)

        df_gold["trade_date"] = pd.to_datetime(df_gold["trade_date"])
        df_idr["trade_date"]  = pd.to_datetime(df_idr["trade_date"])
        df = pd.merge(df_gold, df_idr, on="trade_date", how="outer").set_index("trade_date").sort_index()
        df = df.ffill()  # isi gap untuk garis halus
        return df
    except Exception as e:
        st.error(f"Gagal ambil/merge data makro: {e}")
        return pd.DataFrame()

def aggregate_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "Harian":
        return df
    rule = {"Mingguan": "W", "Bulanan": "M", "Tahunan": "A-DEC"}.get(freq, None)
    if not rule:
        return df
    out = df.resample(rule).last().dropna()
    out["Gold_Change_Pct"] = out["Gold_USD"].pct_change() * 100
    out["IDR_Change_Pct"]  = out["IDR_USD"].pct_change()  * 100
    return out

# ────────────────────────────────────────────────────────────────────────────────
# SIDEBAR: Sheets & Maintenance
# ────────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🔑 Sumber Data Rupiah Riil")
st.sidebar.markdown("Masukkan **ID Spreadsheet** (dan opsional **GID** tab) yang berisi data USD/IDR.")
st.session_state.sheet_id_input = st.sidebar.text_input(
    "Spreadsheet ID (URL antara /d/ dan /edit)",
    value=st.session_state.sheet_id_input,
    key="sheet_id_key",
)
st.session_state.sheet_gid_input = st.sidebar.text_input(
    "GID (ID sheet/tab, default 0)",
    value=st.session_state.sheet_gid_input,
    key="sheet_gid_key",
)

# Cek tabel
if not (_table_exists(GOLD_TABLE) and _table_exists(IDR_TABLE)):
    st.warning(f"Tabel makro belum lengkap. Dibutuhkan `{GOLD_TABLE}` dan `{IDR_TABLE}`.")
    with st.expander("🛠️ Setup Awal: Buat 2 Tabel & Isi Data"):
        st.info("Pastikan Google Sheet di-share **Anyone with the link (Viewer)** dan ID/GID benar.")
        if st.button("📥 Ambil & Buat Tabel (REPLACE)", type="primary", disabled=st.session_state.is_loading or not st.session_state.sheet_id_input):
            upload_full_data_to_db_two_tables(st.session_state.sheet_id_input, st.session_state.sheet_gid_input)
        if st.button("🗑️ Hapus SEMUA Tabel Macro Data", key="delete_all_initial_table", disabled=st.session_state.is_loading):
            delete_all_tables()
    st.stop()
else:
    with st.sidebar:
        st.header("🔄 Isi Gap Data Otomatis")
        last_date_gold = get_latest_trade_date(GOLD_TABLE)
        last_date_idr  = get_latest_trade_date(IDR_TABLE)
        today = datetime.now().date()
        last_update_min = min(last_date_gold, last_date_idr)

        if last_update_min >= today:
            st.success("✅ Data sudah terbaru hingga hari ini.")
        else:
            st.info(f"Update terakhir: {last_update_min.strftime('%Y-%m-%d')} → ada gap.")
            if st.button("⚡ Lengkapi Gap Data (Stooq+Sheets)", type="primary",
                         disabled=st.session_state.is_loading or not st.session_state.sheet_id_input):
                upload_gap_data_two_tables(st.session_state.sheet_id_input, st.session_state.sheet_gid_input)

    with st.expander("🛠️ Opsi Perawatan Data Makro"):
        st.info("Gunakan tombol di bawah untuk timpa semua data atau hapus tabel.")
        col_upd, col_del = st.columns(2)
        with col_upd:
            if st.button("🔄 Timpa Kedua Tabel (REPLACE)", key="btn_replace_sheets",
                         disabled=st.session_state.is_loading or not st.session_state.sheet_id_input):
                upload_full_data_to_db_two_tables(st.session_state.sheet_id_input, st.session_state.sheet_gid_input)
        with col_del:
            if st.button("🗑️ Hapus SEMUA Tabel Macro Data", key="btn_delete_table",
                         disabled=st.session_state.is_loading):
                delete_all_tables()

# ────────────────────────────────────────────────────────────────────────────────
# FILTER PERIODE
# ────────────────────────────────────────────────────────────────────────────────
end_date = datetime.now().date()
try:
    with engine.connect() as con:
        q = text(f"""
            SELECT MIN(trade_date) FROM (
                SELECT MIN(trade_date) AS trade_date FROM {GOLD_TABLE}
                UNION ALL
                SELECT MIN(trade_date) AS trade_date FROM {IDR_TABLE}
            ) AS combined_dates
        """)
        min_date_db = con.execute(q).scalar() or datetime(1990, 1, 1).date()
except Exception:
    min_date_db = datetime(1990, 1, 1).date()

st.session_state.min_date_db = min_date_db
DEFAULT_START_DATE_MANUAL = datetime(2003, 12, 1).date()
default_start_value = max(DEFAULT_START_DATE_MANUAL, st.session_state.min_date_db)

# Set default kalau belum ada nilai (atau None)
if st.session_state.start_date_filter is None:
    st.session_state.start_date_filter = default_start_value
if st.session_state.end_date_filter is None:
    st.session_state.end_date_filter = end_date

selected_start_date = st.sidebar.date_input(
    "Tanggal Mulai",
    value=st.session_state.start_date_filter,
    min_value=st.session_state.min_date_db,
    max_value=end_date,
    key="filter_start",
)
selected_end_date = st.sidebar.date_input(
    "Tanggal Akhir",
    value=st.session_state.end_date_filter,
    min_value=selected_start_date,
    max_value=end_date,
    key="filter_end",
)

st.session_state.start_date_filter = selected_start_date
st.session_state.end_date_filter = selected_end_date

# Agregasi
st.sidebar.markdown("---")
st.sidebar.header("📏 Alat Pengukuran (ROI)")
aggregation_freq = st.sidebar.selectbox("Agregasi Data", ["Harian", "Mingguan", "Bulanan", "Tahunan"], index=0)

# ────────────────────────────────────────────────────────────────────────────────
# DATAFRAME UTAMA & METRICS
# ────────────────────────────────────────────────────────────────────────────────
raw_df = fetch_and_merge_macro_data(
    selected_start_date.strftime("%Y-%m-%d"),
    selected_end_date.strftime("%Y-%m-%d"),
)
if raw_df.empty:
    st.warning("Tidak ada data makro pada rentang ini.")
    st.stop()

df = aggregate_data(raw_df, aggregation_freq)

def _fmt(val, prefix="", suffix=""):
    if pd.isna(val):
        return "N/A"
    if prefix == "Rp":
        return f"Rp{val:,.0f}"
    return f"{prefix}{val:,.2f}{suffix}"

st.subheader("Ringkasan Data Makro Terbaru")
col_g1, col_g2, col_i1, col_i2 = st.columns(4)

if len(df) >= 2:
    latest_gold, prev_gold = df["Gold_USD"].iloc[-1], df["Gold_USD"].iloc[-2]
    latest_idr,  prev_idr  = df["IDR_USD"].iloc[-1],  df["IDR_USD"].iloc[-2]

    change_gold = (latest_gold - prev_gold) if pd.notna(latest_gold) and pd.notna(prev_gold) else np.nan
    change_gold_pct = (change_gold / prev_gold * 100) if pd.notna(change_gold) and prev_gold else 0

    change_idr = (latest_idr - prev_idr) if pd.notna(latest_idr) and pd.notna(prev_idr) else np.nan
    change_idr_pct = (change_idr / prev_idr * 100) if pd.notna(change_idr) and prev_idr else 0
else:
    latest_gold = df["Gold_USD"].iloc[-1] if len(df) else np.nan
    latest_idr  = df["IDR_USD"].iloc[-1]  if len(df) else np.nan
    change_gold = change_gold_pct = change_idr = change_idr_pct = 0

with col_g1:
    st.metric("Harga Emas (USD/oz)", _fmt(latest_gold, prefix="$"), f"{change_gold:+.2f}" if pd.notna(change_gold) and change_gold != 0 else None)
with col_g2:
    st.metric("Perubahan Emas (%)", _fmt(change_gold_pct, suffix="%"))
with col_i1:
    st.metric("Nilai Tukar (IDR/USD)", _fmt(latest_idr, prefix="Rp"), f"Rp{change_idr:+.0f}" if pd.notna(change_idr) and change_idr != 0 else None)
with col_i2:
    st.metric("Perubahan Rupiah (%)", _fmt(change_idr_pct, suffix="%"), delta_color="inverse")

# ROI otomatis berdasarkan filter periode
measurement_results = {}
if not df.empty and len(df) >= 2:
    start_gold, end_gold = df["Gold_USD"].iloc[0], df["Gold_USD"].iloc[-1]
    start_idr,  end_idr  = df["IDR_USD"].iloc[0],  df["IDR_USD"].iloc[-1]
    gold_pct = (end_gold / start_gold - 1) * 100 if start_gold else np.nan
    idr_pct  = (end_idr  / start_idr  - 1) * 100 if start_idr  else np.nan
    measurement_results = {
        "start_date": df.index[0].date(),
        "end_date": df.index[-1].date(),
        "gold_pct": gold_pct,
        "idr_pct": idr_pct,
    }

st.markdown("---")
if measurement_results:
    st.subheader(f"ROI Periode Filter ({measurement_results['start_date']} s/d {measurement_results['end_date']})")
    m1, m2 = st.columns(2)
    with m1: st.metric("ROI Emas (%)", f"{measurement_results['gold_pct']:+.2f}%")
    with m2: st.metric("ROI Rupiah (%)", f"{measurement_results['idr_pct']:+.2f}%", delta_color="inverse")

# ────────────────────────────────────────────────────────────────────────────────
# CHARTS
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Grafik Historis (Emas dan Nilai Tukar)")
st.info("💡 ROI di atas dihitung dari rentang waktu **Filter Periode Makro**.")

# 1) Emas
st.markdown("#### Harga Emas Dunia (USD/oz)")
fig_gold = go.Figure()
fig_gold.add_trace(go.Scatter(x=df.index, y=df["Gold_USD"], name="Gold (USD/oz)", line=dict(color="#FFD700", width=1.5)))
fig_gold.update_layout(
    height=450,
    title_text=f"Agregasi {aggregation_freq} ({selected_start_date} s/d {selected_end_date})",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="right", x=1),
)
fig_gold.update_yaxes(title_text="Harga Emas (USD/oz)", tickformat="$,.0f")
fig_gold.update_xaxes(
    type="date",
    rangeslider_visible=True,
    rangeselector=dict(buttons=[
        dict(count=1, label="1B", step="month", stepmode="backward"),
        dict(count=6, label="6B", step="month", stepmode="backward"),
        dict(count=1, label="1T", step="year", stepmode="backward"),
        dict(step="all"),
    ])
)
st.plotly_chart(fig_gold, key="gold_chart", use_container_width=True,
                config={'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'displaylogo': False, 'displayModeBar': True})

# 2) Rupiah
st.markdown("#### Nilai Tukar Rupiah (IDR/USD)")
fig_idr = go.Figure()
fig_idr.add_trace(go.Scatter(x=df.index, y=df["IDR_USD"], name="IDR/USD Rate", line=dict(color="#008000", width=1.5)))
fig_idr.update_layout(
    height=450,
    title_text=f"Agregasi {aggregation_freq} ({selected_start_date} s/d {selected_end_date})",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="right", x=1),
)
fig_idr.update_yaxes(title_text="Nilai Tukar (IDR/USD)", tickprefix="Rp", tickformat=",0f")
fig_idr.update_xaxes(
    type="date",
    rangeslider_visible=True,
    rangeselector=dict(buttons=[
        dict(count=1, label="1B", step="month", stepmode="backward"),
        dict(count=6, label="6B", step="month", stepmode="backward"),
        dict(count=1, label="1T", step="year", stepmode="backward"),
        dict(step="all"),
    ])
)
st.plotly_chart(fig_idr, use_container_width=True)

# 3) Perbandingan % change
st.markdown("---")
st.subheader(f"Analisis Pergerakan {aggregation_freq} (Persentase)")
if aggregation_freq != "Harian":
    df_chart = df.copy()
else:
    df_chart = raw_df.copy()
df_chart["Gold_Change_Pct"] = df_chart["Gold_USD"].pct_change() * 100
df_chart["IDR_Change_Pct"]  = df_chart["IDR_USD"].pct_change()  * 100
df_chart = df_chart.dropna()

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(
    go.Bar(
        x=df_chart.index,
        y=df_chart["Gold_Change_Pct"],
        name=f"Emas (% per {aggregation_freq})",
        opacity=0.8,
        marker_color=np.where(df_chart["Gold_Change_Pct"] > 0, "green", "red").tolist(),
    ),
    secondary_y=False,
)
fig2.add_trace(
    go.Scatter(x=df_chart.index, y=df_chart["IDR_Change_Pct"], name=f"Rupiah (% per {aggregation_freq})",
               mode="lines", line=dict(color="orange", width=2)),
    secondary_y=True,
)
fig2.update_yaxes(title_text="Emas (% Perubahan)", secondary_y=False)
fig2.update_yaxes(title_text="Rupiah (% Perubahan)", secondary_y=True)
fig2.update_layout(
    title_text=f"Perbandingan Perubahan {aggregation_freq} Emas vs Rupiah",
    height=500, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("⚠️ Data Emas: Stooq. Data Rupiah: Google Sheets. Pastikan Spreadsheet ID/GID benar dan sharing sudah terbuka.")
