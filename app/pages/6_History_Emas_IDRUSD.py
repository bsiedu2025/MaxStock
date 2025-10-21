# -*- coding: utf-8 -*-
# app/pages/6_Pergerakan_Asing_FF.py (Dashboard Makro Baru: Emas & Rupiah dari MariaDB)

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
import time
import requests 
import math
from io import StringIO # Diperlukan untuk parsing Sheets CSV

st.set_page_config(page_title="💰 Historis Emas & Rupiah", page_icon="📈", layout="wide")
st.title("💰 Historis Emas & Nilai Tukar Rupiah (MariaDB)")
st.caption(
    "Menampilkan data historis harga **Emas (riil dari Stooq)** dan **Nilai Tukar Rupiah (riil dari Google Sheets)** yang tersimpan di tabel terpisah (`gold_data` & `idr_data`) database Anda."
)

# Inisialisasi session state
if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False
    
# State untuk menyimpan Sheet ID (untuk digunakan di seluruh app)
if 'sheet_id_input' not in st.session_state:
    st.session_state.sheet_id_input = "13tvBjRlF_BDAfg2sApGG9jW-KI6A8Fdl97FlaHWwjMY" 

# ────────────────────────────────────────────────────────────────────────────────
# DB Connection & Utility 
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _build_engine():
    """Membangun koneksi SQLAlchemy ke database."""
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

@st.cache_data(ttl=3600)
def _table_exists(name: str) -> bool:
    """Mengecek apakah tabel ada di database saat ini."""
    try:
        engine = _build_engine()
        with engine.connect() as con:
            q = text("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = :t
            """)
            return bool(con.execute(q, {"t": name}).scalar())
    except Exception:
        return False

# Inisialisasi Engine
engine = _build_engine() 
GOLD_TABLE = "gold_data"
IDR_TABLE = "idr_data"

# ────────────────────────────────────────────────────────────────────────────────
# Data Fetchers (Emas Stooq & Rupiah Sheets)
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_latest_trade_date(table_name: str) -> datetime.date:
    """Mencari tanggal terakhir data di database untuk tabel tertentu."""
    if not _table_exists(table_name):
         return datetime(1990, 1, 1).date()

    try:
        with engine.connect() as con:
            q = text(f"SELECT MAX(trade_date) FROM {table_name}")
            max_date = con.execute(q).scalar()
            return max_date if max_date else datetime(1990, 1, 1).date()
    except Exception:
        return datetime(1990, 1, 1).date() # Default awal 1990

# --- NEW: Fetch Rupiah dari Google Sheets ---
@st.cache_data(ttl=600)
def fetch_idr_from_sheets(sheet_id: str) -> pd.DataFrame:
    """
    Mengambil data Rupiah dari Google Sheets dengan parsing yang paling robust.
    Asumsi: Data Rupiah ada di Kolom 0 dan 1 (A & B) karena hanya ada satu formula GOOGLEFINANCE.
    """
    if not sheet_id: 
        st.error("Spreadsheet ID tidak boleh kosong.")
        return pd.DataFrame()
    
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid=0"
    
    try:
        response = requests.get(csv_url, timeout=30)
        response.raise_for_status() 
        
        # 1. Baca CSV mentah tanpa header
        df_raw = pd.read_csv(StringIO(response.text), header=None, skiprows=0)
        df_raw.dropna(how='all', inplace=True)
        
        # 2. Identifikasi Baris Awal Data (Baris pertama yang punya format tanggal di Kolom 0)
        first_data_row_index = df_raw[df_raw.iloc[:, 0].astype(str).str.contains(r'\d{1,2}/\d{1,2}/\d{4}')].index.min()
        
        if pd.isna(first_data_row_index):
            st.error("Gagal menemukan baris data Rupiah yang valid. Pastikan hasil formula GOOGLEFINANCE ada di kolom A & B dan file di-set 'Anyone with the link (Viewer)'.")
            return pd.DataFrame()

        # 3. Ambil data Rupiah (Kolom Index 0/Date dan Index 1/Price)
        df_idr = df_raw.iloc[first_data_row_index:].copy()
        df_idr = df_idr.iloc[:, [0, 1]] 
        df_idr.columns = ['trade_date_raw', 'IDR_USD']
        
        # 4. Pembersihan Data
        df_idr.replace('', np.nan, inplace=True)
        df_idr.dropna(how='all', inplace=True)
        
        # Konversi Tanggal (Asumsi format mm/dd/yyyy)
        df_idr['trade_date'] = pd.to_datetime(df_idr['trade_date_raw'], errors='coerce')
        
        # Pembersihan Angka
        df_idr['IDR_USD'] = df_idr['IDR_USD'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
        df_idr['IDR_USD'] = pd.to_numeric(df_idr['IDR_USD'], errors='coerce')
        
        df_idr.dropna(subset=['trade_date', 'IDR_USD'], inplace=True)
        
        df_idr = df_idr[['trade_date', 'IDR_USD']]
        df_idr['trade_date'] = df_idr['trade_date'].dt.date
        
        return df_idr
        
    except requests.exceptions.HTTPError as he:
        # Menangkap Error 404 (Not Found) atau 403 (Forbidden)
        st.error(f"Gagal mengambil data Rupiah dari Sheets. Pastikan link di-set 'Anyone with the link (Viewer)'. Error: {he}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat parsing Rupiah: {e}")
        return pd.DataFrame()


def fetch_gold_from_stooq() -> pd.DataFrame:
    """Mengambil data Emas historis dari Stooq."""
    stooq_url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    
    try:
        df_gold = pd.read_csv(stooq_url, index_col='Date', parse_dates=True)
        df_gold = df_gold[['Close']].copy()
        df_gold.columns = ['Gold_USD']
        df_gold = df_gold.sort_index(ascending=True).reset_index()
        df_gold.rename(columns={'Date': 'trade_date'}, inplace=True)
        df_gold['trade_date'] = df_gold['trade_date'].dt.date
        df_gold.dropna(inplace=True)
        
        return df_gold
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data Emas Stooq: {e}")
        return pd.DataFrame()


# ────────────────────────────────────────────────────────────────────────────────
# Uploader/Seeder Data (Menangani Dua Tabel)
# ────────────────────────────────────────────────────────────────────────────────

def _create_and_upload_table(df_data: pd.DataFrame, table_name: str, value_col: str, action: str = 'full'):
    """Fungsi pembantu untuk membuat tabel dan mengunggah data (full atau append)."""
    total_rows = len(df_data)
    
    engine = _build_engine()
    
    if action == 'full':
        # Mode CREATE TABLE
        with st.status(f"1. Menyiapkan dan Membuat Tabel {table_name}...", expanded=False) as status_bar:
            with engine.connect() as con:
                create_table_sql = f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        trade_date DATE NOT NULL,
                        {value_col} FLOAT,
                        PRIMARY KEY (trade_date)
                    ) ENGINE=InnoDB;
                """
                con.execute(text(f"DROP TABLE IF EXISTS {table_name}")) 
                con.execute(text(create_table_sql))
                con.commit()
            status_bar.update(label=f"✅ Tabel {table_name} siap dibuat.", state="complete")

    # Mode APPEND DATA
    with st.status(f"2. Mengunggah {total_rows} baris ke {table_name}...", expanded=False) as status_bar:
        try:
            df_temp = df_data.set_index('trade_date')
            df_temp.to_sql(
                name=table_name, 
                con=engine, 
                if_exists='append',
                index=True,
                chunksize=5000 
            )
            status_bar.update(label=f"✅ Pengunggahan {table_name} Selesai! ({total_rows} baris)", state="complete")
            return True
        except Exception as e:
            st.error(f"❌ Gagal mengunggah data {table_name}! Error: {e}")
            return False


def upload_full_data_to_db_two_tables(sheet_id: str):
    """Menyimpan data penuh Emas dan Rupiah ke dua tabel terpisah dengan commit terpisah."""
    st.session_state.is_loading = True
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # 1. Fetch Data
    df_gold = fetch_gold_from_stooq()
    df_idr = fetch_idr_from_sheets(sheet_id)
    
    if df_gold.empty:
        st.error("Data Emas tidak ditemukan dari Stooq. Tidak bisa melanjutkan.")
        st.session_state.is_loading = False
        return
    
    if df_idr.empty:
        st.error("Data Rupiah tidak ditemukan dari Sheets. Tidak bisa melanjutkan.")
        st.session_state.is_loading = False
        return
        
    # 2. Upload Emas (Pisah)
    st.subheader("Proses Upload Emas (`gold_data`)")
    df_gold_upload = df_gold[['trade_date', 'Gold_USD']]
    success_gold = _create_and_upload_table(df_gold_upload, GOLD_TABLE, 'Gold_USD', action='full')
    
    # 3. Upload Rupiah (Pisah)
    st.subheader("Proses Upload Rupiah (`idr_data`)")
    df_idr_upload = df_idr[['trade_date', 'IDR_USD']]
    success_idr = _create_and_upload_table(df_idr_upload, IDR_TABLE, 'IDR_USD', action='full')

    st.session_state.is_loading = False
    
    if success_gold and success_idr:
        st.success("🎉 Kedua tabel (Emas & Rupiah) berhasil dibuat dan diisi!")
        st.rerun()
    else:
        st.error("⚠️ Proses selesai dengan kegagalan pada satu atau kedua tabel. Cek log di atas.")


def delete_all_tables():
    """Menghapus kedua tabel."""
    try:
        with engine.connect() as con:
            con.execute(text(f"DROP TABLE IF EXISTS {GOLD_TABLE}"))
            con.execute(text(f"DROP TABLE IF EXISTS {IDR_TABLE}"))
            con.commit()
        st.success(f"Kedua tabel (`{GOLD_TABLE}` dan `{IDR_TABLE}`) berhasil dihapus.")
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Gagal menghapus tabel: {e}")


def upload_gap_data_two_tables(sheet_id: str):
    """Mengisi gap data Emas dan Rupiah."""
    st.session_state.is_loading = True
    
    # 1. Tentukan Gap Dates
    last_date_gold = get_latest_trade_date(GOLD_TABLE)
    last_date_idr = get_latest_trade_date(IDR_TABLE)
    
    # Ambil data sumber lengkap
    df_full_gold = fetch_gold_from_stooq()
    df_full_idr = fetch_idr_from_sheets(sheet_id)
    
    if df_full_gold.empty or df_full_idr.empty:
        st.error("Gagal mengambil data Emas atau Rupiah dari sumber.")
        st.session_state.is_loading = False
        return

    # 2. Gap Emas
    df_gold_gap = df_full_gold[df_full_gold['trade_date'] > last_date_gold].copy()
    
    # 3. Gap Rupiah
    df_idr_gap = df_full_idr[df_full_idr['trade_date'] > last_date_idr].copy()
    
    
    if df_gold_gap.empty and df_idr_gap.empty:
        st.info("Tidak ada data baru yang ditemukan dari sumber. Data Anda sudah terbaru.")
        st.session_state.is_loading = False
        return

    # Upload Emas Gap
    if not df_gold_gap.empty:
        total_rows = len(df_gold_gap)
        with st.status(f"Mengisi gap {GOLD_TABLE} ({total_rows} baris)...", expanded=True) as status_bar:
            df_gold_gap.set_index('trade_date').to_sql(name=GOLD_TABLE, con=engine, if_exists='append', index=True, chunksize=500)
            status_bar.update(label=f"✅ Gap {GOLD_TABLE} selesai.", state="complete", expanded=False)

    # Upload Rupiah Gap
    if not df_idr_gap.empty:
        total_rows = len(df_idr_gap)
        with st.status(f"Mengisi gap {IDR_TABLE} ({total_rows} baris)...", expanded=True) as status_bar:
            df_idr_gap.set_index('trade_date').to_sql(name=IDR_TABLE, con=engine, if_exists='append', index=True, chunksize=500)
            status_bar.update(label=f"✅ Gap {IDR_TABLE} selesai.", state="complete", expanded=False)

    st.cache_data.clear()
    st.success("🎉 Kedua tabel berhasil di-update dengan data terbaru!")
    st.session_state.is_loading = False
    st.rerun() 


# ────────────────────────────────────────────────────────────────────────────────
# Data Fetcher & Aggregator (Satu Fungsi Utama)
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def fetch_and_merge_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Mengambil data Emas dan Rupiah, lalu menggabungkannya."""
    if not (_table_exists(GOLD_TABLE) and _table_exists(IDR_TABLE)):
        return pd.DataFrame()
    
    sql_gold = f"SELECT trade_date, Gold_USD FROM {GOLD_TABLE} WHERE trade_date BETWEEN :start_date AND :end_date"
    sql_idr = f"SELECT trade_date, IDR_USD FROM {IDR_TABLE} WHERE trade_date BETWEEN :start_date AND :end_date"
    params = {"start_date": start_date, "end_date": end_date}
    
    try:
        with engine.connect() as con:
            df_gold = pd.read_sql(text(sql_gold), con, params=params)
            df_idr = pd.read_sql(text(sql_idr), con, params=params)
        
        # Merge data
        df_gold['trade_date'] = pd.to_datetime(df_gold['trade_date'])
        df_idr['trade_date'] = pd.to_datetime(df_idr['trade_date'])
        
        df_merged = pd.merge(df_gold, df_idr, on='trade_date', how='outer') # Outer join agar tidak ada data hilang
        df_merged = df_merged.set_index('trade_date').sort_index()
        
        # Interpolasi untuk mengisi hari yang hilang (Opsional, tapi membuat grafik mulus)
        df_merged = df_merged.ffill() 
        
        return df_merged

    except Exception as e:
        st.error(f"Gagal mengambil atau menggabungkan data makro: {e}")
        return pd.DataFrame()


def aggregate_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Mengagregasi data harian ke frekuensi Mingguan, Bulanan, atau Tahunan."""
    if freq == 'Harian':
        return df
    
    # ... (Logika agregasi tidak berubah)
    if freq == 'Mingguan':
        rule = 'W'
    elif freq == 'Bulanan':
        rule = 'M'
    elif freq == 'Tahunan':
        rule = 'Y'
    else:
        return df 
    
    aggregated_df = df.resample(rule).last().dropna()
    
    aggregated_df['Gold_Change_Pct'] = aggregated_df['Gold_USD'].pct_change() * 100
    aggregated_df['IDR_Change_Pct'] = aggregated_df['IDR_USD'].pct_change() * 100
    
    return aggregated_df

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit Interface
# ────────────────────────────────────────────────────────────────────────────────

# Input Sidebar untuk Google Sheet ID
st.sidebar.header("🔑 Sumber Data Rupiah Riil")
st.sidebar.markdown("Masukkan **ID Spreadsheet** Google Sheets Anda yang berisi data USD/IDR.")
st.session_state.sheet_id_input = st.sidebar.text_input(
    "ID Spreadsheet (URL antara /d/ dan /edit)", 
    value=st.session_state.sheet_id_input, 
    key="sheet_id_key"
)


# Cek keberadaan tabel Emas dan Rupiah
if not (_table_exists(GOLD_TABLE) and _table_exists(IDR_TABLE)):
    st.warning(f"Tabel makro belum lengkap. Butuh tabel `{GOLD_TABLE}` dan `{IDR_TABLE}`.")
    
    with st.expander("🛠️ Klik di sini untuk membuat 2 tabel makro (Setup Awal)") as exp:
        st.info("Pastikan Anda sudah mengatur **Google Sheet** (Share: Anyone with the link) dan mengisi **Spreadsheet ID** di sidebar.")
        
        if st.button("📥 Ambil Emas & Rupiah Riil & Buat Tabel", disabled=st.session_state.is_loading or not st.session_state.sheet_id_input, type="primary"):
            upload_full_data_to_db_two_tables(st.session_state.sheet_id_input)
            
        if st.button("🗑️ Hapus SEMUA Tabel Macro Data", disabled=st.session_state.is_loading, key="delete_all_initial_table"):
            delete_all_tables()
            
    st.stop()
    
# --- LOGIKA KETIKA KEDUA TABEL SUDAH ADA ---
else:
    # --- FORM UPDATE HARIAN DI SIDEBAR (GAP FILLER) ---
    with st.sidebar:
        st.header("🔄 Isi Gap Data Otomatis")
        
        last_date_gold = get_latest_trade_date(GOLD_TABLE)
        last_date_idr = get_latest_trade_date(IDR_TABLE)
        
        last_update_date = min(last_date_gold, last_date_idr)
        today = datetime.now().date()
        
        if last_update_date >= today:
            st.success("✅ Data Anda sudah terbaru hingga hari ini!")
        else:
            st.info(f"Update terakhir: {last_update_date.strftime('%Y-%m-%d')}. Ada Gap.")
            
            if st.button("⚡ Lengkapi Gap Data (Stooq Emas & Sheets Kurs)", 
                         disabled=st.session_state.is_loading or not st.session_state.sheet_id_input, 
                         type="primary"):
                
                upload_gap_data_two_tables(st.session_state.sheet_id_input)

    st.sidebar.markdown("---")
    
    # Tampilkan tombol untuk update data simulasi 1970 atau menghapus tabel
    with st.expander("🛠️ Opsi Perawatan Data Makro (Tabel sudah ada)") as exp:
        st.info("Gunakan tombol ini untuk menghapus tabel atau memperbarui semua data.")
        
        col_upd, col_del = st.columns(2)
        
        with col_upd:
            if st.button("🔄 Ambil Ulang & Timpa Kedua Tabel", key="btn_replace_sheets", disabled=st.session_state.is_loading or not st.session_state.sheet_id_input):
                upload_full_data_to_db_two_tables(st.session_state.sheet_id_input) 

        with col_del:
            if st.button("🗑️ Hapus SEMUA Tabel Macro Data", key="btn_delete_table", disabled=st.session_state.is_loading):
                delete_all_tables()

# Tampilkan loading indicator global jika sedang proses
if st.session_state.is_loading:
    pass

# Filter Sidebar (Tersedia setelah tabel ada)
if not (_table_exists(GOLD_TABLE) and _table_exists(IDR_TABLE)): st.stop()

st.sidebar.header("Filter Periode Makro")
end_date = datetime.now().date()

# Cek tanggal terlama dari data di DB untuk min_value
try:
    with engine.connect() as con:
        # Mencari tanggal terlama untuk inisialisasi filter
        query_min_date = text(f"SELECT MIN(trade_date) FROM (SELECT MIN(trade_date) AS trade_date FROM {GOLD_TABLE} UNION ALL SELECT MIN(trade_date) AS trade_date FROM {IDR_TABLE}) AS combined_dates")
        min_date_db = con.execute(query_min_date).scalar()
        if not min_date_db:
             min_date_db = datetime(1990, 1, 1).date()

except Exception:
    min_date_db = datetime(1990, 1, 1).date()


# =====================================================================================================================
# [FIX TOTAL] Mengatur Nilai Default ke 1 Desember 2003
# =====================================================================================================================

# Tentukan tanggal awal default manual yang diminta user
DEFAULT_START_DATE_MANUAL = datetime(2003, 12, 1).date()

# [FIX] Menyimpan nilai min_date_db ke session_state
if 'min_date_db' not in st.session_state:
    st.session_state.min_date_db = min_date_db

# Jika filter belum pernah di-set, atur nilai default-nya
if 'start_date_filter' not in st.session_state:
    default_start_value = max(DEFAULT_START_DATE_MANUAL, st.session_state.min_date_db)
    st.session_state.start_date_filter = default_start_value
    st.session_state.end_date_filter = end_date

# Gunakan session state untuk mengontrol nilai
selected_start_date = st.sidebar.date_input("Tanggal Mulai", 
    value=st.session_state.start_date_filter, 
    min_value=st.session_state.min_date_db, 
    max_value=end_date, 
    key="filter_start"
)
selected_end_date = st.sidebar.date_input("Tanggal Akhir", 
    value=st.session_state.end_date_filter, 
    min_value=selected_start_date, 
    max_value=end_date, 
    key="filter_end"
)

# Update session state jika ada perubahan (agar nilai persistent)
st.session_state.start_date_filter = selected_start_date
st.session_state.end_date_filter = selected_end_date

st.sidebar.markdown("---")

# ────────────────────────────────────────────────────────────────────────────────
# ALAT PENGUKURAN (ROI)
# ────────────────────────────────────────────────────────────────────────────────
st.sidebar.header("📏 Alat Pengukuran (ROI)")

# [FIX] Menggunakan nilai dari date_input utama sebagai batas
min_meas_date = selected_start_date
max_meas_date = selected_end_date

# Pastikan tanggal pengukuran berada di dalam rentang filter utama
date_start_meas = st.sidebar.date_input("Titik Awal Pengukuran", value=min_meas_date, min_value=min_meas_date, max_value=max_meas_date, key="meas_start")
date_end_meas = st.sidebar.date_input("Titik Akhir Pengukuran", value=max_meas_date, min_value=date_start_meas, max_value=max_meas_date, key="meas_end")

# Pilihan Agregasi (Mingguan, Bulanan, Tahunan)
st.sidebar.markdown("---")
aggregation_freq = st.sidebar.selectbox(
    "Agregasi Data",
    ['Harian', 'Mingguan', 'Bulanan', 'Tahunan'],
    index=0
)

# Fetch data harian dari DB (untuk pengukuran)
meas_df_raw = fetch_and_merge_macro_data(date_start_meas.strftime('%Y-%m-%d'), date_end_meas.strftime('%Y-%m-%d'))

measurement_results = {}
if not meas_df_raw.empty:
    # Agregasi data pengukuran (ambil data terakhir di periode agregasi yang dipilih)
    meas_df_agg = aggregate_data(meas_df_raw, aggregation_freq)
    
    if not meas_df_agg.empty:
        # Cari nilai di tanggal terdekat
        try:
            # Nilai Awal: Ambil baris pertama
            start_gold = meas_df_agg['Gold_USD'].iloc[0]
            start_idr = meas_df_agg['IDR_USD'].iloc[0]
            start_date_actual = meas_df_agg.index[0].date()
            
            # Nilai Akhir: Ambil baris terakhir
            end_gold = meas_df_agg['Gold_USD'].iloc[-1]
            end_idr = meas_df_agg['IDR_USD'].iloc[-1]
            end_date_actual = meas_df_agg.index[-1].date()

            # Hitung Perubahan
            gold_change_pct = (end_gold / start_gold - 1) * 100 if start_gold else np.nan
            idr_change_pct = (end_idr / start_idr - 1) * 100 if start_idr else np.nan
            
            measurement_results = {
                'start_date': start_date_actual,
                'end_date': end_date_actual,
                'start_gold': start_gold,
                'end_gold': end_gold,
                'gold_pct': gold_change_pct,
                'start_idr': start_idr,
                'end_idr': end_idr,
                'idr_pct': idr_change_pct,
            }
        except Exception as e:
             # st.error(f"Gagal hitung ROI: {e}") # Sembunyikan error dari user
             pass

# Fetch data untuk Grafik Utama (berdasarkan selected_start_date dan selected_end_date)
raw_df = fetch_and_merge_macro_data(selected_start_date.strftime('%Y-%m-%d'), selected_end_date.strftime('%Y-%m-%d'))

if raw_df.empty:
    st.warning(f"Tidak ada data makro tersedia untuk rentang waktu ini di database.")
    st.stop()
    
# Agregasi Data
simulated_df = aggregate_data(raw_df, aggregation_freq)


# Data terakhir (Pastikan indeks ada dan hitung perubahan untuk metrik)
if len(simulated_df) >= 2:
    latest_gold = simulated_df['Gold_USD'].iloc[-1]
    latest_idr = simulated_df['IDR_USD'].iloc[-1]
    prev_gold = simulated_df['Gold_USD'].iloc[-2]
    prev_idr = simulated_df['IDR_USD'].iloc[-2]

    # --- [FIXED] Pengecekan NaN/None sebelum menghitung perubahan ---
    # Cek Gold
    if pd.isna(latest_gold) or pd.isna(prev_gold):
        change_gold = 0; change_gold_pct = 0
        latest_gold_val = np.nan
    else:
        change_gold = latest_gold - prev_gold
        change_gold_pct = (change_gold / prev_gold) * 100 if prev_gold != 0 else 0
        latest_gold_val = latest_gold
    
    # Cek IDR
    if pd.isna(latest_idr) or pd.isna(prev_idr):
        change_idr = 0; change_idr_pct = 0
        latest_idr_val = np.nan
    else:
        change_idr = latest_idr - prev_idr
        change_idr_pct = (change_idr / prev_idr) * 100 if prev_idr != 0 else 0
        latest_idr_val = latest_idr
        
elif len(simulated_df) == 1:
    latest_gold = simulated_df['Gold_USD'].iloc[-1]
    latest_idr = simulated_df['IDR_USD'].iloc[-1]
    change_gold = 0; change_gold_pct = 0
    change_idr = 0; change_idr_pct = 0
    latest_gold_val = latest_gold if not pd.isna(latest_gold) else np.nan
    latest_idr_val = latest_idr if not pd.isna(latest_idr) else np.nan
else:
    st.warning("Data tidak cukup untuk menghitung perubahan.")
    st.stop()


# ────────────────────────────────────────────────────────────────────────────────
# Ringkasan Metrik
st.subheader("Ringkasan Data Makro Terbaru")

col_g1, col_g2, col_i1, col_i2 = st.columns(4)

# --- Fungsi helper untuk format metrik yang tahan NaN/None ---
def format_metric_value(value, prefix="", suffix=""):
    if pd.isna(value):
        return "N/A"
    if prefix == "Rp":
        return f"Rp{value:,.0f}"
    return f"{prefix}{value:,.2f}{suffix}"

with col_g1:
    st.metric(
        label="Harga Emas (USD/oz)",
        value=format_metric_value(latest_gold_val, prefix="$"),
        delta=f"{change_gold:+.2f}" if not pd.isna(change_gold) and change_gold != 0 else None
    )
with col_g2:
    st.metric(
        label="Perubahan Emas (%)",
        value=format_metric_value(change_gold_pct, suffix="%"),
        delta=None,
    )

with col_i1:
    st.metric(
        label="Nilai Tukar (IDR/USD)",
        # [FIX] Menggunakan helper function
        value=format_metric_value(latest_idr_val, prefix="Rp"),
        delta=f"Rp{change_idr:+.0f}" if not pd.isna(change_idr) and change_idr != 0 else None
    )
with col_i2:
    st.metric(
        label="Perubahan Rupiah (%)",
        value=format_metric_value(change_idr_pct, suffix="%"),
        delta=None,
        delta_color="inverse" # Warna delta terbalik, karena kenaikan kurs IDR buruk
    )

st.markdown("---") 

# ────────────────────────────────────────────────────────────────────────────────
# Hasil Pengukuran ROI (Muncul setelah metrik utama)
# ────────────────────────────────────────────────────────────────────────────────
if measurement_results:
    st.subheader(f"Hasil Pengukuran ROI ({measurement_results['start_date']} s/d {measurement_results['end_date']})")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.metric("ROI Emas (%)", f"{measurement_results['gold_pct']:+.2f}%", 
                  help=f"Dari ${measurement_results['start_gold']:,.2f} menjadi ${measurement_results['end_gold']:,.2f}")
    with m_col2:
        st.metric("ROI Rupiah (%)", f"{measurement_results['idr_pct']:+.2f}%", 
                  help=f"Dari Rp{measurement_results['start_idr']:,.0f} menjadi Rp{measurement_results['end_idr']:,.0f}",
                  delta_color="inverse")
    
    st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────────
# Grafik Emas dan Rupiah (2 Grafik Terpisah)
st.subheader("Grafik Historis (Emas dan Nilai Tukar)")

# --- 1. GRAFIK EMAS ---
st.markdown("#### Harga Emas Dunia (USD/oz)")
fig_gold = go.Figure()
fig_gold.add_trace(
    go.Scatter(x=simulated_df.index, y=simulated_df['Gold_USD'], name='Gold (USD/oz)', line=dict(color='#FFD700', width=1.5)), 
)

fig_gold.update_layout(
    height=450,
    title_text=f"Agregasi {aggregation_freq} ({selected_start_date.strftime('%Y-%m-%d')} s/d {selected_end_date.strftime('%Y-%m-%d')})",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="right", x=1),
)

fig_gold.update_yaxes(title_text="Harga Emas (USD/oz)", tickformat="$,.0f")

# Range Selector untuk Emas
fig_gold.update_xaxes(
    type='date',
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1B", step="month", stepmode="backward"),
            dict(count=6, label="6B", step="month", stepmode="backward"),
            dict(count=1, label="1T", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

st.plotly_chart(fig_gold, use_container_width=True)


# --- 2. GRAFIK RUPIAH ---
st.markdown("#### Nilai Tukar Rupiah (IDR/USD)")
fig_idr = go.Figure()
fig_idr.add_trace(
    go.Scatter(x=simulated_df.index, y=simulated_df['IDR_USD'], name='IDR/USD Rate', line=dict(color='#008000', width=1.5)), 
)

fig_idr.update_layout(
    height=450,
    title_text=f"Agregasi {aggregation_freq} ({selected_start_date.strftime('%Y-%m-%d')} s/d {selected_end_date.strftime('%Y-%m-%d')})",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="right", x=1),
)

fig_idr.update_yaxes(title_text="Nilai Tukar (IDR/USD)", tickprefix="Rp", tickformat=",0f")

# Range Selector untuk Rupiah
fig_idr.update_xaxes(
    type='date',
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1B", step="month", stepmode="backward"),
            dict(count=6, label="6B", step="month", stepmode="backward"),
            dict(count=1, label="1T", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

st.plotly_chart(fig_idr, use_container_width=True)


st.markdown("---") 
st.subheader(f"Analisis Pergerakan {aggregation_freq} (Persentase)")

# Hitung ulang perubahan persentase untuk grafik di bawah (selain mode Harian)
if aggregation_freq != 'Harian':
    df_chart = simulated_df.copy()
    df_chart['Gold_Change_Pct'] = df_chart['Gold_USD'].pct_change() * 100
    df_chart['IDR_Change_Pct'] = df_chart['IDR_USD'].pct_change() * 100
    df_chart = df_chart.dropna()
else:
    # Untuk harian, kita menggunakan raw_df untuk perubahan % (setelah fetch, sebelum agregasi)
    # Karena raw_df tidak memiliki Gold_Change_Pct, kita hitung di sini
    df_chart = raw_df.copy()
    df_chart['Gold_Change_Pct'] = df_chart['Gold_USD'].pct_change() * 100
    df_chart['IDR_Change_Pct'] = df_chart['IDR_USD'].pct_change() * 100
    df_chart = df_chart.dropna()


fig2 = make_subplots(specs=[[{"secondary_y": True}]])

# Perubahan Emas Harian/Agregasi
fig2.add_trace(
    go.Bar(x=df_chart.index, y=df_chart['Gold_Change_Pct'], name=f'Emas (% per {aggregation_freq})', opacity=0.8, marker_color=np.where(df_chart['Gold_Change_Pct'] > 0, 'green', 'red')),
    secondary_y=False
)

# Perubahan Rupiah Harian/Agregasi
fig2.add_trace(
    go.Scatter(x=df_chart.index, y=df_chart['IDR_Change_Pct'], name=f'Rupiah (% per {aggregation_freq})', mode='lines', line=dict(color='orange', width=2)), 
    secondary_y=True
)

fig2.update_yaxes(title_text="Emas (% Perubahan)", secondary_y=False)
fig2.update_yaxes(title_text="Rupiah (% Perubahan)", secondary_y=True)

fig2.update_layout(
    title_text=f"Perbandingan Perubahan {aggregation_freq} Emas vs Rupiah",
    height=500,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("---") 
st.caption("⚠️ **Penting:** Data Emas diambil dari Stooq. Data Rupiah diambil dari Google Sheets. Pastikan **Spreadsheet ID** di sidebar valid dan Sheets sudah diatur *sharing*-nya.")
