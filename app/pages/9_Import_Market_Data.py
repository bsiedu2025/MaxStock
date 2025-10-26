# app/pages/9_Import_Market_Data.py
# Upload data harian pergerakan asing (CSV/XLSX) → data_harian
# Mendukung import satuan dan bulk. Aman dari duplikasi (Primary Key: kode_saham, tanggal_perdagangan).

import io
import os
import math
import tempfile
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import streamlit as st
import mysql.connector
from mysql.connector import pooling
from mysql.connector.errors import IntegrityError

# Asumsi: db_utils.py dan fungsi get_pool() tersedia di path
try:
    # Memuat utilitas DB dari db_utils.py atau dari 7_Import_KSEI_Bulanan.py
    # Karena saya tidak dapat memuat db_utils.py secara langsung, saya akan mendefinisikan
    # fungsi dasar koneksi DB di sini untuk contoh yang self-contained, mengikuti pola 7_Import_KSEI_Bulanan.py.
    # Namun, idealnya Anda harus mengimportnya dari db_utils atau menggunakan yang sudah ada.
    
    # -------------------------------------------------------------------------
    # HACK: Definisikan ulang fungsi get_pool() jika tidak bisa diimport
    # GANTI INI DENGAN IMPORT YANG SESUAI JIKA MENGGUNAKAN STRUKTUR ASLI
    # -------------------------------------------------------------------------
    def _get_db_params():
        # Ambil konfigurasi DB dari Streamlit secrets atau environment variables
        return {
            "host": os.getenv("DB_HOST", st.secrets.get("DB_HOST", "")),
            "port": int(os.getenv("DB_PORT", st.secrets.get("DB_PORT", 3306))),
            "database": os.getenv("DB_NAME", st.secrets.get("DB_NAME", "")),
            "user": os.getenv("DB_USER", st.secrets.get("DB_USER", "")),
            "password": os.getenv("DB_PASSWORD", st.secrets.get("DB_PASSWORD", "")),
            "ssl_ca_str": os.getenv("DB_SSL_CA", st.secrets.get("DB_SSL_CA", "")),
        }

    @st.cache_resource
    def get_pool():
        params = _get_db_params()
        
        # Penanganan SSL CA seperti di 7_Import_KSEI_Bulanan.py
        ssl_ca_file = None
        ssl_kwargs = {}
        if params["ssl_ca_str"] and "BEGIN CERTIFICATE" in params["ssl_ca_str"]:
            # Simpan sertifikat ke file sementara
            try:
                temp_dir = tempfile.gettempdir()
                ssl_ca_file = os.path.join(temp_dir, "db_ca.pem")
                with open(ssl_ca_file, "w") as f:
                    f.write(params["ssl_ca_str"])
                
                # Tambahkan SSL kwargs
                ssl_kwargs = {
                    "ssl_ca": ssl_ca_file,
                    "ssl_verify_cert": True, # Diperlukan saat menggunakan ssl_ca
                }
            except Exception as e:
                st.error(f"Gagal menyiapkan sertifikat SSL: {e}")
                
        try:
            pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="mypool",
                pool_size=5,
                **params,
                **ssl_kwargs
            )
            return pool, ssl_ca_file # Kembalikan file CA untuk dihapus nanti
        except Exception as e:
            st.error(f"Gagal membuat koneksi pool: {e}")
            return None, None
    # -------------------------------------------------------------------------

    # Ambil pool dan ssl_ca_file (untuk dihapus nanti)
    db_pool, ssl_ca_file = get_pool()
except Exception as e:
    db_pool = None
    ssl_ca_file = None
    st.error(f"Gagal menginisialisasi koneksi database. Pastikan 'db_utils.py' atau konfigurasi DB sudah benar: {e}")


st.set_page_config(page_title="📥 Import Data Harian", page_icon="📈", layout="wide")
st.title("📥 Import Data Harian Saham & Asing → `data_harian`")
st.markdown("Unggah file CSV/XLSX (format `Ringkasan Saham-20251003.csv`) untuk diimpor ke tabel `data_harian`. Data duplikat (berdasarkan Kode Saham dan Tanggal) akan diabaikan.")


# Nama-nama kolom dari file sumber ke nama kolom di tabel DB
COL_MAP = {
    'No': 'no',
    'Kode Saham': 'kode_saham',
    'Nama Perusahaan': 'nama_perusahaan',
    'Remarks': 'remarks',
    'Sebelumnya': 'sebelumnya',
    'Open Price': 'open_price',
    'Tanggal Perdagangan Terakhir': 'tanggal_perdagangan',
    'First Trade': 'first_trade',
    'Tertinggi': 'tertinggi',
    'Terendah': 'terendah',
    'Penutupan': 'penutupan',
    'Selisih': 'selisih',
    'Volume': 'volume',
    'Nilai': 'nilai',
    'Frekuensi': 'frekuensi',
    'Index Individual': 'index_individual',
    'Offer': 'offer',
    'Offer Volume': 'offer_volume',
    'Bid': 'bid',
    'Bid Volume': 'bid_volume',
    'Listed Shares': 'listed_shares',
    'Tradeble Shares': 'tradeble_shares',
    'Weight For Index': 'weight_for_index',
    'Foreign Sell': 'foreign_sell',
    'Foreign Buy': 'foreign_buy',
    'Non Regular Volume': 'non_regular_volume',
    'Non Regular Value': 'non_regular_value',
    'Non Regular Frequency': 'non_regular_frequency',
}


def _process_dataframe(df: pd.DataFrame, source_filename: str) -> Optional[pd.DataFrame]:
    """Bersihkan, ubah nama kolom, dan konversi tipe data DataFrame."""
    try:
        # 1. Ubah nama kolom
        df.columns = [COL_MAP.get(col, col) for col in df.columns]

        # Filter hanya kolom yang dibutuhkan (yang ada di COL_MAP.values())
        valid_cols = list(COL_MAP.values())
        if 'source_file' not in valid_cols:
             valid_cols.append('source_file')
             
        df = df[[col for col in df.columns if col in valid_cols]]
        
        # 2. Pembersihan dan Konversi
        
        # Konversi tanggal
        df['tanggal_perdagangan'] = pd.to_datetime(df['tanggal_perdagangan'], errors='coerce')
        
        # Hapus baris dengan tanggal atau kode saham yang hilang
        df.dropna(subset=['tanggal_perdagangan', 'kode_saham'], inplace=True)
        
        # Konversi tipe data numerik (mengatasi separator desimal/ribuan jika ada)
        numeric_cols = [c for c in df.columns if c not in ['kode_saham', 'nama_perusahaan', 'remarks', 'tanggal_perdagangan', 'source_file']]
        for col in numeric_cols:
            # Gunakan pd.to_numeric untuk penanganan yang lebih baik
            # errors='coerce' akan mengubah nilai yang tidak dapat dikonversi menjadi NaN
            df[col] = pd.to_numeric(df[col], errors='coerce') 
            df[col].fillna(0, inplace=True) # Isi NaN dengan 0
            
        # Pastikan Kode Saham huruf kapital dan tanpa spasi
        df['kode_saham'] = df['kode_saham'].astype(str).str.upper().str.strip()

        # Tambahkan nama file sumber
        df['source_file'] = source_filename

        return df

    except Exception as e:
        st.error(f"Kesalahan saat memproses data: {e}")
        st.dataframe(df.head())
        return None


def _import_file_to_db(df: pd.DataFrame, conn: mysql.connector.MySQLConnection, source_filename: str) -> int:
    """Mengimpor data dari DataFrame ke tabel data_harian dengan INSERT IGNORE."""
    table_name = "data_harian"
    
    # Kolom untuk SQL (sama dengan urutan di DataFrame yang sudah diproses)
    cols = list(df.columns)
    cols_str = ", ".join(cols)
    placeholders = ", ".join(["%s"] * len(cols))
    
    # Gunakan INSERT IGNORE untuk menghindari duplikasi (Primary Key: kode_saham, tanggal_perdagangan)
    sql = f"INSERT IGNORE INTO {table_name} ({cols_str}) VALUES ({placeholders})"
    
    # Ubah DataFrame menjadi list of tuples
    data_to_insert = [tuple(r) for r in df.itertuples(index=False, name=None)]
    total_rows = len(data_to_insert)
    inserted_total = 0

    if total_rows == 0:
        return 0

    st.info(f"Mencoba insert {total_rows:,} baris dari file: {source_filename}...")
    
    try:
        cur = conn.cursor()
        
        batch_size = 5000  # Ukuran batch untuk performa
        total_batches = math.ceil(total_rows / batch_size)
        pbar = st.progress(0, text=f"Menyimpan ke database (Batch 0/{total_batches})...")

        for b in range(total_batches):
            start = b * batch_size
            end = min(start + batch_size, total_rows)
            batch = data_to_insert[start:end]
            
            # Eksekusi batch
            cur.executemany(sql, batch)
            
            # Catat jumlah baris yang benar-benar terinsert
            # MySQL connector hanya mengembalikan 0 untuk INSERT IGNORE yang diabaikan.
            # Kita akan mengandalkan cur.rowcount untuk jumlah baris yang diproses (terinsert/diabaikan)
            # Karena sulit mendapatkan count IGNORED, kita hanya laporkan total baris yang dicoba.
            inserted_total += cur.rowcount
            
            conn.commit()
            pbar.progress((b + 1) / total_batches, text=f"Menyimpan batch {b+1}/{total_batches}...")

        cur.close()
        pbar.empty()
        
        # Karena kita tidak bisa mendapatkan rowcount yang sukses untuk INSERT IGNORE dengan mudah, 
        # kita laporkan total baris yang diproses.
        st.success(f"Selesai. Total baris dalam file: {total_rows:,}. Baris yang diimpor/diperbarui: {inserted_total:,}. Baris duplikat diabaikan.")
        return total_rows
        
    except Exception as e:
        conn.rollback()
        st.error(f"Terjadi kesalahan saat menyimpan data ke DB: {e}")
        return 0


def main_upload_processor(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], is_bulk: bool):
    """Fungsi utama untuk memproses unggahan file."""
    if not db_pool:
        st.warning("Koneksi database tidak tersedia. Mohon cek konfigurasi DB Anda.")
        return

    total_processed = 0
    
    # Dapatkan koneksi dari pool
    conn = None
    try:
        conn = db_pool.get_connection()
        
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            st.subheader(f"Memproses file: {file_name}")

            # 1. Baca File
            try:
                if file_name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
                elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error(f"Format file '{file_name}' tidak didukung. Hanya CSV/XLSX.")
                    continue
                
                # Cek apakah DataFrame kosong
                if df.empty:
                    st.warning(f"File '{file_name}' kosong.")
                    continue

            except Exception as e:
                st.error(f"Gagal membaca file '{file_name}': {e}")
                continue

            # 2. Proses DataFrame
            processed_df = _process_dataframe(df, file_name)
            if processed_df is None:
                st.error(f"Gagal memproses data dalam file '{file_name}'.")
                continue
            
            # Tampilkan pratinjau data yang sudah diproses
            if not is_bulk:
                st.info("Pratinjau data yang sudah diproses (5 baris pertama):")
                st.dataframe(processed_df.head())

            # 3. Impor ke DB
            rows_imported = _import_file_to_db(processed_df, conn, file_name)
            total_processed += rows_imported

    except Exception as e:
        st.error(f"Kesalahan koneksi database: {e}")
    finally:
        # Tutup koneksi dan hapus file SSL CA jika ada
        if conn and conn.is_connected():
            conn.close()
        
        if ssl_ca_file and os.path.exists(ssl_ca_file):
            try: os.unlink(ssl_ca_file) 
            except Exception: pass
            
    if total_processed > 0:
        st.balloons()
        st.metric(label="Total Baris Data Diproses", value=f"{total_processed:,}")


# --- Tampilan Streamlit ---

col1, col2 = st.columns(2)

with col1:
    st.header("Upload Satuan (Single File)")
    uploaded_file = st.file_uploader(
        "Pilih satu file (CSV/XLSX)",
        type=['csv', 'xlsx'],
        accept_multiple_files=False,
        key='single_upload'
    )
    if uploaded_file and st.button("Proses Upload Satuan", key='btn_single'):
        with st.spinner(f"Memproses {uploaded_file.name}..."):
            main_upload_processor([uploaded_file], is_bulk=False)

with col2:
    st.header("Upload Bulk (Multi-file)")
    uploaded_files_bulk = st.file_uploader(
        "Pilih banyak file (CSV/XLSX)",
        type=['csv', 'xlsx'],
        accept_multiple_files=True,
        key='bulk_upload'
    )
    if uploaded_files_bulk and st.button(f"Proses Upload Bulk ({len(uploaded_files_bulk)} file)", key='btn_bulk'):
        with st.spinner(f"Memproses {len(uploaded_files_bulk)} file..."):
            main_upload_processor(uploaded_files_bulk, is_bulk=True)

st.divider()

# Contoh struktur kolom untuk referensi
st.subheader("Struktur Data yang Diharapkan")
st.markdown("""
Pastikan file CSV/XLSX Anda memiliki kolom-kolom berikut (tidak harus berurutan):
- `No`
- `Kode Saham`
- `Nama Perusahaan`
- `Remarks`
- `Sebelumnya`
- `Open Price`
- `Tanggal Perdagangan Terakhir` (Penting: Digunakan sebagai Primary Key bersama Kode Saham)
- `First Trade`
- `Tertinggi`, `Terendah`, `Penutupan`, `Selisih`
- `Volume`, `Nilai`, `Frekuensi`, `Index Individual`
- `Offer`, `Offer Volume`, `Bid`, `Bid Volume`
- `Listed Shares`, `Tradeble Shares`, `Weight For Index`
- `Foreign Sell`, `Foreign Buy` (Data pergerakan asing)
- `Non Regular Volume`, `Non Regular Value`, `Non Regular Frequency`
""")
