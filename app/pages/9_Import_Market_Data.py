# app/pages/9_Import_Market_Data.py
import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
import os
from datetime import datetime

# PERBAIKAN: Import fungsi yang baru kita pastikan ada di db_utils
from db_utils import insert_daily_market_data 

st.set_page_config(page_title="📥 Import Data Market Harian", layout="wide")
st.title("📥 Import Data Market Harian")
st.markdown("Halaman untuk memasukkan data ringkasan harian (Ringkasan Saham) ke tabel `daily_stock_market_data`.")

# --- Daftar Kolom yang Diharapkan dari File CSV/Excel (Harus Cocok dengan file lo)
# Digunakan untuk validasi
EXPECTED_COLS = {
    'KODE_SAHAM': 'Ticker',
    'TANGGAL_PERDAGANGAN_TERAKHIR': 'Tanggal',
    'PENUTUPAN': 'Close Price',
    'VOLUME': 'Volume',
    'NILAI': 'Nilai',
    'FOREIGN_SELL': 'Foreign Sell',
    'FOREIGN_BUY': 'Foreign Buy'
    # Kolom harga dan volume lainnya juga harus ada
}

# --- Fungsi Bantuan untuk Memproses File ---
def process_uploaded_file(uploaded_file):
    """Membaca file, membersihkan header, dan menampilkan preview."""
    try:
        if uploaded_file.name.endswith('.xlsx'):
             # Asumsi Streamlit support membaca Excel jika library seperti openpyxl terinstall
             df = pd.read_excel(uploaded_file, sheet_name=0)
        else: # Asumsi CSV
             data = uploaded_file.getvalue().decode("utf-8")
             df = pd.read_csv(StringIO(data))
        
        # 1. Bersihkan Nama Kolom
        original_cols = list(df.columns)
        df.columns = df.columns.str.strip().str.replace(' ', '_', regex=False).str.replace('.', '', regex=False).str.upper()
        
        # 2. Cek Kolom Wajib
        required_cols = list(EXPECTED_COLS.keys())
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
             st.error(f"Kolom wajib tidak ditemukan di file. Pastikan header sesuai. Missing: {', '.join(missing_cols)}")
             st.write("Daftar Kolom di File Anda:", original_cols)
             return None

        # 3. Konversi Tanggal dan Bersihkan data
        df['TANGGAL_PERDAGANGAN_TERAKHIR'] = pd.to_datetime(df['TANGGAL_PERDAGANGAN_TERAKHIR'], errors='coerce')
        df = df.dropna(subset=['KODE_SAHAM', 'TANGGAL_PERDAGANGAN_TERAKHIR'])
        
        return df
        
    except Exception as e:
        st.error(f"Gagal memproses file: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────

# --- UI Single/Bulk Import ---

tab1, tab2 = st.tabs(["Import File Tunggal", "Import Banyak File Sekaligus"])

with tab1:
    st.subheader("Import Single File")
    uploaded_file = st.file_uploader(
        "Pilih file CSV/Excel Ringkasan Harian:",
        type=['csv', 'xlsx'],
        key="single_uploader"
    )

    if uploaded_file is not None:
        df_preview = process_uploaded_file(uploaded_file)
        
        if df_preview is not None:
            st.success(f"File **{uploaded_file.name}** berhasil dimuat. Ditemukan {len(df_preview)} baris data valid.")
            st.write("Pratinjau 5 baris pertama (sebelum disimpan):")
            st.dataframe(df_preview[list(EXPECTED_COLS.keys())].head(), use_container_width=True)
            
            if st.button("🚀 Simpan Data ke Database", type="primary", key="save_single_button"):
                with st.spinner("Memproses dan menyimpan data..."):
                    rows_affected = insert_daily_market_data(df_preview)
                    if rows_affected > 0:
                        st.success(f"✅ Berhasil menyimpan/mengupdate {rows_affected} baris data ke `daily_stock_market_data`!")
                        st.balloons()
                    else:
                        st.warning("⚠️ Proses selesai, tetapi tidak ada baris yang diubah (mungkin data sudah ada).")

with tab2:
    st.subheader("Import Bulk Files")
    uploaded_files = st.file_uploader(
        "Pilih semua file CSV/Excel Ringkasan Harian (Bulk Import):",
        type=['csv', 'xlsx'],
        accept_multiple_files=True,
        key="bulk_uploader"
    )

    if uploaded_files:
        if st.button(f"🚀 Simpan {len(uploaded_files)} File Sekaligus", type="primary", key="save_bulk_button"):
            total_rows_affected = 0
            
            with st.status("Memulai Bulk Import...", expanded=True) as status:
                for i, file in enumerate(uploaded_files):
                    status.write(f"[{i+1}/{len(uploaded_files)}] Memproses file: **{file.name}**")
                    
                    df_to_save = process_uploaded_file(file)
                    
                    if df_to_save is not None and not df_to_save.empty:
                        rows = insert_daily_market_data(df_to_save)
                        if rows > 0:
                            total_rows_affected += rows
                            status.success(f"   -> Berhasil menyimpan/mengupdate {rows} baris.")
                        elif rows == 0:
                            status.info("   -> Data sudah ada, 0 baris diubah.")
                        else:
                             status.error("   -> Gagal menyimpan. Cek log DB.")
                    elif df_to_save is not None and df_to_save.empty:
                        status.warning("   -> File kosong setelah filter/pembersihan data.")
                    else:
                        status.error(f"   -> Gagal memproses file {file.name}. Lihat error di atas.")

                status.update(label=f"🎉 Bulk Import Selesai! Total {total_rows_affected} baris terpengaruh.", state="complete", expanded=False)
            
            if total_rows_affected > 0:
                 st.success("🎉 Bulk Import Selesai Total Data Berhasil Diupdate!")
            else:
                 st.info("Bulk Import Selesai, Tidak Ada Data Baru yang Disimpan.")
