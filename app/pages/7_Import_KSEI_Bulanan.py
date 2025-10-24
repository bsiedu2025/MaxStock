# app/pages/7_Import_KSEI_Bulanan.py
import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
import os
from datetime import datetime
from typing import List

# Import fungsi-fungsi DB dari db_utils.py
from db_utils import insert_ksei_month_data, insert_daily_market_data 

st.set_page_config(page_title="📥 Import Data Utama", layout="wide")
st.title("📥 Import Data Utama")
st.markdown("Gunakan halaman ini untuk memasukkan data KSEI Bulanan dan Data Market Harian.")

# --- Daftar Kolom yang Diharapkan dari File CSV/Excel Harian (Wajib ada!)
EXPECTED_DAILY_COLS = {
    'KODE_SAHAM': 'Ticker',
    'TANGGAL_PERDAGANGAN_TERAKHIR': 'Tanggal',
    'SEBELUMNYA': 'Prev. Close',
    'OPEN_PRICE': 'Open',
    'TERTINGGI': 'High',
    'TERENDAH': 'Low',
    'PENUTUPAN': 'Close',
    'SELISIH': 'Change',
    'VOLUME': 'Volume',
    'NILAI': 'Value',
    'FREKUENSI': 'Frequency',
    'INDEX_INDIVIDUAL': 'Indiv. Index',
    'LISTED_SHARES': 'Listed Shares',
    'TRADEBLE_SHARES': 'Tradeable Shares',
    'WEIGHT_FOR_INDEX': 'Weight',
    'FOREIGN_SELL': 'Foreign Sell',
    'FOREIGN_BUY': 'Foreign Buy'
}
REQUIRED_DAILY_COLS = ['KODE_SAHAM', 'TANGGAL_PERDAGANGAN_TERAKHIR', 'PENUTUPAN', 'VOLUME', 'FOREIGN_SELL', 'FOREIGN_BUY']


# -----------------------------------------------------------------------------
# Fungsi Bantuan untuk Memproses File
# -----------------------------------------------------------------------------
def process_uploaded_file(uploaded_file, required_cols: List[str]):
    """Membaca file, membersihkan header, dan melakukan validasi wajib."""
    try:
        # 1. Baca File
        if uploaded_file.name.endswith('.xlsx'):
             df = pd.read_excel(uploaded_file, sheet_name=0)
        else:
             data = uploaded_file.getvalue().decode("utf-8")
             df = pd.read_csv(StringIO(data))
        
        # 2. Bersihkan Nama Kolom (Wajib dicocokkan dengan db_utils)
        df.columns = df.columns.str.strip().str.replace(' ', '_', regex=False).str.replace('.', '', regex=False).str.upper()
        
        # DEBUG: Tampilkan header yang ditemukan (membantu debugging di Streamlit)
        st.info(f"Header file {uploaded_file.name} yang ditemukan: {list(df.columns)}")

        # 3. Cek Kolom Wajib
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             st.error(f"Kolom wajib tidak ditemukan di file. Missing: {', '.join(missing_cols)}")
             return None

        # 4. Konversi Tanggal dan Bersihkan data
        if 'TANGGAL_PERDAGANGAN_TERAKHIR' in df.columns:
            # Jika ini data Harian, konversi kolom tanggal yang benar
            df['TANGGAL_PERDAGANGAN_TERAKHIR'] = pd.to_datetime(df['TANGGAL_PERDAGANGAN_TERAKHIR'], errors='coerce')
            df = df.dropna(subset=['KODE_SAHAM', 'TANGGAL_PERDAGANGAN_TERAKHIR'])
        
        # Jika ini data KSEI Bulanan, lo perlu logika konversi tanggal KSEI lo di sini
        # (Asumsi data KSEI lo sudah aman/tidak perlu diubah di sini)
        
        return df
        
    except Exception as e:
        st.error(f"Gagal memproses file {uploaded_file.name}: {e}")
        return None

# -----------------------------------------------------------------------------
# Inisialisasi Tabs
# -----------------------------------------------------------------------------
tab_ksei, tab_daily = st.tabs(["Import Data KSEI Bulanan (Eksisting)", "Import Data Market Harian (BARU)"])

# -----------------------------------------------------------------------------
# TAB 1: Import Data KSEI Bulanan (Code Eksisting Lo)
# -----------------------------------------------------------------------------
with tab_ksei:
    st.subheader("Import File KSEI Bulanan")
    st.markdown("Digunakan untuk mengupdate data kepemilikan saham bulanan dari KSEI.")

    # (ASUMSI: Lo punya REQUIRED_KSEI_COLS yang sama dengan file asli lo)
    REQUIRED_KSEI_COLS = ['PERIOD', 'TICKER', 'CUSTODIAN_CATEGORY', 'LOCAL_FOREIGN', 'SHARES_HELD']
    
    col_single, col_bulk = st.columns(2)
    with col_single:
        st.caption("Import File Tunggal")
        uploaded_file_ksei = st.file_uploader("Pilih file KSEI Bulanan:", type=['csv', 'xlsx'], key="ksei_single_uploader")
        if uploaded_file_ksei:
            df_ksei = process_uploaded_file(uploaded_file_ksei, REQUIRED_KSEI_COLS)
            if df_ksei is not None:
                st.write(f"File **{uploaded_file_ksei.name}** dimuat. {len(df_ksei)} baris.")
                if st.button("🚀 Simpan KSEI ke DB", type="primary", key="save_ksei_single"):
                    with st.spinner("Menyimpan data KSEI..."):
                        rows_affected = insert_ksei_month_data(df_ksei) # Ganti dengan fungsi KSEI lo yang sebenarnya
                        if rows_affected > 0:
                            st.success(f"✅ Berhasil menyimpan/mengupdate {rows_affected} baris data KSEI!")
                        else:
                            st.warning("⚠️ Proses KSEI selesai, tidak ada baris yang diubah.")
    
    with col_bulk:
        st.caption("Import Banyak File Sekaligus")
        uploaded_files_ksei_bulk = st.file_uploader("Pilih file KSEI Bulanan (Bulk):", type=['csv', 'xlsx'], accept_multiple_files=True, key="ksei_bulk_uploader")
        if uploaded_files_ksei_bulk:
            if st.button(f"🚀 Mulai Bulk KSEI untuk {len(uploaded_files_ksei_bulk)} File", key="save_ksei_bulk"):
                total_ksei_rows = 0
                with st.status("Memulai Bulk Import KSEI...", expanded=True) as status:
                    for i, file in enumerate(uploaded_files_ksei_bulk):
                        df_ksei_bulk = process_uploaded_file(file, REQUIRED_KSEI_COLS)
                        if df_ksei_bulk is not None and not df_ksei_bulk.empty:
                            rows = insert_ksei_month_data(df_ksei_bulk) # Ganti dengan fungsi KSEI lo
                            if rows > 0:
                                total_ksei_rows += rows
                                status.success(f"[{i+1}/{len(uploaded_files_ksei_bulk)}] Berhasil menyimpan {rows} baris dari {file.name}.")
                            else:
                                status.info(f"[{i+1}/{len(uploaded_files_ksei_bulk)}] File {file.name} selesai, 0 baris diubah.")
                    
                    status.update(label=f"🎉 Bulk KSEI Selesai! Total {total_ksei_rows} baris terpengaruh.", state="complete")
                    if total_ksei_rows > 0: st.balloons()


# -----------------------------------------------------------------------------
# TAB 2: Import Data Market Harian (NEW)
# -----------------------------------------------------------------------------
with tab_daily:
    st.subheader("Import File Market Harian (Ringkasan Saham)")
    st.markdown("Digunakan untuk memasukkan data ringkasan harian ke tabel `daily_stock_market_data`.")
    
    col_single, col_bulk = st.columns(2)
    with col_single:
        st.caption("Import File Tunggal")
        uploaded_file = st.file_uploader("Pilih file CSV/Excel Ringkasan Harian:", type=['csv', 'xlsx'], key="daily_single_uploader")

        if uploaded_file is not None:
            df_preview = process_uploaded_file(uploaded_file, REQUIRED_DAILY_COLS)
            
            if df_preview is not None:
                st.success(f"File **{uploaded_file.name}** berhasil dimuat. Ditemukan {len(df_preview)} baris data valid.")
                st.write("Pratinjau 5 baris pertama:")
                # Tampilkan hanya kolom yang penting untuk preview
                preview_cols = [c for c in df_preview.columns if c in EXPECTED_DAILY_COLS.keys()]
                st.dataframe(df_preview[preview_cols].head(), use_container_width=True)
                
                if st.button("🚀 Simpan Data Harian ke Database", type="primary", key="save_daily_single_button"):
                    with st.spinner("Memproses dan menyimpan data..."):
                        # Panggil fungsi insert yang baru
                        rows_affected = insert_daily_market_data(df_preview)
                        if rows_affected > 0:
                            st.success(f"✅ Berhasil menyimpan/mengupdate {rows_affected} baris data ke `daily_stock_market_data`!")
                            st.balloons()
                        else:
                            st.warning("⚠️ Proses selesai, tetapi tidak ada baris yang diubah.")

    with col_bulk:
        st.caption("Import Banyak File Sekaligus")
        uploaded_files_bulk = st.file_uploader("Pilih file Ringkasan Harian (Bulk Import):", type=['csv', 'xlsx'], accept_multiple_files=True, key="daily_bulk_uploader")

        if uploaded_files_bulk:
            if st.button(f"🚀 Simpan {len(uploaded_files_bulk)} File Sekaligus", type="primary", key="save_daily_bulk_button"):
                total_rows_affected = 0
                
                with st.status("Memulai Bulk Import Market Data...", expanded=True) as status:
                    for i, file in enumerate(uploaded_files_bulk):
                        status.write(f"[{i+1}/{len(uploaded_files_bulk)}] Memproses file: **{file.name}**")
                        
                        df_to_save = process_uploaded_file(file, REQUIRED_DAILY_COLS)
                        
                        if df_to_save is not None and not df_to_save.empty:
                            rows = insert_daily_market_data(df_to_save)
                            if rows > 0:
                                total_rows_affected += rows
                                status.success(f"   -> Berhasil menyimpan/mengupdate {rows} baris.")
                            else:
                                 status.info("   -> Data sudah ada atau gagal disimpan.")
                        else:
                            status.warning("   -> File kosong atau gagal diproses.")

                    status.update(label=f"🎉 Bulk Import Selesai! Total {total_rows_affected} baris terpengaruh.", state="complete")
                
                if total_rows_affected > 0:
                     st.success("🎉 Bulk Import Market Data Selesai!")
                else:
                     st.info("Bulk Import Market Data Selesai, Tidak Ada Data Baru yang Disimpan.")
