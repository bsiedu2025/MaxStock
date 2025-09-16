# D:\Docker\BrokerSummary\app\pages\2_Update_Data_Harga_Saham.py
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from db_utils import (
    insert_stock_price_data,
    fetch_stock_prices_from_db,
    get_saved_tickers_summary,
    execute_query,
)

st.set_page_config(page_title="Update Data Harga Saham", layout="wide")
st.title("🛠️ Kelola Data Historis Harga Saham")
st.markdown("Gunakan halaman ini untuk mengunduh dan menyimpan data harga saham ke database.")

# --- Fungsi Bantuan Baru ---
@st.cache_data(ttl=600)
def load_tickers_summary_cached():
    """Memuat ringkasan saham yang tersimpan dari database dengan caching."""
    return get_saved_tickers_summary()

def format_numbers(val):
    """Fungsi format untuk angka di dataframe."""
    if isinstance(val, (int, float)):
        return f"{val:,.2f}"
    return val

# --- Menampilkan Ringkasan Saham yang Sudah Tersimpan ---
st.subheader("Ringkasan Saham Tersimpan di Database")

summary_df = load_tickers_summary_cached()
if not summary_df.empty:
    st.dataframe(summary_df.style.format(format_numbers, na_rep="N/A"))
else:
    st.info("Tidak ada data saham yang tersimpan di database.")

st.markdown("---")

# --- Form Input untuk Mengunduh Data Baru ---
st.subheader("Unduh dan Simpan Data Harga Saham Baru")
with st.form("download_form"):
    input_ticker = st.text_input("Masukkan Ticker Saham (contoh: BBCA.JK)", value="BBCA.JK").upper()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Tanggal Mulai", datetime.now().date() - timedelta(days=365))
    with col2:
        end_date = st.date_input("Tanggal Akhir", datetime.now().date())

    download_button = st.form_submit_button("Unduh dan Simpan Data")

    if download_button:
        if start_date > end_date:
            st.error("Tanggal Mulai tidak boleh setelah Tanggal Akhir.")
        else:
            with st.spinner(f"Mengunduh dan menyimpan data {input_ticker}..."):
                try:
                    df = yf.download(input_ticker, start=start_date, end=end_date)
                    if not df.empty:
                        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                        if insert_stock_price_data(input_ticker, df):
                            st.success(f"Data {input_ticker} berhasil diunduh dan disimpan.")
                            load_tickers_summary_cached.clear()
                        else:
                            st.error("Gagal menyimpan data ke database. Cek log Streamlit untuk detail.")
                    else:
                        st.warning(f"Tidak ada data ditemukan untuk ticker {input_ticker} pada rentang tanggal tersebut.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat mengunduh data: {e}")

st.markdown("---")

# --- Tampilan Data Historis dari Database ---
st.subheader("Tampilkan Data Historis dari Database")
if not summary_df.empty:
    input_ticker_view = st.selectbox("Pilih Ticker untuk Dilihat", summary_df['Ticker'].tolist())

    col1_view, col2_view = st.columns(2)
    with col1_view:
        view_start_date = st.date_input("Tanggal Mulai Lihat Data DB", datetime.now().date() - timedelta(days=30), key=f"view_db_start_{input_ticker_view.replace('.', '_')}")
    with col2_view:
        view_end_date = st.date_input("Tanggal Akhir Lihat Data DB", datetime.now().date(), key=f"view_db_end_{input_ticker_view.replace('.', '_')}")

    if view_start_date and view_end_date:
        if view_start_date > view_end_date:
            st.error("Tanggal Mulai tidak boleh setelah Tanggal Akhir.")
        else:
            if st.button("Tampilkan Data dari Database", key=f"show_db_data_button_{input_ticker_view.replace('.', '_')}"):
                with st.spinner(f"Mengambil data {input_ticker_view} dari database..."):
                    df_db_view = fetch_stock_prices_from_db(input_ticker_view, view_start_date, view_end_date)
                    if not df_db_view.empty:
                        df_db_view_display = df_db_view.sort_index(ascending=False).copy()
                        df_db_view_display.index = df_db_view_display.index.strftime('%Y-%m-%d')
                        st.dataframe(df_db_view_display.style.format(format_numbers, na_rep="N/A"))
                        st.info(f"Menampilkan {len(df_db_view)} baris data untuk ticker {input_ticker_view}.")
                    else:
                        st.warning("Tidak ada data ditemukan di database untuk rentang tanggal tersebut.")
else:
    st.info("Silakan unggah data saham terlebih dahulu untuk melihat ringkasannya di sini.")