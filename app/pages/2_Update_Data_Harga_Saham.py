# D:\Docker\BrokerSummary\app\pages\4_Kelola_Data_Harga_Saham.py
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from db_utils import (
    insert_stock_price_data,
    fetch_stock_prices_from_db,
    create_tables_if_not_exist,
    get_saved_tickers_summary,
    execute_query # [BARU] Import execute_query untuk mengambil list ticker
)

st.set_page_config(page_title="Kelola Data Harga Saham", layout="wide")
st.title("🛠️ Kelola Data Historis Harga Saham")
st.markdown("Gunakan halaman ini untuk mengunduh dan menyimpan data harga saham ke database.")

# --- Fungsi Bantuan Baru ---
@st.cache_data(ttl=600)
def get_distinct_tickers_from_price_history_with_suffix():
    """Mengambil daftar ticker unik dari tabel stock_prices_history, mempertahankan suffix .JK"""
    query = "SELECT DISTINCT Ticker FROM stock_prices_history ORDER BY Ticker ASC;"
    result, error = execute_query(query, fetch_all=True)
    if error or not result:
        return []
    return [row['Ticker'] for row in result if row['Ticker']]

# --- Menampilkan Ringkasan Saham yang Sudah Tersimpan ---
st.subheader("Ringkasan Saham Tersimpan di Database")

@st.cache_data(ttl=300)
def load_tickers_summary_cached():
    return get_saved_tickers_summary()

if st.button("🔄 Refresh Ringkasan Saham Tersimpan", key="refresh_summary_button_page4"):
    st.cache_data.clear()

summary_df = load_tickers_summary_cached()

if not summary_df.empty:
    # (Kode untuk menampilkan summary_df tidak berubah)
    summary_df_display = summary_df.copy()
    summary_df_display['Jumlah_Data'] = summary_df_display['Jumlah_Data'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
    summary_df_display['Tanggal_Awal'] = pd.to_datetime(summary_df_display['Tanggal_Awal']).dt.strftime('%Y-%m-%d')
    summary_df_display['Tanggal_Terakhir'] = pd.to_datetime(summary_df_display['Tanggal_Terakhir']).dt.strftime('%Y-%m-%d')
    price_cols_to_format = ['Harga_Penutupan_Terakhir', 'Harga_Tertinggi_Periode', 'Harga_Terendah_Periode']
    for col in price_cols_to_format:
        if col in summary_df_display.columns:
            summary_df_display[col] = summary_df_display[col].apply(lambda x: f"{float(x):,.0f}" if pd.notna(x) else "N/A")
    summary_df_display.rename(columns={'Ticker': 'Simbol Saham','Jumlah_Data': 'Total Baris Data','Tanggal_Awal': 'Dari Tanggal','Tanggal_Terakhir': 'Sampai Tanggal','Harga_Penutupan_Terakhir': 'Penutupan Terakhir (IDR)','Harga_Tertinggi_Periode': 'Tertinggi Tersimpan (IDR)','Harga_Terendah_Periode': 'Terendah Tersimpan (IDR)'}, inplace=True)
    display_columns_ordered = ['Simbol Saham', 'Total Baris Data', 'Dari Tanggal', 'Sampai Tanggal', 'Penutupan Terakhir (IDR)', 'Tertinggi Tersimpan (IDR)', 'Terendah Tersimpan (IDR)']
    final_display_columns = [col for col in display_columns_ordered if col in summary_df_display.columns]
    st.dataframe(summary_df_display[final_display_columns], use_container_width=True)
else:
    st.info("Belum ada data harga saham yang tersimpan di database.")
st.markdown("---")


# --- [BARU] SEKSI UPDATE MASSAL ---
st.subheader("🔄 Update Massal (Bulk Update)")
st.markdown("Gunakan tombol di bawah ini untuk mengunduh data terbaru untuk **semua saham yang sudah ada** di database Anda.")

tickers_in_db = get_distinct_tickers_from_price_history_with_suffix()

if not tickers_in_db:
    st.info("Tidak ada saham di database untuk diupdate.")
else:
    st.info(f"Ditemukan **{len(tickers_in_db)}** saham di database yang siap untuk diupdate.")
    
    update_period = st.selectbox(
        "Pilih Periode Data untuk Diunduh per Saham:",
        options=['1mo', '6mo', '1y', '2y', '5y', 'max'],
        index=0, # Default 1 tahun, pilihan aman untuk update
        help="Periode ini akan diterapkan untuk setiap saham saat proses update massal."
    )

    if st.button(f"🚀 Mulai Update Massal untuk {len(tickers_in_db)} Saham", type="primary"):
        st.warning("Proses ini mungkin memakan waktu beberapa menit. Jangan tutup halaman ini.", icon="⚠️")
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_tickers = len(tickers_in_db)
        
        for i, ticker in enumerate(tickers_in_db):
            status_text.info(f"Mengupdate {i+1}/{total_tickers}: **{ticker}** (Periode: {update_period})")
            try:
                stock_data = yf.Ticker(ticker).history(period=update_period, auto_adjust=True)
                if not stock_data.empty:
                    columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df_to_save = stock_data[[col for col in columns_to_keep if col in stock_data.columns]].copy()
                    insert_stock_price_data(df_to_save, ticker)
                else:
                    st.toast(f"Tidak ada data baru ditemukan untuk {ticker}", icon="🤷‍♂️")
            except Exception as e:
                st.toast(f"Gagal mengupdate {ticker}: {e}", icon="❌")
            
            progress_bar.progress((i + 1) / total_tickers)

        st.cache_data.clear()
        status_text.success("🎉 SEMUA SAHAM TELAH BERHASIL DIUPDATE! 🎉 Silakan klik 'Refresh Ringkasan' di atas.")
        st.balloons()
st.markdown("---")


# --- SEKSI INPUT MANUAL (TIDAK BERUBAH) ---
st.subheader("Pilih atau Masukkan Ticker Saham Baru (Manual)")
input_ticker = st.text_input(
    "Masukkan Simbol Ticker Saham (misalnya: BRIS.JK, BBCA.JK, AAPL, GOOGL)",
    value="BRIS.JK",
    help="Gunakan format ticker yang sesuai dengan Yahoo Finance. Untuk saham Indonesia, biasanya diakhiri dengan '.JK'."
).upper()

# (Sisa dari script untuk input manual tidak ada yang berubah)
if input_ticker:
    format_numbers = {'Open': '{:,.0f}', 'High': '{:,.0f}', 'Low': '{:,.0f}', 'Close': '{:,.0f}', 'Volume': '{:,.0f}'}
    st.subheader(f"Unduh dan Simpan Data untuk {input_ticker}")
    fetch_period = st.selectbox("Pilih Periode Data untuk Diunduh:", options=['1d','5d','1mo', '3mo', '6mo', '1y', '2y', '5y', 'max', 'ytd'], index=8, help="Pilih periode data historis yang ingin diunduh. 'max' untuk semua data, 'ytd' untuk year-to-date.", key=f"fetch_period_{input_ticker.replace('.', '_')}")
    if st.button(f"Unduh & Simpan Data {input_ticker} (Periode: {fetch_period})", type="primary", key=f"download_save_{input_ticker.replace('.', '_')}"):
        with st.spinner(f"Mengunduh data historis untuk {input_ticker} (periode: {fetch_period})..."):
            try:
                stock = yf.Ticker(input_ticker)
                hist_data = stock.history(period=fetch_period, auto_adjust=True)
                if hist_data.empty:
                    st.warning(f"Tidak ada data historis yang ditemukan untuk {input_ticker} dengan periode '{fetch_period}'. Periksa kembali simbol ticker.")
                else:
                    st.success(f"Berhasil mengunduh {len(hist_data)} baris data untuk {input_ticker}.")
                    st.write("Pratinjau data yang diunduh (5 baris pertama):")
                    hist_data_preview = hist_data.head().copy()
                    hist_data_preview.index = hist_data_preview.index.strftime('%Y-%m-%d')
                    st.dataframe(hist_data_preview.style.format(format_numbers, na_rep="N/A"))
                    columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df_to_save = hist_data[[col for col in columns_to_keep if col in hist_data.columns]].copy()
                    with st.spinner(f"Menyimpan data {input_ticker} ke database..."):
                        inserted_count = insert_stock_price_data(df_to_save, input_ticker)
                        st.cache_data.clear() 
                        st.success(f"Proses penyimpanan untuk {input_ticker} selesai. Klik 'Refresh Ringkasan' di atas untuk melihat pembaruan.")
                        st.subheader(f"Ringkasan Data {input_ticker} di Database Setelah Proses")
                        df_from_db_after_insert = fetch_stock_prices_from_db(input_ticker)
                        if not df_from_db_after_insert.empty:
                            st.metric(f"Total Baris Data {input_ticker} di DB", f"{len(df_from_db_after_insert):,.0f}")
                            st.metric(f"Tanggal Data Terawal di DB", df_from_db_after_insert.index.min().strftime('%Y-%m-%d'))
                            st.metric(f"Tanggal Data Terbaru di DB", df_from_db_after_insert.index.max().strftime('%Y-%m-%d'))
                            df_from_db_after_insert_display = df_from_db_after_insert.tail().copy()
                            df_from_db_after_insert_display.index = df_from_db_after_insert_display.index.strftime('%Y-%m-%d')
                            st.dataframe(df_from_db_after_insert_display.style.format(format_numbers, na_rep="N/A"))
                        else:
                            st.info(f"Belum ada data {input_ticker} di database atau gagal mengambil setelah insert.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat mengunduh atau memproses data {input_ticker}: {e}")
                st.error("Pastikan simbol ticker valid dan Anda terhubung ke internet.")

    st.markdown("---")
    st.subheader(f"Lihat Data {input_ticker} yang Tersimpan di Database")
    col1_view, col2_view = st.columns(2)
    with col1_view:
        view_start_date = st.date_input("Tanggal Mulai Lihat Data DB", datetime.now().date() - timedelta(days=30), key=f"view_db_start_{input_ticker.replace('.', '_')}")
    with col2_view:
        view_end_date = st.date_input("Tanggal Akhir Lihat Data DB", datetime.now().date(), key=f"view_db_end_{input_ticker.replace('.', '_')}")
    if view_start_date and view_end_date:
        if view_start_date > view_end_date:
            st.error("Tanggal Mulai tidak boleh setelah Tanggal Akhir.")
        else:
            if st.button("Tampilkan Data dari Database", key=f"show_db_data_button_{input_ticker.replace('.', '_')}"):
                with st.spinner(f"Mengambil data {input_ticker} dari database..."):
                    df_db_view = fetch_stock_prices_from_db(input_ticker, view_start_date, view_end_date)
                    if not df_db_view.empty:
                        df_db_view_display = df_db_view.sort_index(ascending=False).copy()
                        df_db_view_display.index = df_db_view_display.index.strftime('%Y-%m-%d')
                        st.dataframe(df_db_view_display.style.format(format_numbers, na_rep="N/A"))
                        st.info(f"Menampilkan {len(df_db_view)} baris data untuk {input_ticker}.")
                    else:
                        st.info(f"Tidak ada data {input_ticker} ditemukan di database untuk rentang tanggal yang dipilih.")
else:
    st.info("Silakan masukkan simbol ticker saham untuk memulai.")