# D:\Docker\BrokerSummary\app\pages\4_Sinyal_MACD.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from db_utils import get_distinct_tickers_from_price_history_with_suffix, fetch_stock_prices_from_db
from urllib.parse import quote_plus

st.set_page_config(page_title="Sinyal Beli MACD", layout="wide")
st.title("Sinyal Beli MACD (Histogram Hijau)")
st.markdown("Halaman ini menampilkan daftar saham dari database yang histogram MACD-nya baru saja berubah dari merah ke hijau, menandakan potensi sinyal momentum positif.")
st.markdown("---")

# --- Fungsi untuk Kalkulasi MACD (diambil dari 1_Harga_Saham.py) ---
def calculate_macd_pandas(data_series, fast_window=12, slow_window=26, signal_window=9):
    """
    Menghitung MACD, Signal Line, dan Histogram dari data harga penutupan.
    """
    if data_series is None or data_series.empty or len(data_series) < slow_window:
        empty_series = pd.Series(dtype='float64', index=data_series.index)
        return empty_series, empty_series, empty_series
    ema_fast = data_series.ewm(span=fast_window, adjust=False).mean()
    ema_slow = data_series.ewm(span=slow_window, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# --- Main Logic ---
st.info("Klik tombol di bawah untuk memindai semua saham yang tersimpan di database dan menemukan sinyal MACD terbaru.")
if st.button("🔍 Cari Sinyal MACD Terbaru", type="primary"):
    
    with st.spinner("Memindai semua saham, harap tunggu..."):
        all_tickers = get_distinct_tickers_from_price_history_with_suffix()

        if not all_tickers:
            st.warning("Belum ada data saham tersimpan di database. Silakan unduh data terlebih dahulu di halaman 'Kelola Data Harga Saham'.")
        else:
            total_tickers = len(all_tickers)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            signal_tickers = []
            
            for i, ticker in enumerate(all_tickers):
                status_text.info(f"Menganalisis {i+1}/{total_tickers}: **{ticker}**")
                
                # Mengambil data yang cukup untuk kalkulasi MACD (minimal 26+2 hari)
                # Ambil 50 hari terakhir untuk aman
                df = fetch_stock_prices_from_db(ticker, end_date=datetime.now().date(), start_date=datetime.now().date() - timedelta(days=50))
                
                if df.empty or len(df) < 27:
                    continue # Lewati jika data tidak cukup
                
                # Pastikan data diurutkan berdasarkan tanggal
                df = df.sort_index()

                # Hitung MACD
                macd_line, signal_line, histogram = calculate_macd_pandas(df['Close'])
                
                # Pastikan MACD berhasil dihitung dan ada data yang cukup
                if histogram.empty or len(histogram) < 2:
                    continue
                
                # Ambil 2 nilai histogram terakhir
                latest_hist = histogram.iloc[-1]
                prev_hist = histogram.iloc[-2]

                # Cek kondisi sinyal: histogram berubah dari negatif ke positif
                if prev_hist < 0 and latest_hist >= 0:
                    latest_close = df['Close'].iloc[-1]
                    latest_date = df.index[-1].strftime('%Y-%m-%d')
                    
                    signal_tickers.append({
                        "Ticker": ticker,
                        "Tanggal Sinyal": latest_date,
                        "Harga Penutupan": latest_close,
                        "Histogram Terakhir": latest_hist,
                        "Histogram Sebelumnya": prev_hist
                    })
                
                progress_bar.progress((i + 1) / total_tickers)
            
            progress_bar.empty()
            status_text.empty()

            if signal_tickers:
                st.success(f"🎉 Ditemukan **{len(signal_tickers)}** saham dengan sinyal MACD Buy!")
                df_signals = pd.DataFrame(signal_tickers)
                
                # PERBAIKAN: Mengubah Ticker menjadi link dengan URL yang benar
                # Nama file halaman adalah Harga_Saham.py
                df_signals['Ticker'] = [
                    f'<a href="/Harga_Saham?ticker={quote_plus(ticker)}" target="_self">{ticker}</a>'
                    for ticker in df_signals['Ticker']
                ]
                
                format_dict = {
                    'Harga Penutupan': lambda x: f"{x:,.0f}",
                    'Histogram Terakhir': lambda x: f"{x:.4f}",
                    'Histogram Sebelumnya': lambda x: f"{x:.4f}"
                }

                st.markdown(df_signals.to_html(escape=False, formatters=format_dict), unsafe_allow_html=True)
                
                st.balloons()
            else:
                st.info("🤷‍♂️ Tidak ada saham yang terdeteksi dengan sinyal MACD Buy pada data terakhir.")
                
st.markdown("---")
st.caption("Analisis ini bersifat otomatis dan hanya sebagai ilustrasi. Selalu lakukan riset pribadi sebelum mengambil keputusan investasi.")