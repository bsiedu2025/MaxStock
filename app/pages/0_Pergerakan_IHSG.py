# D:\Docker\BrokerSummary\app\pages\6_Pergerakan_IHSG.py
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Pergerakan IHSG", layout="wide")
st.title("🇮🇩 Pergerakan Indeks Harga Saham Gabungan (IHSG)")
st.markdown("Data historis IHSG (`^JKSE`) dari Yahoo Finance.")

# Ticker untuk IHSG
TICKER_IHSG = "^JKSE"

# --- Fungsi untuk mengambil data saham/indeks ---
@st.cache_data(ttl=1800) # Cache data selama 30 menit untuk indeks
def get_index_data(ticker, start_date, end_date):
    """Mengambil data indeks historis menggunakan yfinance."""
    try:
        idx = yf.Ticker(ticker)
        # Tambah 1 hari ke end_date karena yfinance tidak inklusif untuk end_date
        hist_data = idx.history(start=start_date, end=end_date + timedelta(days=1), auto_adjust=True)
        if hist_data.empty:
            st.warning(f"Tidak ada data yang ditemukan untuk {ticker} pada rentang tanggal yang dipilih.")
            return pd.DataFrame()
        hist_data.index = pd.to_datetime(hist_data.index).normalize()
        return hist_data
    except Exception as e:
        st.error(f"Error saat mengambil data untuk {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300) # Cache info selama 5 menit
def get_index_quick_info(ticker_symbol):
    """Mengambil informasi ringkas indeks menggunakan yfinance."""
    try:
        idx = yf.Ticker(ticker_symbol)
        info = idx.info
        # Fallback sederhana jika info utama kurang lengkap
        if not info or info.get('regularMarketPrice') is None and info.get('currentPrice') is None:
            # st.toast(f"Info utama untuk {ticker_symbol} kurang lengkap, mencoba fallback histori singkat...", icon="⚠️")
            fallback_hist = idx.history(period="5d", auto_adjust=True)
            if not fallback_hist.empty:
                if not isinstance(info, dict): info = {}
                info['regularMarketPrice'] = fallback_hist['Close'].iloc[-1]
                info['previousClose'] = fallback_hist['Close'].iloc[-2] if len(fallback_hist) > 1 else info.get('regularMarketPrice')
                info['volume'] = fallback_hist['Volume'].iloc[-1] # Volume untuk indeks mungkin 0 atau tidak relevan
                info['dayHigh'] = fallback_hist['High'].iloc[-1]
                info['dayLow'] = fallback_hist['Low'].iloc[-1]
            else:
                # st.warning(f"Gagal mendapatkan info dan data histori fallback untuk {ticker_symbol}.")
                if not isinstance(info, dict): info = {}
        return info if isinstance(info, dict) else {}
    except Exception as e:
        # st.error(f"Error saat mengambil informasi ringkas untuk {ticker_symbol} dari yfinance: {e}")
        return {}


# --- Sidebar untuk Filter Tanggal ---
st.sidebar.header("Filter Data IHSG")
default_end_date_ihsg = datetime.now().date()
default_start_date_ihsg = default_end_date_ihsg - timedelta(days=365 * 1) # Default 1 tahun terakhir

selected_start_date_ihsg = st.sidebar.date_input(
    "Tanggal Mulai Data IHSG",
    value=default_start_date_ihsg,
    max_value=default_end_date_ihsg - timedelta(days=1),
    key="ihsg_start_date"
)

selected_end_date_ihsg = st.sidebar.date_input(
    "Tanggal Akhir Data IHSG",
    value=default_end_date_ihsg,
    min_value=selected_start_date_ihsg + timedelta(days=1),
    max_value=default_end_date_ihsg,
    key="ihsg_end_date"
)

# --- Ambil Informasi Ringkas IHSG ---
ihsg_info = get_index_quick_info(TICKER_IHSG)

if ihsg_info:
    st.subheader("Ringkasan IHSG Terkini")
    latest_price = ihsg_info.get('regularMarketPrice', ihsg_info.get('currentPrice', None))
    previous_close = ihsg_info.get('previousClose', None)
    price_change, percentage_change = None, None

    if latest_price is not None and previous_close is not None:
        price_change = latest_price - previous_close
        percentage_change = (price_change / previous_close) * 100 if previous_close != 0 else 0

    volume_today = ihsg_info.get('volume', 0) # Volume untuk indeks seringkali 0 atau tidak terlalu bermakna
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="IHSG Terkini", value=f"{latest_price:,.2f}" if latest_price is not None else "N/A")
    with col2:
        if price_change is not None:
            st.metric(label="Perubahan", value=f"{price_change:,.2f}", delta=f"{percentage_change:.2f}%" if percentage_change is not None else None)
        else:
            st.metric(label="Perubahan", value="N/A")
    with col3:
        st.metric(label="Volume Hari Ini", value=f"{volume_today:,.0f}" if volume_today is not None else "N/A", help="Volume untuk indeks mungkin tidak selalu representatif.")
    st.markdown("---")
else:
    st.warning(f"Tidak dapat mengambil informasi ringkasan terkini untuk IHSG ({TICKER_IHSG}).")


# --- Proses dan Tampilkan Data Historis ---
if selected_start_date_ihsg and selected_end_date_ihsg:
    if selected_start_date_ihsg >= selected_end_date_ihsg:
        st.sidebar.error("Tanggal Mulai harus sebelum Tanggal Akhir.")
    else:
        with st.spinner(f"Mengambil data historis IHSG ({TICKER_IHSG})..."):
            ihsg_data_df = get_index_data(TICKER_IHSG, selected_start_date_ihsg, selected_end_date_ihsg)

        if not ihsg_data_df.empty:
            st.subheader(f"Data Historis IHSG ({selected_start_date_ihsg.strftime('%d %b %Y')} - {selected_end_date_ihsg.strftime('%d %b %Y')})")
            
            # Format tampilan untuk DataFrame
            ihsg_data_display = ihsg_data_df.copy()
            ihsg_data_display.index = ihsg_data_df.index.strftime('%Y-%m-%d')
            format_numbers_ihsg = {
                'Open': '{:,.2f}', 'High': '{:,.2f}', 'Low': '{:,.2f}', 
                'Close': '{:,.2f}', 'Volume': '{:,.0f}'
            }
            st.dataframe(ihsg_data_display.tail().sort_index(ascending=False).style.format(format_numbers_ihsg, na_rep="N/A"))
            st.markdown("---")

            # --- Visualisasi Gabungan Harga Penutupan & Volume ---
            st.subheader("Grafik Harga Penutupan dan Volume Transaksi IHSG")
            fig_combined_ihsg = make_subplots(specs=[[{"secondary_y": True}]])
            fig_combined_ihsg.add_trace(
                go.Scatter(x=ihsg_data_df.index, y=ihsg_data_df['Close'], name="IHSG (Close)", mode='lines'),
                secondary_y=False,
            )
            fig_combined_ihsg.add_trace(
                go.Bar(x=ihsg_data_df.index, y=ihsg_data_df['Volume'], name="Volume", opacity=0.6),
                secondary_y=True,
            )
            fig_combined_ihsg.update_layout(
                title_text=f'Pergerakan Harga Penutupan & Volume IHSG',
                xaxis_title='Tanggal',
                xaxis_rangeslider_visible=True
            )
            fig_combined_ihsg.update_yaxes(title_text="<b>Nilai Indeks IHSG</b>", secondary_y=False, tickformat=",.0f")
            fig_combined_ihsg.update_yaxes(title_text="<b>Volume</b>", secondary_y=True, tickformat=",.0s") # Format volume dengan suffix (K, M, B)
            st.plotly_chart(fig_combined_ihsg, use_container_width=True)
            st.markdown("---")

            # --- Candlestick Chart ---
            st.subheader("Grafik Candlestick IHSG")
            fig_candlestick_ihsg = go.Figure(data=[go.Candlestick(x=ihsg_data_df.index,
                                                               open=ihsg_data_df['Open'],
                                                               high=ihsg_data_df['High'],
                                                               low=ihsg_data_df['Low'],
                                                               close=ihsg_data_df['Close'],
                                                               name=TICKER_IHSG)])
            fig_candlestick_ihsg.update_layout(
                title=f'Candlestick Chart IHSG',
                xaxis_title='Tanggal',
                yaxis_title='Nilai Indeks',
                xaxis_rangeslider_visible=True
            )
            fig_candlestick_ihsg.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig_candlestick_ihsg, use_container_width=True)

        else:
            st.info("Tidak ada data historis IHSG untuk ditampilkan dengan filter yang dipilih.")
else:
    st.info("Silakan pilih rentang tanggal yang valid di sidebar untuk melihat data historis IHSG.")

st.sidebar.markdown("---")
st.sidebar.caption("Data IHSG diambil dari Yahoo Finance.")

