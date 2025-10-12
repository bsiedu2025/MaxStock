# D:\Docker\BrokerSummary\app\pages\1_Harga_Saham.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from db_utils import fetch_stock_prices_from_db, get_saved_tickers_summary, get_stock_info, _build_engine, _table_exists
from sqlalchemy import text
import numpy as np

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Harga Saham (Database)", layout="wide")
st.title("💹 Pergerakan Harga & Analisis Asing")
st.markdown("Menampilkan data historis harga saham dari MariaDB, analisis teknikal, serta ringkasan Foreign Flow Bulanan (KSEI).")

# ────────────────────────────────────────────────────────────────────────────────
# DB Connection & Utility (Copied from 6_Pergerakan_Asing_FF (1).py)
# NOTE: Harus disalin agar page ini bisa berdiri sendiri.

engine = _build_engine() # Asumsi _build_engine ada di db_utils, jika tidak, harus diimport/didefinisikan di sini.

USE_KSEI = _table_exists("ksei_month")

def _date_filter_ksei(period: str, field: str) -> tuple[str, dict]:
    """Menghasilkan kondisi WHERE SQL dan params berdasarkan periode yang dipilih."""
    params = {}
    cond = ""
    if period == "ALL":
        return "", {}
    elif period.endswith("M"):
        n = int(period[:-1])
        cond = f"AND {field} >= DATE_SUB(CURDATE(), INTERVAL :n MONTH)"
        params["n"] = n
    elif period.endswith("Y"):
        n = int(period[:-1])
        cond = f"AND {field} >= DATE_SUB(CURDATE(), INTERVAL :n YEAR)"
        params["n"] = n
    return cond, params

# Helper: rangebreaks utk kompres weekend & tanggal libur/absent
# NOTE: Tidak digunakan di page ini karena datanya bulanan/harian, tapi dipertahankan jika user ingin menambah grafik harian FF
def _rangebreaks_from_dates(dates_series):
    try:
        s = pd.to_datetime(dates_series, errors="coerce").dropna()
    except Exception:
        return []
    if s.empty:
        return []
    all_days = pd.date_range(s.min().normalize(), s.max().normalize(), freq="D")
    present = pd.to_datetime(pd.Series(s.unique())).dt.normalize()
    missing = all_days.difference(present)  # termasuk hari libur bursa
    return [dict(bounds=["sat","mon"]), dict(values=missing.to_pydatetime().tolist())]

# ────────────────────────────────────────────────────────────────────────────────

# --- Fungsi Bantuan ---
def format_large_number(num_val):
    if pd.isna(num_val): return "N/A"
    num_val = float(num_val)
    if num_val >= 1_000_000_000_000: return f"{num_val / 1_000_000_000_000:,.2f} T"
    elif num_val >= 1_000_000_000: return f"{num_val / 1_000_000_000:,.2f} M"
    elif num_val >= 1_000_000: return f"{num_val / 1_000_000:.2f} Jt"
    else: return f"{num_val:,.0f}"

# --- Fungsi untuk Indikator Teknikal ---
def calculate_sma_pandas(data_series, window):
    if data_series is None or data_series.empty or len(data_series) < window: return pd.Series(dtype='float64')
    return data_series.rolling(window=window, min_periods=1).mean()

def calculate_rsi_pandas(data_series, window=14):
    if data_series is None or data_series.empty or len(data_series) < window + 1: return pd.Series(dtype='float64')
    delta = data_series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    rs = rs.replace([float('inf'), float('-inf')], float('nan')).fillna(0) 
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd_pandas(data_series, fast_window=12, slow_window=26, signal_window=9):
    if data_series is None or data_series.empty or len(data_series) < slow_window:
        empty_series = pd.Series(dtype='float64', index=data_series.index)
        return empty_series, empty_series, empty_series
    ema_fast = data_series.ewm(span=fast_window, adjust=False).mean()
    ema_slow = data_series.ewm(span=slow_window, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands_pandas(data_series, window=20, num_std_dev=2):
    if data_series is None or data_series.empty or len(data_series) < window:
        empty_series = pd.Series(dtype='float64', index=data_series.index)
        return empty_series, empty_series, empty_series
    middle_band = calculate_sma_pandas(data_series, window)
    std_dev = data_series.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return upper_band, middle_band, lower_band

def calculate_obv_pandas(close_series, volume_series):
    if close_series is None or volume_series is None or close_series.empty or volume_series.empty: return pd.Series(dtype='float64')
    price_change = close_series.diff()
    direction = pd.Series(1, index=price_change.index)
    direction[price_change < 0] = -1
    direction[price_change == 0] = 0
    obv = (volume_series * direction).cumsum()
    return obv

def calculate_stochastic_pandas(high_series, low_series, close_series, window=14, smooth_window=3):
    if high_series is None or low_series is None or close_series is None or len(close_series) < window:
        empty_series = pd.Series(dtype='float64', index=close_series.index)
        return empty_series, empty_series
    lowest_low = low_series.rolling(window=window).min()
    highest_high = high_series.rolling(window=window).max()
    k_percent = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=smooth_window).mean()
    return k_percent, d_percent

def generate_technical_signals(df):
    df_signal = df.copy()
    df_signal['SMA20'] = calculate_sma_pandas(df_signal['Close'], 20)
    df_signal['SMA50'] = calculate_sma_pandas(df_signal['Close'], 50)
    df_signal['RSI'] = calculate_rsi_pandas(df_signal['Close'], 14)
    df_signal['MACD_line'], df_signal['MACD_signal'], df_signal['MACD_hist'] = calculate_macd_pandas(df_signal['Close'])
    df_signal['BB_Upper'], df_signal['BB_Middle'], df_signal['BB_Lower'] = calculate_bollinger_bands_pandas(df_signal['Close'])
    df_signal['OBV'] = calculate_obv_pandas(df_signal['Close'], df_signal['Volume'])
    df_signal['Stoch_K'], df_signal['Stoch_D'] = calculate_stochastic_pandas(df_signal['High'], df_signal['Low'], df_signal['Close'])
    
    df_signal['Buy_Signal_Price'] = pd.Series(dtype='float64')
    buy_condition = (df_signal['SMA20'] > df_signal['SMA50']) & (df_signal['SMA20'].shift(1) <= df_signal['SMA50'].shift(1)) & (df_signal['RSI'] < 75)
    df_signal.loc[buy_condition, 'Buy_Signal_Price'] = df_signal['Low'] * 0.98
    
    df_signal['Sell_Signal_Price'] = pd.Series(dtype='float64')
    sell_condition = (df_signal['SMA20'] < df_signal['SMA50']) & (df_signal['SMA20'].shift(1) >= df_signal['SMA50'].shift(1)) & (df_signal['RSI'] > 25)
    df_signal.loc[sell_condition, 'Sell_Signal_Price'] = df_signal['High'] * 1.02
    return df_signal

@st.cache_data(ttl=300)
def load_tickers_summary_for_dropdown():
    summary = get_saved_tickers_summary()
    return summary.set_index('Ticker') if not summary.empty else pd.DataFrame()

tickers_summary_df = load_tickers_summary_for_dropdown()

if tickers_summary_df.empty:
    st.warning("Belum ada data harga saham tersimpan. Gunakan halaman 'Kelola Data Harga Saham' untuk menambahkan data.")
    st.stop()

st.sidebar.header("Pilih Saham & Filter Tanggal")

# Tentukan default ticker: BRIS.JK
DEFAULT_TICKER = "BRIS.JK"
tickers_list = tickers_summary_df.index.tolist()

# Cek apakah DEFAULT_TICKER ada di list, jika tidak, pakai yang pertama
if DEFAULT_TICKER not in tickers_list and tickers_list:
    DEFAULT_TICKER = tickers_list[0]
elif not tickers_list:
    DEFAULT_TICKER = ""

# Cek apakah ada parameter 'ticker' di URL
query_ticker = st.query_params.get("ticker")

# Tentukan indeks default untuk selectbox
default_index = 0  
if query_ticker and query_ticker in tickers_list:
    default_index = tickers_list.index(query_ticker)
elif DEFAULT_TICKER in tickers_list:
    default_index = tickers_list.index(DEFAULT_TICKER)

selected_ticker = st.sidebar.selectbox("Pilih Ticker Saham:", options=tickers_list, index=default_index, key="db_stock_ticker_select_v5_final")


if selected_ticker:
    ticker_info_from_summary = tickers_summary_df.loc[selected_ticker]
    min_date_for_ticker = pd.to_datetime(ticker_info_from_summary['Tanggal_Awal']).date()
    max_date_for_ticker = pd.to_datetime(ticker_info_from_summary['Tanggal_Terakhir']).date()
    
    # ────────────────────────────────────────────────────────────────────────────────
    # SIDEBAR FILTER UTAMA (untuk tab Harga)
    st.sidebar.subheader("Filter untuk Grafik Harga")
    default_start_date_main_charts = max(min_date_for_ticker, max_date_for_ticker - timedelta(days=365 * 1))
    selected_start_date_main = st.sidebar.date_input("Tanggal Mulai Grafik", value=default_start_date_main_charts, min_value=min_date_for_ticker, max_value=max_date_for_ticker, key=f"db_start_v5_final_{selected_ticker.replace('.', '_')}")
    selected_end_date_main = st.sidebar.date_input("Tanggal Akhir Grafik", value=max_date_for_ticker, min_value=selected_start_date_main, max_value=max_date_for_ticker, key=f"db_end_v5_final_{selected_ticker.replace('.', '_')}")

    # Sidebar Filter Tambahan untuk Tab Asing
    st.sidebar.subheader("Filter untuk Analisis Asing (KSEI)")
    period_ksei = st.sidebar.selectbox("Periode KSEI", ["1M", "3M", "6M", "1Y", "ALL"], index=3, key="ksei_period_select")
    
    # ────────────────────────────────────────────────────────────────────────────────
    # MAIN TABS
    tab_harga, tab_asing = st.tabs(["Harga Saham & Analisis Teknikal", "Asing (KSEI Bulanan)"])

    # --------------------------------------------------------------------------------
    # TAB HARGA SAHAM
    with tab_harga:
        if selected_start_date_main and selected_end_date_main and selected_start_date_main <= selected_end_date_main:
            with st.spinner(f"Mengambil & memproses data {selected_ticker}..."):
                stock_data_df_main = fetch_stock_prices_from_db(selected_ticker, selected_start_date_main, selected_end_date_main)
                
                if stock_data_df_main is not None and not stock_data_df_main.empty:
                    stock_data_with_signals = generate_technical_signals(stock_data_df_main.copy())
                else:
                    stock_data_with_signals = pd.DataFrame()

            if not stock_data_with_signals.empty:
                latest_data = stock_data_with_signals.iloc[-1]
                prev_data = stock_data_with_signals.iloc[-2] if len(stock_data_with_signals) > 1 else latest_data
                
                latest_price = latest_data.get('Close')
                price_change = latest_price - prev_data.get('Close')
                pct_change = (price_change / prev_data.get('Close')) * 100 if prev_data.get('Close') != 0 else 0
                
                arrow = "➖"; 
                if price_change > 0: arrow = "📈"
                elif price_change < 0: arrow = "📉"
                
                trend_status, trend_icon = "Netral", "↔️"
                sma20 = latest_data.get('SMA20'); sma50 = latest_data.get('SMA50')
                if pd.notna(sma20) and pd.notna(sma50):
                    if sma20 > sma50: trend_status, trend_icon = "Bullish", "🐂"
                    elif sma20 < sma50: trend_status, trend_icon = "Bearish", "🐻"

                header_cols = st.columns([0.8, 0.2]) 
                with header_cols[0]: st.header(f"Analisis Saham: {selected_ticker} {arrow}")
                with header_cols[1]: st.markdown(f"<div style='text-align: right; margin-top: 10px;'><span style='font-size: 30px;'>{trend_icon}</span><br><span style='font-size: 10px; color: grey;'>{trend_status} (SMA)</span></div>", unsafe_allow_html=True)
                st.markdown(f"Periode Data: **{selected_start_date_main.strftime('%d %b %Y')}** s/d **{selected_end_date_main.strftime('%d %b %Y')}**")
                st.markdown("---") 

                st.subheader("Ringkasan Metrik")
                stock_yf = get_stock_info(selected_ticker)
                ath_period = stock_data_with_signals['High'].max()
                atl_period = stock_data_with_signals['Low'].min()
                pe_yf = stock_yf.get('trailingPE', stock_yf.get('forwardPE'))
                mc_yf = stock_yf.get('marketCap')
                high_latest = latest_data.get('High'); low_latest = latest_data.get('Low')
                volatility_intra = ((high_latest - low_latest) / low_latest) * 100 if pd.notna(low_latest) and low_latest != 0 else 0
                sma5 = calculate_sma_pandas(stock_data_with_signals['Close'], 5).iloc[-1]
                
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric(label="Harga Terakhir (IDR)", value=f"{latest_price:,.0f}")
                    st.metric(label="Vol. Hari Terakhir (Jt Lbr)", value=f"{latest_data.get('Volume', 0) / 1_000_000:.2f}")
                    st.metric(label="P/E (yfinance)", value=f"{pe_yf:.2f}" if pd.notna(pe_yf) else "N/A")
                with m_col2:
                    st.metric(label="Perubahan Harga", value=f"{price_change:,.0f}", delta=f"{pct_change:.2f}%")
                    st.metric(label="SMA 5 Hari (IDR)", value=f"{sma5:,.0f}" if pd.notna(sma5) else "N/A")
                    st.metric(label="Market Cap (yfinance)", value=format_large_number(mc_yf))
                with m_col3:
                    st.metric(label="Tertinggi (Periode Filter)", value=f"{ath_period:,.0f}")
                    st.metric(label="Terendah (Periode Filter)", value=f"{atl_period:,.0f}")
                    st.metric(label="Volatilitas Terakhir", value=f"{volatility_intra:.2f}%", help=f"High: {high_latest:,.0f} & Low: {low_latest:,.0f}")
                st.markdown("---") 
                
                x_axis_labels = stock_data_with_signals.index.strftime('%d %b %y')

                st.subheader("Grafik Analisis Terpadu (SMA, RSI, MACD)")
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
                fig.add_trace(go.Candlestick(x=x_axis_labels, open=stock_data_with_signals['Open'], high=stock_data_with_signals['High'], low=stock_data_with_signals['Low'], close=stock_data_with_signals['Close'], name='Harga'), row=1, col=1)
                fig.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['SMA20'], line=dict(color='orange', width=1), name='SMA 20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['SMA50'], line=dict(color='purple', width=1), name='SMA 50'), row=1, col=1)
                fig.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['Buy_Signal_Price'], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy Signal'), row=1, col=1)
                fig.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['Sell_Signal_Price'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell Signal'), row=1, col=1)
                fig.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['RSI'], line=dict(color='cyan', width=1), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                colors = ['green' if val > 0 else 'red' for val in stock_data_with_signals['MACD_hist']]
                fig.add_trace(go.Bar(x=x_axis_labels, y=stock_data_with_signals['MACD_hist'], name='Histogram', marker_color=colors), row=3, col=1)
                fig.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['MACD_line'], line=dict(color='blue', width=1.5), name='MACD'), row=3, col=1)
                fig.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['MACD_signal'], line=dict(color='orange', width=1.5), name='Signal'), row=3, col=1)
                fig.update_layout(height=800, title_text=f'Grafik Teknikal {selected_ticker}', xaxis_rangeslider_visible=False)
                fig.update_yaxes(title_text="Harga (IDR)", row=1, col=1); fig.update_yaxes(title_text="RSI", row=2, col=1); fig.update_yaxes(title_text="MACD", row=3, col=1)
                fig.update_xaxes(type='category')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
                
                st.subheader("Penjelasan Sinyal Teknikal (Ilustrasi)")
                st.info("""
                - **Sinyal Beli (▲) / Jual (▼)**: Berdasarkan persilangan SMA 20 dan SMA 50 (*Golden/Death Cross*).
                - **RSI**: Indikator momentum untuk melihat area jenuh beli (*overbought* > 70) atau jenuh jual (*oversold* < 30).
                - **MACD**: Indikator momentum lain. Sinyal beli saat garis biru memotong ke atas garis oranye, sinyal jual saat memotong ke bawah. Histogram hijau/merah menunjukkan kekuatan momentum.
                """)
                st.warning("**Disclaimer:** Analisis ini dibuat secara otomatis dan hanya bersifat ilustratif. Ini BUKAN merupakan rekomendasi finansial untuk membeli atau menjual. Selalu lakukan riset Anda sendiri (DYOR).")
                st.markdown("---")
                
                st.subheader("Analisis Volatilitas & Volume (Bollinger Bands, OBV)")
                with st.expander("Buka/Tutup Grafik Volatilitas & Volume"):
                    fig_A = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                    fig_A.add_trace(go.Candlestick(x=x_axis_labels, open=stock_data_with_signals['Open'], high=stock_data_with_signals['High'], low=stock_data_with_signals['Low'], close=stock_data_with_signals['Close'], name='Harga'), row=1, col=1)
                    fig_A.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name='Upper Band'), row=1, col=1)
                    fig_A.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['BB_Middle'], line=dict(color='orange', width=1.5), name='Middle Band (SMA20)'), row=1, col=1)
                    fig_A.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), name='Lower Band'), row=1, col=1)
                    fig_A.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['OBV'], line=dict(color='green', width=1.5), name='OBV'), row=2, col=1)
                    fig_A.update_layout(height=700, title_text=f'Analisis Volatilitas & Volume {selected_ticker}', xaxis_rangeslider_visible=False)
                    fig_A.update_yaxes(title_text="Harga (IDR)", row=1, col=1); fig_A.update_yaxes(title_text="OBV", row=2, col=1)
                    fig_A.update_xaxes(type='category')
                    st.plotly_chart(fig_A, use_container_width=True)
                    st.info("""**Cara Baca:** Grafik ini fokus pada gejolak harga dan kekuatan volume. **Bollinger Bands** (atas) menyempit saat pasar tenang dan melebar saat volatil. **OBV** (bawah) harusnya bergerak searah dengan tren harga untuk konfirmasi yang kuat.""")

                st.subheader("Analisis Momentum Lanjutan (Stochastic, MACD)")
                with st.expander("Buka/Tutup Grafik Momentum Lanjutan"):
                    fig_B = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
                    fig_B.add_trace(go.Candlestick(x=x_axis_labels, open=stock_data_with_signals['Open'], high=stock_data_with_signals['High'], low=stock_data_with_signals['Low'], close=stock_data_with_signals['Close'], name='Harga'), row=1, col=1)
                    fig_B.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['Stoch_K'], line=dict(color='cyan', width=1.5), name='%K'), row=2, col=1)
                    fig_B.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['Stoch_D'], line=dict(color='magenta', width=1.5), name='%D'), row=2, col=1)
                    fig_B.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
                    fig_B.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
                    colors_macd = ['green' if val > 0 else 'red' for val in stock_data_with_signals['MACD_hist']]
                    fig_B.add_trace(go.Bar(x=x_axis_labels, y=stock_data_with_signals['MACD_hist'], name='Histogram', marker_color=colors_macd), row=3, col=1)
                    fig_B.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['MACD_line'], line=dict(color='blue', width=1.5), name='MACD'), row=3, col=1)
                    fig_B.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['MACD_signal'], line=dict(color='orange', width=1.5), name='Signal'), row=3, col=1)
                    fig_B.update_layout(height=800, title_text=f'Analisis Momentum Lanjutan {selected_ticker}', xaxis_rangeslider_visible=False)
                    fig_B.update_yaxes(title_text="Harga (IDR)", row=1, col=1); fig_B.update_yaxes(title_text="Stochastic", row=2, col=1); fig_B.update_yaxes(title_text="MACD", row=3, col=1)
                    fig_B.update_xaxes(type='category')
                    st.plotly_chart(fig_B, use_container_width=True)
                    st.info("""**Cara Baca:** Grafik ini membandingkan dua indikator momentum. **Stochastic Oscillator** (tengah) bagus untuk melihat kondisi jenuh beli (>80) dan jenuh jual (<20). **MACD** (bawah) menunjukkan momentum tren secara umum.""")

                st.markdown("---")
                st.subheader("Grafik Harga Penutupan dan Volume Transaksi")
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(x=x_axis_labels, y=stock_data_with_signals['Close'], name='Harga Penutupan'), secondary_y=False)
                fig2.add_trace(go.Bar(x=x_axis_labels, y=stock_data_with_signals['Volume'], name='Volume Transaksi', opacity=0.4), secondary_y=True)
                fig2.update_layout(title_text=f"Harga Penutupan & Volume {selected_ticker}", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                fig2.update_yaxes(title_text="Harga Penutupan (IDR)", secondary_y=False)
                fig2.update_yaxes(title_text="Volume Transaksi", secondary_y=True)
                fig2.update_xaxes(type='category')
                st.plotly_chart(fig2, use_container_width=True)
                
                # --- [UPDATE] Tabel data dipindahkan ke sini ---
                st.markdown("---")
                with st.expander("📖 Lihat Tabel Data Harga Saham (30 Hari Terakhir)"):
                    # Definisi tabel dipindahkan ke dalam expander
                    table_data = stock_data_with_signals.tail(30).sort_index(ascending=False).copy()
                    table_data.index = table_data.index.strftime('%Y-%m-%d')
                    format_dict = {
                        'Open': '{:,.0f}', 'High': '{:,.0f}', 'Low': '{:,.0f}', 'Close': '{:,.0f}',
                        'Volume': '{:,.0f}', 'SMA20': '{:,.1f}', 'SMA50': '{:,.1f}', 'RSI': '{:.2f}',
                        'MACD_line': '{:.2f}', 'MACD_signal': '{:.2f}', 'MACD_hist': '{:.2f}',
                        'BB_Upper': '{:,.1f}', 'BB_Middle': '{:,.1f}', 'BB_Lower': '{:,.1f}',
                        'OBV': '{:,.0f}', 'Stoch_K': '{:.2f}', 'Stoch_D': '{:.2f}'
                    }
                    st.dataframe(table_data.style.format(format_dict, na_rep="N/A"), use_container_width=True)

            else:
                st.info(f"Tidak ada data untuk {selected_ticker} pada rentang tanggal terpilih, atau data tidak cukup untuk analisis.")
        else:
            st.info("Atur rentang tanggal di sidebar.")

    # --------------------------------------------------------------------------------
    # TAB ANALISIS ASING (KSEI BULANAN) - Logic dipindahkan dari 6_Pergerakan_Asing_FF (1).py
    with tab_asing:
        def ksei_analysis_tab(selected_ticker_with_suffix: str, period: str):
            st.title(f"📊 Analisis KSEI Bulanan: {selected_ticker_with_suffix}")
            st.caption("Data partisipasi asing/lokal per bulan berdasarkan laporan KSEI.")
            
            # Ticker tanpa suffix .JK untuk query ke ksei_month
            symbol_ksei = selected_ticker_with_suffix.replace(".JK", "")
            
            if not USE_KSEI:
                st.info("Tabel `ksei_month` belum tersedia di database.")
                return

            # ────────────────────────────────────────────────────────────────────────────────
            # Ringkasan Bulanan KSEI (dari ksei_month)
            st.markdown("---")
            st.subheader("📅 Ringkasan Bulanan KSEI")
            
            show_all_ksei = st.checkbox("Tampilkan semua data KSEI (abaikan filter Periode)", value=True, key="ksei_show_all")
            chart_type = st.radio("Tipe grafik bulanan", ["Line", "Bar"], index=0, horizontal=True, key="ksei_chart_type")

            params = {"sym": symbol_ksei}
            cond = ""
            if not show_all_ksei:
                cond, params = _date_filter_ksei(period, "trade_date")

            sql_k = f"""
                SELECT
                  trade_date, base_symbol,
                  (COALESCE(local_total,0) + COALESCE(foreign_total,0)) AS total_volume,
                  CASE WHEN COALESCE(local_total,0) + COALESCE(foreign_total,0) > 0
                       THEN 100.0 * COALESCE(foreign_total,0) / (COALESCE(local_total,0) + COALESCE(foreign_total,0))
                       ELSE NULL END AS foreign_pct,
                  CASE WHEN COALESCE(local_total,0) + COALESCE(foreign_total,0) > 0
                       THEN 100.0 - (100.0 * COALESCE(foreign_total,0) / (COALESCE(local_total,0) + COALESCE(foreign_total,0)))
                       ELSE NULL END AS retail_pct,
                  CASE WHEN price IS NOT NULL
                       THEN price * (COALESCE(local_total,0) + COALESCE(foreign_total,0))
                       ELSE NULL END AS total_value
                FROM ksei_month
                WHERE base_symbol = :sym
                {cond}
                ORDER BY trade_date
            """
            
            try:
                with engine.connect() as con:
                    kdf = pd.read_sql(text(sql_k), con, params=params)
            except Exception as e:
                st.error(f"Gagal mengambil data Ringkasan KSEI: {e}")
                kdf = pd.DataFrame()


            if not kdf.empty:
                kdf["trade_date"] = pd.to_datetime(kdf["trade_date"])
                kdf["Month"] = kdf["trade_date"].dt.strftime("%Y-%m")

                # Perhitungan estimasi volume
                tmp = kdf.copy()
                tmp["foreign_frac"] = pd.to_numeric(tmp["foreign_pct"], errors="coerce") / 100.0
                tmp["vol_foreign_est"] = pd.to_numeric(tmp["total_volume"], errors="coerce") * tmp["foreign_frac"]
                tmp["vol_local_est"]   = pd.to_numeric(tmp["total_volume"], errors="coerce") * (1.0 - tmp["foreign_frac"])
                
                vol = (tmp.groupby("Month", as_index=False)
                          .agg({"vol_foreign_est": "sum", "vol_local_est": "sum"}))
                
                agg_pct = (kdf.sort_values("trade_date")
                            .groupby("Month", as_index=False)
                            .agg({"foreign_pct": "mean"}))

                # Gabungkan untuk plotting
                agg_plot = vol.merge(agg_pct, on="Month")

                sub = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=("Volume Estimasi: Asing vs Lokal (Bulanan)", "Pa (%) Bulanan"),
                )

                if chart_type == "Line":
                    sub.add_trace(go.Scatter(x=agg_plot["Month"], y=agg_plot["vol_foreign_est"],
                                             name="Foreign (est.)", mode="lines+markers"), row=1, col=1)
                    sub.add_trace(go.Scatter(x=agg_plot["Month"], y=agg_plot["vol_local_est"],
                                             name="Lokal/Ritel (est.)", mode="lines+markers"), row=1, col=1)
                else:
                    sub.add_trace(go.Bar(x=agg_plot["Month"], y=agg_plot["vol_foreign_est"], name="Foreign (est.)"), row=1, col=1)
                    sub.add_trace(go.Bar(x=agg_plot["Month"], y=agg_plot["vol_local_est"],  name="Lokal/Ritel (est.)"), row=1, col=1)
                    sub.update_layout(barmode="stack")

                sub.update_yaxes(title_text="Volume (est.)", row=1, col=1)
                sub.add_trace(go.Scatter(x=agg_plot["Month"], y=agg_plot["foreign_pct"],
                                         name="Pa (%)", mode="lines+markers", line=dict(color='red')), row=2, col=1)
                sub.update_yaxes(title_text="Pa (%)", range=[0, 100], row=2, col=1)

                sub.update_layout(
                    height=650, hovermode="x unified",
                    showlegend=True, margin=dict(l=40, r=40, t=60, b=40),
                    title=f"Ringkasan Partisipasi Asing ({symbol_ksei})"
                )
                st.plotly_chart(sub, use_container_width=True)
                
                with st.expander("📄 Tabel Ringkas Partisipasi (per Bulan)"):
                    tbl_ringkas = kdf.groupby("Month", as_index=False).agg({
                        "total_volume": "sum",
                        "total_value": "sum",
                        "foreign_pct": "mean",
                        "retail_pct": "mean"
                    })
                    st.dataframe(tbl_ringkas.style.format({
                        "total_volume": '{:,.0f}',
                        "total_value": '{:,.0f}',
                        "foreign_pct": '{:.2f}%',
                        "retail_pct": '{:.2f}%'
                    }), use_container_width=True)
            else:
                st.info(f"Belum ada baris `ksei_month` untuk simbol {symbol_ksei}.")

            # ────────────────────────────────────────────────────────────────────────────────
            # 📊 DETAIL KATEGORI (paling bawah) — sumber: ksei_month
            st.markdown("---")
            st.subheader("📊 Detail Kategori KSEI (Semua Bulan)")

            CATEGORY_LABEL = {
                "ID": "Individual (Perorangan)",
                "CP": "Corporate (Perusahaan/Corporate)",
                "MF": "Mutual Fund (Reksa Dana)",
                "IB": "Financial Institution (Lembaga Keuangan Lainnya)",
                "IS": "Insurance (Perusahaan Perasuransian)",
                "SC": "Securities Company (Perusahaan Efek/Sekuritas)",
                "PF": "Pension Fund (Dana Pensiun)",
                "FD": "Foundation (Yayasan)",
                "OT": "Others (Lainnya)",
                "OTH": "Others (Gabungan Top-K)", # Untuk Shift Map
            }

            if USE_KSEI:
                detail_side = st.radio("Sisi", ["Lokal", "Asing", "Keduanya"], index=2, horizontal=True, key="ksei_detail_side")
                detail_chart = st.radio("Tipe grafik detail", ["Bar", "Line"], index=0, horizontal=True, key="ksei_detail_chart")
                use_all_detail = st.checkbox("Tampilkan semua bulan (abaikan filter Periode)", value=True, key="ksei_detail_show_all")

                with st.expander("ℹ️ Keterangan Kategori (sumber: Panduan KSEI)"):
                    k_rows = [f"- **{code}** — {CATEGORY_LABEL[code]}" for code in ["ID","CP","MF","IB","IS","SC","PF","FD","OT"]]
                    st.markdown("\n".join(k_rows))

                params_d = {"sym": symbol_ksei}
                cond_d = ""
                if not use_all_detail:
                    cond_d, params_d = _date_filter_ksei(period, "trade_date")

                sql_det = f"""
                    SELECT
                      trade_date, base_symbol,
                      local_is, local_cp, local_pf, local_ib, local_id, local_mf, local_sc, local_fd, local_ot, local_total,
                      foreign_is, foreign_cp, foreign_pf, foreign_ib, foreign_id, foreign_mf, foreign_sc, foreign_fd, foreign_ot, foreign_total
                    FROM ksei_month
                    WHERE base_symbol = :sym
                    {cond_d}
                    ORDER BY trade_date
                """
                
                try:
                    with engine.connect() as con:
                        kcat = pd.read_sql(text(sql_det), con, params=params_d)
                except Exception as e:
                    st.error(f"Gagal mengambil data Detail Kategori KSEI: {e}")
                    kcat = pd.DataFrame()

                if kcat.empty:
                    st.info(f"Belum ada data kategori di `ksei_month` untuk simbol {symbol_ksei}.")
                else:
                    kcat["trade_date"] = pd.to_datetime(kcat["trade_date"])
                    kcat["Month"] = kcat["trade_date"].dt.strftime("%Y-%m")

                    num_cols = [
                        "local_is","local_cp","local_pf","local_ib","local_id","local_mf","local_sc","local_fd","local_ot","local_total",
                        "foreign_is","foreign_cp","foreign_pf","foreign_ib","foreign_id","foreign_mf","foreign_sc","foreign_fd","foreign_ot","foreign_total"
                    ]
                    kcat[num_cols] = kcat[num_cols].apply(pd.to_numeric, errors="coerce")
                    agg_cat = kcat.groupby("Month", as_index=False)[num_cols].sum()

                    # Plot helper
                    def _plot_categories(df_month: pd.DataFrame, cols: list, title: str, side_label: str):
                        long = df_month.melt(id_vars="Month", value_vars=cols, var_name="category", value_name="volume").fillna(0.0)
                        long["code"] = (long["category"]
                                        .str.replace(r"^local_", "", regex=True)
                                        .str.replace(r"^foreign_", "", regex=True)
                                        .str.upper())

                        figc = go.Figure()
                        for code in sorted(long["code"].unique().tolist()):
                            pretty = f"{code} — {CATEGORY_LABEL.get(code, code)}"
                            y = long.loc[long["code"] == code, ["Month", "volume"]]
                            if detail_chart == "Bar":
                                figc.add_bar(x=y["Month"], y=y["volume"], name=pretty)
                            else:
                                figc.add_scatter(x=y["Month"], y=y["volume"], mode="lines+markers", name=pretty)

                        figc.update_layout(
                            title=title,
                            barmode="stack" if detail_chart == "Bar" else None,
                            height=420, hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=40, r=40, t=40, b=40),
                        )
                        figc.update_yaxes(title_text=f"Volume ({side_label})")
                        st.plotly_chart(figc, use_container_width=True)

                    # Lokal
                    if detail_side in ("Lokal", "Keduanya"):
                        _plot_categories(
                            agg_cat,
                            ["local_is","local_cp","local_pf","local_ib","local_id","local_mf","local_sc","local_fd","local_ot"],
                            f"Detail Kategori — **Lokal** ({symbol_ksei})",
                            "Lokal",
                        )
                    # Asing
                    if detail_side in ("Asing", "Keduanya"):
                        _plot_categories(
                            agg_cat,
                            ["foreign_is","foreign_cp","foreign_pf","foreign_ib","foreign_id","foreign_mf","foreign_sc","foreign_fd","foreign_ot"],
                            f"Detail Kategori — **Asing** ({symbol_ksei})",
                            "Asing",
                        )

                    with st.expander("📄 Tabel Ringkas Kategori (per Bulan)"):
                        show_cols = ["Month",
                                     "local_is","local_cp","local_pf","local_ib","local_id","local_mf","local_sc","local_fd","local_ot","local_total",
                                     "foreign_is","foreign_cp","foreign_pf","foreign_ib","foreign_id","foreign_mf","foreign_sc","foreign_fd","foreign_ot","foreign_total"]
                        
                        # Format angka
                        format_cols = {c: '{:,.0f}' for c in show_cols if c != "Month"}
                        st.dataframe(agg_cat[show_cols].style.format(format_cols), use_container_width=True)

            else:
                st.info("Tabel `ksei_month` belum tersedia untuk detail kategori.")
            
            st.markdown("---")
            st.info("Catatan: Fitur Analisis Foreign Flow Harian, Heatmap/Shift Map, Event Study, dan Agregasi Sektor/Breadth dari halaman 6_Pergerakan_Asing_FF (1).py belum dipindahkan. Hanya data Ringkasan dan Detail Kategori KSEI yang dipindahkan.")
            
        # Panggil fungsi tab KSEI
        ksei_analysis_tab(selected_ticker, period_ksei)

else:
    st.info("Silakan pilih ticker saham dari sidebar.")

st.sidebar.markdown("---")
st.sidebar.caption("Data harga saham historis dan analisis KSEI ditampilkan dari database lokal.")
