# -*- coding: utf-8 -*-
# app/pages/6_Pergerakan_Asing_FF.py (Dashboard Makro Baru: Emas & Rupiah)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import json
import random

# TINGKATKAN JUMLAH PARAMETER IMPOR KARENA FUNGSI LAMA TIDAK DIPAKAI
# Impor dasar Streamlit, Pandas, Plotly (tidak butuh koneksi DB di sini)
# Import yang dibutuhkan untuk simulasi/fetch data
import time
from tools import google_search

st.set_page_config(page_title="💰 Historis Emas & Rupiah", page_icon="📈", layout="wide")
st.title("💰 Historis Emas & Nilai Tukar Rupiah")
st.caption(
    "Menampilkan data historis harga emas dunia (USD/oz) dan nilai tukar Rupiah terhadap Dolar (IDR/USD) "
    "untuk analisis makroekonomi. Data diambil menggunakan Google Search (simulasi)."
)

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────

# Simulasikan nama queries untuk Google Search (seolah-olah mencari data)
GOLD_QUERY = "Historical gold price in USD per ounce last 5 years"
IDR_USD_QUERY = "Historical IDR to USD exchange rate last 5 years"

# ────────────────────────────────────────────────────────────────────────────────
# Data Fetch Simulation (menggunakan Google Search Tool)
# ────────────────────────────────────────────────────────────────────────────────

# Fungsi untuk memanggil Google Search Tool
def fetch_historical_data(query: str, search_query: str) -> str:
    """Simulasi fetching data historis menggunakan Google Search."""
    
    # Menghindari Google Search Tool dipanggil terus-menerus
    if 'history_cache' not in st.session_state:
        st.session_state.history_cache = {}
    
    if query in st.session_state.history_cache:
        return st.session_state.history_cache[query]

    # Google Search Tool dipanggil di sini
    results = google_search.search(queries=[search_query])
    
    # Karena kita tidak dapat memprediksi format output search,
    # kita akan melakukan simulasi data di bagian selanjutnya.
    # Untuk tujuan demo, kita akan kembalikan string yang menandakan pencarian sukses.
    
    time.sleep(1) # Simulasikan latency
    st.session_state.history_cache[query] = f"Search success for: {search_query}"
    return st.session_state.history_cache[query]

# Fungsi Simulasi Data Historis (karena tool tidak bisa mengembalikan DataFrame)
def generate_simulated_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Menghasilkan DataFrame simulasi untuk harga emas dan kurs USD/IDR."""
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    df = pd.DataFrame(index=dates)
    
    # Emas (Gold) - Mulai dari 1500 USD, fluktuasi
    np.random.seed(42)
    gold_base = 1500
    gold_price = [gold_base]
    for _ in range(1, len(dates)):
        change = np.random.normal(0.5, 8) # Drift positif kecil
        gold_price.append(gold_price[-1] * (1 + change / 1000))
    df['Gold_USD'] = np.array(gold_price) + 200 # Agar mendekati harga saat ini
    
    # Rupiah (IDR/USD) - Mulai dari 14500, fluktuasi
    idr_base = 14500
    idr_rate = [idr_base]
    for _ in range(1, len(dates)):
        change = np.random.normal(0.01, 3) # Drift positif (pelebaran kurs)
        idr_rate.append(idr_rate[-1] * (1 + change / 10000))
    df['IDR_USD'] = np.array(idr_rate) + 1000 # Agar mendekati kurs saat ini

    # Perhitungan perubahan (persen)
    df['Gold_Change_Pct'] = df['Gold_USD'].pct_change() * 100
    df['IDR_Change_Pct'] = df['IDR_USD'].pct_change() * 100
    
    return df.dropna()

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit Interface
# ────────────────────────────────────────────────────────────────────────────────

# Filter Sidebar
st.sidebar.header("Filter Periode Makro")
end_date = datetime.now().date()
start_date_default = end_date - timedelta(days=365 * 3) # Default 3 tahun
selected_start_date = st.sidebar.date_input("Tanggal Mulai", value=start_date_default, max_value=end_date)
selected_end_date = st.sidebar.date_input("Tanggal Akhir", value=end_date, min_value=selected_start_date)

# Fetch data simulasi
simulated_df = generate_simulated_data(selected_start_date.strftime('%Y-%m-%d'), selected_end_date.strftime('%Y-%m-%d'))

if simulated_df.empty:
    st.warning("Data historis tidak tersedia untuk rentang waktu ini.")
    st.stop()

# Panggil fungsi fetch (meskipun hanya simulasi)
with st.expander("🔎 Hasil Google Search (Simulasi Data Historis)"):
    col1, col2 = st.columns(2)
    with col1:
        st.info(fetch_historical_data("gold", GOLD_QUERY))
    with col2:
        st.info(fetch_historical_data("idr", IDR_USD_QUERY))
        
# Data terakhir
latest_gold = simulated_df['Gold_USD'].iloc[-1]
latest_idr = simulated_df['IDR_USD'].iloc[-1]
prev_gold = simulated_df['Gold_USD'].iloc[-2] if len(simulated_df) > 1 else latest_gold
prev_idr = simulated_df['IDR_USD'].iloc[-2] if len(simulated_df) > 1 else latest_idr

# Perubahan
change_gold = latest_gold - prev_gold
change_gold_pct = (change_gold / prev_gold) * 100
change_idr = latest_idr - prev_idr
change_idr_pct = (change_idr / prev_idr) * 100

# ────────────────────────────────────────────────────────────────────────────────
# Ringkasan Metrik
st.subheader("Ringkasan Data Makro Terbaru")

col_g1, col_g2, col_i1, col_i2 = st.columns(4)

with col_g1:
    st.metric(
        label="Harga Emas (USD/oz)",
        value=f"${latest_gold:,.2f}",
        delta=f"{change_gold:+.2f}"
    )
with col_g2:
    st.metric(
        label="Perubahan Emas (%)",
        value=f"{change_gold_pct:+.2f}%",
        delta=None,
    )

with col_i1:
    st.metric(
        label="Nilai Tukar (IDR/USD)",
        value=f"Rp{latest_idr:,.0f}",
        delta=f"Rp{change_idr:+.0f}"
    )
with col_i2:
    st.metric(
        label="Perubahan Rupiah (%)",
        value=f"{change_idr_pct:+.2f}%",
        delta=None,
        delta_color="inverse" # Warna delta terbalik, karena kenaikan kurs IDR buruk
    )

st.markdown("---") 

# ────────────────────────────────────────────────────────────────────────────────
# Grafik Terpadu (Emas dan Rupiah)
st.subheader("Grafik Historis (Emas dan Nilai Tukar)")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, 
                    subplot_titles=("Harga Emas Dunia (USD/oz)", "Nilai Tukar Rupiah (IDR/USD)"))

# Grafik Emas (Row 1)
fig.add_trace(
    go.Scatter(x=simulated_df.index, y=simulated_df['Gold_USD'], mode='lines', name='Gold Price (USD)', line=dict(color='#FFD700')), 
    row=1, col=1
)
fig.update_yaxes(title_text="USD/oz", tickformat="$,.0f", row=1, col=1)

# Grafik Rupiah (Row 2)
fig.add_trace(
    go.Scatter(x=simulated_df.index, y=simulated_df['IDR_USD'], mode='lines', name='IDR/USD Rate', line=dict(color='#008000')), 
    row=2, col=1
)
fig.update_yaxes(title_text="IDR/USD", tickprefix="Rp", tickformat=",0f", row=2, col=1)


# Update Layout
fig.update_layout(
    height=700,
    title_text=f"Historis Harga Emas dan Nilai Tukar ({selected_start_date} s/d {selected_end_date})",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=60, b=40),
)

fig.update_xaxes(
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

st.plotly_chart(fig, use_container_width=True)

st.markdown("---") 
st.subheader("Analisis Pergerakan Harian (Persentase)")

fig2 = make_subplots(specs=[[{"secondary_y": True}]])

# Perubahan Emas Harian
fig2.add_trace(
    go.Bar(x=simulated_df.index, y=simulated_df['Gold_Change_Pct'], name='Emas (% Harian)', opacity=0.8, marker_color=np.where(simulated_df['Gold_Change_Pct'] > 0, 'green', 'red')),
    secondary_y=False
)

# Perubahan Rupiah Harian
fig2.add_trace(
    go.Scatter(x=simulated_df.index, y=simulated_df['IDR_Change_Pct'], name='Rupiah (% Harian)', mode='lines', line=dict(color='orange', width=2)), 
    secondary_y=True
)

fig2.update_yaxes(title_text="Emas (% Perubahan)", secondary_y=False)
fig2.update_yaxes(title_text="Rupiah (% Perubahan)", secondary_y=True)

fig2.update_layout(
    title_text="Perbandingan Perubahan Harian Emas vs Rupiah",
    height=500,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("Disclaimer: Data Emas dan Nilai Tukar di atas adalah data simulasi yang dihasilkan secara matematis. Silakan gunakan sumber data yang terpercaya untuk keputusan investasi nyata.")
