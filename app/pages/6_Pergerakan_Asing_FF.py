# -*- coding: utf-8 -*-
# app/pages/6_Pergerakan_Asing_FF.py (Dashboard Makro Baru: Emas & Rupiah dari MariaDB)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import json
import random
import os
import tempfile
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
import time

st.set_page_config(page_title="💰 Historis Emas & Rupiah", page_icon="📈", layout="wide")
st.title("💰 Historis Emas & Nilai Tukar Rupiah (MariaDB)")
st.caption(
    "Menampilkan data historis harga emas dunia (USD/oz) dan nilai tukar Rupiah terhadap Dolar (IDR/USD) "
    "yang tersimpan di tabel `macro_data` database Anda."
)

# ────────────────────────────────────────────────────────────────────────────────
# DB Connection & Utility (Didefinisikan ulang di sini)
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
TABLE_NAME = "macro_data"

# ────────────────────────────────────────────────────────────────────────────────
# Fungsi Uploader/Seeder Data (Simulasi untuk keperluan demo)
# ────────────────────────────────────────────────────────────────────────────────

def generate_macro_data(start_date, end_date) -> pd.DataFrame:
    """Menghasilkan DataFrame simulasi untuk harga emas dan kurs USD/IDR."""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    df = pd.DataFrame(index=dates)
    
    # Emas (Gold)
    np.random.seed(42)
    gold_base = 1800
    gold_price = [gold_base]
    for _ in range(1, len(dates)):
        change = np.random.normal(0.5, 8)
        gold_price.append(gold_price[-1] * (1 + change / 1000))
    df['Gold_USD'] = np.array(gold_price)
    
    # Rupiah (IDR/USD)
    idr_base = 15000
    idr_rate = [idr_base]
    for _ in range(1, len(dates)):
        change = np.random.normal(0.01, 3)
        idr_rate.append(idr_rate[-1] * (1 + change / 10000))
    df['IDR_USD'] = np.array(idr_rate)
    
    df = df.reset_index().rename(columns={'index': 'trade_date'})
    df['trade_date'] = df['trade_date'].dt.date
    return df

def upload_simulated_data(df: pd.DataFrame):
    """Menyimpan DataFrame ke tabel macro_data dengan PRIMARY KEY."""
    
    try:
        with engine.connect() as con:
            # 1. Pastikan tabel dibuat dengan PRIMARY KEY (trade_date)
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    trade_date DATE NOT NULL,
                    Gold_USD FLOAT,
                    IDR_USD FLOAT,
                    PRIMARY KEY (trade_date)
                ) ENGINE=InnoDB;
            """
            con.execute(text(create_table_sql))
            con.commit()
            
            # 2. Gunakan REPLACE INTO untuk memasukkan/memperbarui data (Upsert)
            data_to_insert = df[['trade_date', 'Gold_USD', 'IDR_USD']].to_dict(orient='records')
            
            # Buat query REPLACE INTO
            replace_sql = f"""
                REPLACE INTO {TABLE_NAME} (trade_date, Gold_USD, IDR_USD)
                VALUES (:trade_date, :Gold_USD, :IDR_USD)
            """
            con.execute(text(replace_sql), data_to_insert)
            con.commit()
            
        st.success(f"Berhasil mengunggah {len(df)} baris data makro ke tabel `{TABLE_NAME}`.")
        
        # Membersihkan SEMUA cache dan menjalankan ulang
        st.cache_data.clear()
        st.cache_resource.clear()
        
        st.rerun() 
    except Exception as e:
        st.error(f"Gagal mengunggah data ke database: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# Data Fetcher
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def fetch_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Mengambil data makro dari tabel MariaDB."""
    if not _table_exists(TABLE_NAME):
        return pd.DataFrame()
    
    sql = f"""
        SELECT trade_date, Gold_USD, IDR_USD
        FROM {TABLE_NAME}
        WHERE trade_date BETWEEN :start_date AND :end_date
        ORDER BY trade_date
    """
    params = {"start_date": start_date, "end_date": end_date}
    
    try:
        with engine.connect() as con:
            df = pd.read_sql(text(sql), con, params=params)
        
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        
        # Hitung perubahan
        df['Gold_Change_Pct'] = df['Gold_USD'].pct_change() * 100
        df['IDR_Change_Pct'] = df['IDR_USD'].pct_change() * 100
        return df.dropna()

    except Exception as e:
        st.error(f"Gagal mengambil data dari `{TABLE_NAME}`: {e}")
        return pd.DataFrame()

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit Interface
# ────────────────────────────────────────────────────────────────────────────────

# Cek keberadaan tabel dan tampilkan uploader jika belum ada
if not _table_exists(TABLE_NAME):
    st.warning(f"Tabel `{TABLE_NAME}` belum ditemukan di database Anda.")
    
    with st.expander("🛠️ Klik di sini untuk membuat tabel `macro_data` (Simulasi)") as exp:
        st.info("Fitur ini akan membuat tabel `macro_data` dengan *Primary Key* dan mengisi data historis Emas dan Rupiah (simulasi 5 tahun) untuk keperluan demo.")
        
        today = datetime.now().date()
        sim_start = today - timedelta(days=365 * 5)
        
        if st.button("Buat & Isi Data Makro Simulasi"):
            sim_df = generate_macro_data(sim_start, today)
            upload_simulated_data(sim_df)
            # st.rerun() sudah ada di dalam upload_simulated_data
    st.stop()
    
# Filter Sidebar
st.sidebar.header("Filter Periode Makro")
end_date = datetime.now().date()
start_date_default = end_date - timedelta(days=365 * 3) # Default 3 tahun
selected_start_date = st.sidebar.date_input("Tanggal Mulai", value=start_date_default, max_value=end_date)
selected_end_date = st.sidebar.date_input("Tanggal Akhir", value=end_date, min_value=selected_start_date)

# Fetch data dari DB
simulated_df = fetch_macro_data(selected_start_date.strftime('%Y-%m-%d'), selected_end_date.strftime('%Y-%m-%d'))

if simulated_df.empty:
    st.warning(f"Tidak ada data makro tersedia untuk rentang waktu ini di tabel `{TABLE_NAME}`.")
    st.stop()
    
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
    title_text=f"Historis Harga Emas dan Nilai Tukar ({selected_start_date.strftime('%Y-%m-%d')} s/d {selected_end_date.strftime('%Y-%m-%d')})",
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
st.caption("⚠️ **Penting:** Data makro di atas adalah **data simulasi** yang di-*generate* secara matematis dan disimpan ke database Anda. Untuk data riil, Anda perlu mengintegrasikan dengan API data finansial yang kredibel.")
