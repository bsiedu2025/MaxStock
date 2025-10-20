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

# Inisialisasi session state untuk loading
if 'is_loading' not in st.session_state: # <--- Perbaikan di sini
    st.session_state.is_loading = False

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
    # Menghasilkan tanggal hanya di hari kerja
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    df = pd.DataFrame(index=dates)
    
    # Emas (Gold)
    # Gunakan seed yang berbeda dari sebelumnya agar data terlihat baru jika di-update
    np.random.seed(int(time.time())) 
    gold_base = 35  # Harga emas di tahun 1970an
    gold_price = [gold_base]
    
    # KENAISAN REVISI: Menggunakan pembagi 200 (sebelumnya 1000) untuk mencapai nilai ~2000 USD di tahun 2025
    for _ in range(1, len(dates)):
        # Perubahan yang lebih besar untuk rentang waktu yang panjang (Drift 0.08, Vol 15)
        change = np.random.normal(0.08, 15)
        gold_price.append(gold_price[-1] * (1 + change / 200)) # Revisi pembagi dari 1000 ke 200
    df['Gold_USD'] = np.array(gold_price)
    
    # Rupiah (IDR/USD)
    idr_base = 380 # Kurs Rupiah di tahun 1970an
    idr_rate = [idr_base]
    for _ in range(1, len(dates)):
        # Perubahan yang lebih kecil untuk kurs
        change = np.random.normal(0.05, 5)
        idr_rate.append(idr_rate[-1] * (1 + change / 10000))
    df['IDR_USD'] = np.array(idr_rate)
    
    df = df.reset_index().rename(columns={'index': 'trade_date'})
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date # Simpan sebagai date object
    return df

def upload_simulated_data(df: pd.DataFrame, mode: str):
    """Menyimpan DataFrame ke tabel macro_data menggunakan to_sql untuk efisiensi.
       Mode: 'create' (buat baru), 'append' (tambah/update harian)."""
    
    st.session_state.is_loading = True
    total_rows = len(df)
    
    with st.status("Memproses data makro...", expanded=True) as status_bar:
        try:
            status_bar.update(label="1. Menyiapkan koneksi database...", state="running")
            
            engine = _build_engine()
            
            if mode == 'create':
                # Hapus tabel lama (jika ada) dan buat baru dengan data 1970
                status_bar.update(label="2. Menghapus tabel lama dan membuat tabel baru...")
                
                # Menggunakan to_sql dengan if_exists='replace' akan menangani CREATE TABLE
                # Kita perlu memastikan Primary Key terbuat, jadi kita buat manual DULU.
                with engine.connect() as con:
                     # Pastikan tabel dibuat dengan PRIMARY KEY (trade_date)
                    create_table_sql = f"""
                        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                            trade_date DATE NOT NULL,
                            Gold_USD FLOAT,
                            IDR_USD FLOAT,
                            PRIMARY KEY (trade_date)
                        ) ENGINE=InnoDB;
                    """
                    con.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME}")) # Hapus yang lama
                    con.execute(text(create_table_sql))
                    con.commit()
                
                status_bar.update(label=f"3. Mengunggah {total_rows} baris data (1970-sekarang)...")
                
                # Menggunakan to_sql (append) untuk mengisi data ke tabel yang baru dibuat
                df_temp = df.set_index('trade_date') # to_sql akan menggunakan index sebagai kolom jika tidak ditentukan
                df_temp.to_sql(
                    name=TABLE_NAME, 
                    con=engine, 
                    if_exists='append', 
                    index=True,
                    chunksize=5000 # Tetapkan chunksize untuk transaksi yang besar
                )
                
            elif mode == 'append':
                status_bar.update(label=f"2. Mengunggah {total_rows} baris data harian...")
                
                with engine.connect() as con:
                    # Hapus data yang ada (jika tanggal sama) dan insert data baru
                    con.execute(text(f"DELETE FROM {TABLE_NAME} WHERE trade_date = :trade_date"), 
                                {"trade_date": df['trade_date'].iloc[0]})
                    con.commit()
                    
                    df.to_sql(
                        name=TABLE_NAME, 
                        con=con, 
                        if_exists='append', 
                        index=False
                    )
                
            
            status_bar.update(label=f"✅ Pengunggahan Data Makro Selesai! ({total_rows} baris)", state="complete", expanded=False)
            
            # Membersihkan SEMUA cache dan menjalankan ulang
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success(f"Berhasil mengunggah {total_rows} baris data makro ke tabel `{TABLE_NAME}`.")

            st.rerun() 
            
        except Exception as e:
            status_bar.update(label="❌ Gagal mengunggah data!", state="error", expanded=True)
            st.error(f"Terjadi kesalahan database: {e}")
        finally:
            st.session_state.is_loading = False # Reset loading state

# Fungsi untuk menghapus tabel
def delete_table():
    try:
        with engine.connect() as con:
            con.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))
            con.commit()
        st.success(f"Tabel `{TABLE_NAME}` berhasil dihapus.")
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Gagal menghapus tabel: {e}")


# ────────────────────────────────────────────────────────────────────────────────
# Data Fetcher & Aggregator
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def fetch_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Mengambil data makro harian dari tabel MariaDB."""
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
        
        return df

    except Exception as e:
        st.error(f"Gagal mengambil data dari `{TABLE_NAME}`: {e}")
        return pd.DataFrame()

def aggregate_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Mengagregasi data harian ke frekuensi Mingguan, Bulanan, atau Tahunan."""
    if freq == 'Harian':
        return df
    
    # Menentukan resampling code (W: Weekly, M: Monthly, Y: Yearly)
    if freq == 'Mingguan':
        rule = 'W'
    elif freq == 'Bulanan':
        rule = 'M'
    elif freq == 'Tahunan':
        rule = 'Y'
    else:
        return df # Fallback
    
    # Resample: ambil data penutupan terakhir (end of period)
    aggregated_df = df.resample(rule).last().dropna()
    
    # Hitung perubahan
    aggregated_df['Gold_Change_Pct'] = aggregated_df['Gold_USD'].pct_change() * 100
    aggregated_df['IDR_Change_Pct'] = aggregated_df['IDR_USD'].pct_change() * 100
    
    return aggregated_df

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit Interface
# ────────────────────────────────────────────────────────────────────────────────

# Cek keberadaan tabel dan tampilkan uploader jika belum ada
if not _table_exists(TABLE_NAME):
    st.warning(f"Tabel `{TABLE_NAME}` belum ditemukan di database Anda.")
    
    with st.expander("🛠️ Klik di sini untuk membuat tabel `macro_data` (Simulasi)") as exp:
        st.info("Fitur ini akan membuat tabel `macro_data` dengan *Primary Key* dan mengisi data historis Emas dan Rupiah (simulasi dari **1970**) untuk keperluan demo.")
        
        today = datetime.now().date()
        # Mengubah tanggal mulai simulasi ke 1 Januari 1970
        sim_start = datetime(1970, 1, 1).date()
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("Buat & Isi Data Makro Simulasi", disabled=st.session_state.is_loading):
                st.session_state.is_loading = True
                sim_df = generate_macro_data(sim_start, today)
                upload_simulated_data(sim_df, 'create') 
        
        with col_btn2:
            if st.button("🗑️ Hapus Tabel Macro Data", disabled=st.session_state.is_loading, key="delete_initial_table"):
                delete_table()
            
    st.stop()
    
# --- LOGIKA KETIKA TABEL SUDAH ADA ---
else:
    # Tampilkan tombol untuk update data simulasi 1970 atau menghapus tabel
    with st.expander("🛠️ Opsi Perawatan Data Makro (Tabel sudah ada)") as exp:
        st.info("Anda dapat menggunakan opsi ini untuk memperbarui simulasi ke rentang 1970 atau menghapus tabel secara total.")
        
        col_upd, col_del = st.columns(2)
        today = datetime.now().date()
        sim_start = datetime(1970, 1, 1).date()

        with col_upd:
            if st.button("🔄 Ganti dengan Data 1970 (Semua data lama akan tertimpa)", key="btn_replace_1970", disabled=st.session_state.is_loading):
                st.session_state.is_loading = True
                sim_df = generate_macro_data(sim_start, today)
                upload_simulated_data(sim_df, 'create') # Menggunakan mode create untuk menimpa total

        with col_del:
            if st.button("🗑️ Hapus Tabel Macro Data", key="btn_delete_table", disabled=st.session_state.is_loading):
                delete_table()

# Tampilkan loading indicator global jika sedang proses
if st.session_state.is_loading:
    # Jangan tampilkan st.stop() di sini agar placeholder bisa di-render
    pass


# --- FORM UPDATE HARIAN DI SIDEBAR ---
with st.sidebar:
    st.header("🔄 Update Data Harian")
    with st.form("macro_update_form", clear_on_submit=True):
        update_date = st.date_input("Tanggal Data", value=datetime.now().date(), max_value=datetime.now().date(), disabled=st.session_state.is_loading)
        gold_price = st.number_input("Harga Emas (USD/oz)", min_value=1.0, format="%.2f", step=1.0, disabled=st.session_state.is_loading)
        idr_rate = st.number_input("Nilai Tukar (IDR/USD)", min_value=1.0, format="%.0f", step=1.0, disabled=st.session_state.is_loading)
        
        submitted = st.form_submit_button("Simpan Data ke Database", disabled=st.session_state.is_loading)
        
        if submitted:
            if gold_price > 0 and idr_rate > 0:
                # Buat DataFrame dari input tunggal
                new_data = pd.DataFrame([{
                    'trade_date': update_date,
                    'Gold_USD': gold_price,
                    'IDR_USD': idr_rate
                }])
                
                # Gunakan fungsi upload_simulated_data dengan mode 'append'
                upload_simulated_data(new_data, 'append')
                # Fungsi upload_simulated_data akan otomatis melakukan st.rerun()
            else:
                st.error("Harga Emas dan Nilai Tukar harus diisi dengan nilai > 0.")


# Filter Sidebar (Tersedia setelah tabel ada)
if not _table_exists(TABLE_NAME): st.stop()

st.sidebar.header("Filter Periode Makro")
end_date = datetime.now().date()

# Cek tanggal terlama dari data di DB untuk min_value
try:
    with engine.connect() as con:
        min_date_db = pd.read_sql(text(f"SELECT MIN(trade_date) FROM {TABLE_NAME}"), con).iloc[0, 0]
        if min_date_db:
             min_date_db = min_date_db
        else:
             min_date_db = datetime(2010, 1, 1).date()
except Exception:
    min_date_db = datetime(2010, 1, 1).date()


start_date_default = end_date - timedelta(days=365 * 3) # Default 3 tahun
selected_start_date = st.sidebar.date_input("Tanggal Mulai", value=start_date_default, min_value=min_date_db, max_value=end_date, key="filter_start")
selected_end_date = st.sidebar.date_input("Tanggal Akhir", value=end_date, min_value=selected_start_date, max_value=end_date, key="filter_end")

st.sidebar.markdown("---")
# Pilihan Agregasi (Mingguan, Bulanan, Tahunan)
aggregation_freq = st.sidebar.selectbox(
    "Agregasi Data",
    ['Harian', 'Mingguan', 'Bulanan', 'Tahunan'],
    index=0
)

# Fetch data harian dari DB
raw_df = fetch_macro_data(selected_start_date.strftime('%Y-%m-%d'), selected_end_date.strftime('%Y-%m-%d'))

if raw_df.empty:
    st.warning(f"Tidak ada data makro tersedia untuk rentang waktu ini di tabel `{TABLE_NAME}`.")
    st.stop()
    
# Agregasi Data
simulated_df = aggregate_data(raw_df, aggregation_freq)


# Data terakhir (Pastikan indeks ada dan hitung perubahan untuk metrik)
if len(simulated_df) >= 2:
    latest_gold = simulated_df['Gold_USD'].iloc[-1]
    latest_idr = simulated_df['IDR_USD'].iloc[-1]
    prev_gold = simulated_df['Gold_USD'].iloc[-2]
    prev_idr = simulated_df['IDR_USD'].iloc[-2]

    # Hitung perubahan (diperlukan karena aggregate_data tidak mengembalikan Gold_Change_Pct secara default untuk Harian)
    change_gold = latest_gold - prev_gold
    change_gold_pct = (change_gold / prev_gold) * 100 if prev_gold != 0 else 0
    change_idr = latest_idr - prev_idr
    change_idr_pct = (change_idr / prev_idr) * 100 if prev_idr != 0 else 0

elif len(simulated_df) == 1:
    latest_gold = simulated_df['Gold_USD'].iloc[-1]
    latest_idr = simulated_df['IDR_USD'].iloc[-1]
    change_gold = 0; change_gold_pct = 0
    change_idr = 0; change_idr_pct = 0
else:
    st.warning("Data tidak cukup untuk menghitung perubahan.")
    st.stop()


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

# Membangun subplot tanpa subplot titles, karena akan dihapus
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)

# Grafik Emas (Row 1)
fig.add_trace(
    go.Scatter(x=simulated_df.index, y=simulated_df['Gold_USD'], name='Gold (USD/oz)', line=dict(color='#FFD700', width=1.5)), 
    row=1, col=1
)
fig.update_yaxes(title_text="Harga Emas (USD/oz)", tickformat="$,.0f", row=1, col=1)

# Grafik Rupiah (Row 2)
fig.add_trace(
    go.Scatter(x=simulated_df.index, y=simulated_df['IDR_USD'], name='IDR/USD Rate', line=dict(color='#008000', width=1.5)), 
    row=2, col=1
)
fig.update_yaxes(title_text="Nilai Tukar (IDR/USD)", tickprefix="Rp", tickformat=",0f", row=2, col=1)


# Update Layout
fig.update_layout(
    height=700,
    title_text=f"Historis Harga Emas dan Nilai Tukar Agregasi {aggregation_freq} ({selected_start_date.strftime('%Y-%m-%d')} s/d {selected_end_date.strftime('%Y-%m-%d')})",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="right", x=1)
)

fig.update_xaxes(
    type='date',
    # Range slider hanya ditampilkan di sumbu-x paling bawah (row=2)
    rangeslider_visible=False,
    row=1, col=1
)

fig.update_xaxes(
    type='date',
    # Range slider ditampilkan di sumbu-x paling bawah (row=2)
    rangeslider_visible=True, 
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1B", step="month", stepmode="backward"),
            dict(count=6, label="6B", step="month", stepmode="backward"),
            dict(count=1, label="1T", step="year", stepmode="backward"),
            dict(step="all")
        ])
    ),
    row=2, col=1
)


st.plotly_chart(fig, use_container_width=True)

st.markdown("---") 
st.subheader(f"Analisis Pergerakan {aggregation_freq} (Persentase)")

# Hitung ulang perubahan persentase untuk grafik di bawah (selain mode Harian)
if aggregation_freq != 'Harian':
    df_chart = simulated_df.copy()
    df_chart['Gold_Change_Pct'] = df_chart['Gold_USD'].pct_change() * 100
    df_chart['IDR_Change_Pct'] = df_chart['IDR_USD'].pct_change() * 100
    df_chart = df_chart.dropna()
else:
    # Untuk harian, kita menggunakan raw_df untuk perubahan % (setelah fetch, sebelum agregasi)
    # Karena raw_df tidak memiliki Gold_Change_Pct, kita hitung di sini
    df_chart = raw_df.copy()
    df_chart['Gold_Change_Pct'] = df_chart['Gold_USD'].pct_change() * 100
    df_chart['IDR_Change_Pct'] = df_chart['IDR_USD'].pct_change() * 100
    df_chart = df_chart.dropna()


fig2 = make_subplots(specs=[[{"secondary_y": True}]])

# Perubahan Emas Harian/Agregasi
fig2.add_trace(
    go.Bar(x=df_chart.index, y=df_chart['Gold_Change_Pct'], name=f'Emas (% per {aggregation_freq})', opacity=0.8, marker_color=np.where(df_chart['Gold_Change_Pct'] > 0, 'green', 'red')),
    secondary_y=False
)

# Perubahan Rupiah Harian/Agregasi
fig2.add_trace(
    go.Scatter(x=df_chart.index, y=df_chart['IDR_Change_Pct'], name=f'Rupiah (% per {aggregation_freq})', mode='lines', line=dict(color='orange', width=2)), 
    secondary_y=True
)

fig2.update_yaxes(title_text="Emas (% Perubahan)", secondary_y=False)
fig2.update_yaxes(title_text="Rupiah (% Perubahan)", secondary_y=True)

fig2.update_layout(
    title_text=f"Perbandingan Perubahan {aggregation_freq} Emas vs Rupiah",
    height=500,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("⚠️ **Penting:** Data makro di atas adalah **data simulasi** yang di-*generate* secara matematis dan disimpan ke database Anda. Untuk data riil, Anda perlu mengintegrasikan dengan API data finansial yang kredibel.")
