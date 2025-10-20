# -*- coding: utf-8 -*-
# app/pages/6_Pergerakan_Asing_FF.py (Dashboard Makro Baru: Emas & Rupiah dari MariaDB)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import os
import tempfile
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
import time
import requests 
import math

st.set_page_config(page_title="💰 Historis Emas & Rupiah", page_icon="📈", layout="wide")
st.title("💰 Historis Emas & Nilai Tukar Rupiah (MariaDB)")
st.caption(
    "Menampilkan data historis harga emas dunia (USD/oz) yang diambil **langsung dari Stooq** dan Nilai Tukar Rupiah (manual input) "
    "yang tersimpan di tabel `macro_data` database Anda."
)

# Inisialisasi session state
if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False
if 'sheet_id' not in st.session_state: # Dibiarkan agar tidak error, tapi tidak dipakai
    st.session_state.sheet_id = "" 
    
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
# Data Fetchers (Emas & Rupiah)
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_latest_trade_date() -> datetime.date:
    """Mencari tanggal terakhir data di database."""
    try:
        with engine.connect() as con:
            q = text(f"SELECT MAX(trade_date) FROM {TABLE_NAME}")
            max_date = con.execute(q).scalar()
            return max_date if max_date else datetime(1970, 1, 1).date()
    except Exception:
        return datetime(1970, 1, 1).date()

@st.cache_data(ttl=60)
def get_last_idr_rate() -> float:
    """Mencari kurs Rupiah terakhir di database."""
    try:
        with engine.connect() as con:
            q = text(f"SELECT IDR_USD FROM {TABLE_NAME} ORDER BY trade_date DESC LIMIT 1")
            return con.execute(q).scalar()
    except Exception:
        return 15000.0


def fetch_gold_from_stooq(full_history: bool = False) -> pd.DataFrame:
    """Mengambil data Emas historis dari Stooq."""
    stooq_url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    
    try:
        df_gold = pd.read_csv(stooq_url, index_col='Date', parse_dates=True)
        
        df_gold = df_gold[['Close']].copy()
        df_gold.columns = ['Gold_USD']
        df_gold = df_gold.sort_index(ascending=True).reset_index()
        
        df_gold.rename(columns={'Date': 'trade_date'}, inplace=True)
        df_gold['trade_date'] = df_gold['trade_date'].dt.date
        df_gold.dropna(inplace=True)
        
        return df_gold
        
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal mengambil data dari Stooq. Error koneksi: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data Emas Stooq: {e}")
        return pd.DataFrame()

def generate_idr_rate_today(last_rate) -> float:
    """Menghasilkan kurs Rupiah simulasi hari ini berdasarkan kurs terakhir."""
    np.random.seed(int(time.time())) 
    change = np.random.normal(0.05, 5)
    new_rate = last_rate * (1 + change / 20000)
    return float(new_rate)


def generate_idr_rate_history(start_date: datetime.date, end_date: datetime.date, initial_rate: float) -> pd.DataFrame:
    """Menghasilkan DataFrame Kurs Rupiah simulasi sepanjang periode yang dibutuhkan."""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    df = pd.DataFrame(index=dates)
    
    # Kurs Rupiah (IDR/USD) - Menggunakan simulasi
    np.random.seed(int(time.time())) 
    idr_rate = [initial_rate]
    for _ in range(1, len(dates)):
        # Pastikan rate awal di tahun 1970 lebih rendah (sekitar 400an)
        if start_date.year < 1980:
             change = np.random.normal(0.5, 5) # Perubahan yang lebih besar untuk tren naik jangka panjang
        else:
             change = np.random.normal(0.05, 5)
        
        idr_rate.append(idr_rate[-1] * (1 + change / 10000)) 
        
    df['IDR_USD'] = np.array(idr_rate)
    
    df = df.reset_index().rename(columns={'index': 'trade_date'})
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
    return df[['trade_date', 'IDR_USD']]


def prepare_full_macro_data(start_date_gold: datetime.date):
    """Menggabungkan data Emas Stooq dan data Kurs IDR (simulasi)."""
    df_gold = fetch_gold_from_stooq()
    
    if df_gold.empty:
        st.warning("Gagal mendapatkan data Emas dari Stooq. Tidak bisa melanjutkan setup.")
        return pd.DataFrame()
        
    # Pastikan start date untuk Gold dari data yang tersedia
    start_date = df_gold['trade_date'].min()
    end_date = datetime.now().date()
    
    # Generate IDR Rate sepanjang periode data Emas yang tersedia.
    # Kurs awal IDR di tahun 1970 sekitar 415 IDR/USD.
    initial_idr = 415.0 if start_date.year < 1980 else 15000.0
    
    df_idr = generate_idr_rate_history(start_date, end_date, initial_idr)
    
    # Merge kedua data berdasarkan trade_date (hanya ambil tanggal yang ada di kedua sisi)
    df_merged = pd.merge(df_gold, df_idr, on='trade_date', how='inner')
    
    return df_merged[['trade_date', 'Gold_USD', 'IDR_USD']].sort_values('trade_date').reset_index(drop=True)


def upload_full_data_to_db(df: pd.DataFrame):
    """Menyimpan data penuh ke database."""
    
    st.session_state.is_loading = True
    total_rows = len(df)
    
    with st.status("Memproses data makro...", expanded=True) as status_bar:
        try:
            status_bar.update(label="1. Menyiapkan koneksi database...", state="running")
            engine = _build_engine()
            
            # 2. Menghapus tabel lama dan membuat tabel baru (Mode CREATE)
            status_bar.update(label="2. Menghapus tabel lama dan membuat tabel baru...")
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
            
            # 3. Mengunggah data menggunakan to_sql
            status_bar.update(label=f"3. Mengunggah {total_rows} baris data Emas Stooq dan Kurs IDR ke DB...")
            
            df_temp = df.set_index('trade_date')
            df_temp.to_sql(
                name=TABLE_NAME, 
                con=engine, 
                if_exists='append', 
                index=True,
                chunksize=5000 
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

def upload_gap_data(df_gap: pd.DataFrame):
    """Menyimpan data gap ke database menggunakan mode upsert."""
    st.session_state.is_loading = True
    total_rows = len(df_gap)
    
    with st.status("Memproses data Gap Filler...", expanded=True) as status_bar:
        try:
            status_bar.update(label="1. Menyiapkan koneksi database...", state="running")
            engine = _build_engine()
            
            # Upload data gap menggunakan mode 'append' (karena kita sudah memastikan tanggalnya baru)
            status_bar.update(label=f"2. Mengunggah {total_rows} baris data gap ke DB...")
            
            df_temp = df_gap.set_index('trade_date')
            df_temp.to_sql(
                name=TABLE_NAME, 
                con=engine, 
                if_exists='append', 
                index=True,
                chunksize=500 
            )
            
            status_bar.update(label=f"✅ Pengunggahan Data Gap Selesai! ({total_rows} baris)", state="complete", expanded=False)
            
            st.cache_data.clear()
            st.success(f"Berhasil mengisi {total_rows} hari gap data makro.")

            st.rerun() 
            
        except Exception as e:
            status_bar.update(label="❌ Gagal mengunggah data gap!", state="error", expanded=True)
            st.error(f"Terjadi kesalahan database saat mengisi gap: {e}")
        finally:
            st.session_state.is_loading = False # Reset loading state


# ────────────────────────────────────────────────────────────────────────────────
# Data Fetcher & Aggregator (Sama seperti sebelumnya)
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
    
    with st.expander("🛠️ Klik di sini untuk membuat tabel `macro_data` (Setup Awal)") as exp:
        st.info("Fitur ini akan mengambil data Emas dari Stooq, menggabungkannya dengan Kurs Rupiah simulasi, dan membuat tabel Anda.")
        
        if st.button("📥 Ambil Emas Stooq & Buat Tabel", disabled=st.session_state.is_loading, type="primary"):
            df_full = prepare_full_macro_data(datetime(1970, 1, 1).date()) # Set start date ke 1970
            if not df_full.empty:
                 upload_full_data_to_db(df_full) 
            
        if st.button("🗑️ Hapus Tabel Macro Data", disabled=st.session_state.is_loading, key="delete_initial_table"):
            delete_table()
            
    st.stop()
    
# --- LOGIKA KETIKA TABEL SUDAH ADA ---
else:
    # Tampilkan tombol untuk update data simulasi 1970 atau menghapus tabel
    with st.expander("🛠️ Opsi Perawatan Data Makro (Tabel sudah ada)") as exp:
        st.info("Gunakan tombol ini untuk menghapus tabel atau memperbarui semua data (Emas dari Stooq).")
        
        col_upd, col_del = st.columns(2)
        
        with col_upd:
            if st.button("🔄 Ambil Ulang Emas Stooq & Timpa Semua Data", key="btn_replace_sheets", disabled=st.session_state.is_loading):
                df_to_upload = prepare_full_macro_data(datetime(1970, 1, 1).date())
                if not df_to_upload.empty:
                    upload_full_data_to_db(df_to_upload) # Menggunakan fungsi upload_full_data_to_db untuk menimpa total
                else:
                    st.error("Gagal mengambil data dari Stooq. Tidak ada data yang ditimpa.")

        with col_del:
            if st.button("🗑️ Hapus Tabel Macro Data", key="btn_delete_table", disabled=st.session_state.is_loading):
                delete_table()


# Tampilkan loading indicator global jika sedang proses
if st.session_state.is_loading:
    pass


# --- FORM UPDATE HARIAN DI SIDEBAR (GAP FILLER) ---
with st.sidebar:
    st.header("🔄 Isi Gap Data Otomatis")
    
    last_date = get_latest_trade_date()
    today = datetime.now().date()
    
    if last_date >= today:
        st.success("✅ Data Anda sudah terbaru hingga hari ini!")
    else:
        gap_start_date = last_date + timedelta(days=1)
        st.warning(f"⚠️ Ditemukan gap data: {gap_start_date.strftime('%Y-%m-%d')} s/d {today.strftime('%Y-%m-%d')}")
        
        if st.button("⚡ Lengkapi Gap Data (Stooq Emas & Simulasi Kurs)", disabled=st.session_state.is_loading, type="primary"):
            
            # Ambil data emas Stooq secara penuh
            df_full_gold = fetch_gold_from_stooq()
            if df_full_gold.empty:
                 st.error("Gagal mengambil data Emas dari Stooq untuk mengisi gap.")
                 st.rerun() # Refresh agar tidak stuck
            
            # Tentukan periode gap
            df_gold_gap = df_full_gold[df_full_gold['trade_date'] >= gap_start_date].copy()
            
            if df_gold_gap.empty:
                 st.info(f"Tidak ada data baru dari Stooq sejak {gap_start_date.strftime('%Y-%m-%d')}.")
            else:
                # 1. Ambil kurs terakhir di DB
                last_idr_rate = get_last_idr_rate()
                
                # 2. Tentukan range gap di data emas
                idr_gap_start = df_gold_gap['trade_date'].min()
                idr_gap_end = df_gold_gap['trade_date'].max()
                
                # 3. Generate IDR Rate hanya untuk periode gap
                df_idr_gap = generate_idr_rate_history(idr_gap_start, idr_gap_end, last_idr_rate)
                
                # 4. Gabungkan dan hapus kolom Rupiah yang lama
                df_gap_merged = pd.merge(df_gold_gap, df_idr_gap, on='trade_date', how='inner')
                
                # 5. Upload gap data
                upload_gap_data(df_gap_merged)

st.sidebar.markdown("---")


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
selected_start_date = st.sidebar.date_input("Tanggal Mulai", value=min_date_db, min_value=min_date_db, max_value=end_date, key="filter_start")
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
