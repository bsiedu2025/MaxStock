# D:\Docker\BrokerSummary\app\app_main.py
import streamlit as st
# from db_utils import create_tables_if_not_exist, get_db_connection

# --- Konfigurasi Halaman Streamlit ---
# Ini harus menjadi perintah Streamlit pertama yang dijalankan, dan hanya sekali per aplikasi.
st.set_page_config(
    page_title="Analisis Broker Saham",
    page_icon="📊",
    layout="wide",  # 'centered' atau 'wide'
    initial_sidebar_state="expanded",  # 'auto', 'expanded', 'collapsed'
    menu_items={
        'Get Help': 'https://www.example.com/help', # Ganti dengan URL bantuan Anda
        'Report a bug': "https://www.example.com/bug", # Ganti dengan URL laporan bug
        'About': """
        ## Analisis Broker Saham BRIS
        Aplikasi ini dibuat untuk menganalisis data transaksi broker saham.
        Versi 1.0 (Docker & MariaDB)
        """
    }
)

def main():
    """Fungsi utama untuk menjalankan aplikasi Streamlit."""

    st.title("📊 Analisis Aktivitas Broker Saham")
    st.caption(f"Terhubung ke database MariaDB.")

    # --- Inisialisasi Database ---
    # Memastikan tabel ada saat aplikasi pertama kali dijalankan atau jika init.sql tidak berjalan
    # Ini bisa dipindahkan ke bagian yang hanya dijalankan sekali jika diperlukan optimasi
    # Namun, untuk sekarang, ini memastikan tabel selalu ada.
    with st.spinner("Memeriksa dan menginisialisasi database jika perlu..."):
        create_tables_if_not_exist() # Fungsi dari db_utils.py

    st.sidebar.success("Pilih halaman di atas.")

    # Konten halaman utama (jika ada, sebelum pengguna memilih halaman dari sidebar)
    st.markdown("---")
    st.header("Selamat Datang!")
    st.markdown(
        """
        Aplikasi ini dirancang untuk membantu Anda menganalisis data transaksi broker saham.
        Gunakan sidebar di sebelah kiri untuk navigasi ke halaman:

        - **Input Data**: Untuk memasukkan atau memperbarui data transaksi dan informasi broker.
        - **Analisis Broker**: Untuk melihat visualisasi dan analisis mendalam dari data yang ada.

        Pastikan layanan database MariaDB Anda berjalan jika Anda menjalankan ini secara lokal di luar Docker.
        Jika menggunakan Docker, pastikan container MariaDB sudah aktif.
        """
    )
    st.markdown("---")

    # Cek koneksi database awal untuk memberikan feedback ke pengguna
    conn = get_db_connection()
    if conn:
        st.sidebar.info("Status Database: Terhubung ✅")
        conn.close()
    else:
        st.sidebar.error("Status Database: Gagal Terhubung ❌")
        st.warning(
            """
            **Peringatan Koneksi Database!**
            Aplikasi tidak dapat terhubung ke database MariaDB.
            Beberapa fitur mungkin tidak berfungsi dengan benar.
            Pastikan layanan MariaDB berjalan dan konfigurasi koneksi di `docker-compose.yml` atau `db_utils.py` sudah benar.
            Jika Anda baru saja memulai container Docker, tunggu beberapa saat hingga MariaDB siap.
            """
        )

if __name__ == "__main__":
    main()
