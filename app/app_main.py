# D:\Docker\BrokerSummary\app\app_main.py
import streamlit as st
from db_utils import get_connection

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Analisis Broker Saham",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': """
        ## Analisis Broker Saham BRIS
        Aplikasi ini dibuat untuk menganalisis data transaksi broker saham.
        Versi 1.0 (Docker & MariaDB)
        """
    }
)

def main():
    """Fungsi utama untuk menjalankan aplikasi Streamlit."""

    st.title("📊 Max Stock")
    st.caption(f"Terhubung ke database Supabase.")

    # st.sidebar.success("Pilih halaman di atas.") # Pindahkan pesan ini ke bawah untuk menghindari error indentasi

    # Konten halaman utama
    st.markdown("---")
    st.header("Selamat Datang!")
    st.markdown(
        """
        Aplikasi ini dirancang untuk membantu Anda menganalisis data transaksi broker saham.
        Gunakan sidebar di sebelah kiri untuk navigasi ke halaman:

        - **Input Data**: Untuk memasukkan atau memperbarui data transaksi dan informasi broker.
        - **Analisis Broker**: Untuk melihat visualisasi dan analisis mendalam dari data yang ada.
        """
    )
    st.markdown("---")

    # Cek koneksi database awal untuk memberikan feedback ke pengguna
    conn = get_connection()
    if conn:
        st.sidebar.info("Status Database: Terhubung ✅")
        conn.close()
    else:
        st.sidebar.error("Status Database: Gagal Terhubung ❌")
        st.warning(
            """
            **Peringatan Koneksi Database!**
            Aplikasi tidak dapat terhubung ke database.
            Beberapa fitur mungkin tidak berfungsi dengan benar.
            """
        )
    
    st.sidebar.success("Pilih halaman di atas.")

if __name__ == "__main__":
    main()