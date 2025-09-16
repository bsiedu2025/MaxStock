# D:\Docker\BrokerSummary\app\Dockerfile
# Menggunakan base image Python yang lebih lengkap untuk build tools
FROM python:3.9-bullseye

# Menetapkan direktori kerja di dalam container
WORKDIR /app

# Instal dependensi sistem dasar yang mungkin dibutuhkan
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    wget \
    unzip \
    # Tidak perlu libta-lib-dev atau kompilasi manual TA-Lib C lagi
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Menyalin file requirements.txt terlebih dahulu
COPY requirements.txt requirements.txt

# Menginstal dependensi Python
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Menyalin seluruh isi direktori 'app' (lokal) ke direktori kerja '/app' (container)
COPY . .

# Memberitahu Docker bahwa container akan listen di port 8501
EXPOSE 8501

# Perintah untuk menjalankan aplikasi Streamlit saat container dimulai
CMD ["streamlit", "run", "app_main.py"]
