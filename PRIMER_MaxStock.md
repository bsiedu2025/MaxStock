# MaxStock – Project Primer

Gunakan dokumen ini sebagai **primer** saat memulai sesi baru agar asisten langsung “nyambung” konteks proyek.

---

## 🔧 Ringkasan Proyek
- **Nama:** MaxStock  
- **Stack:** Streamlit + MySQL (Aiven) + GitHub Actions + yfinance + pandas  
- **Repo GitHub:** `bsiedu2025/MaxStock` (branch: `master`)  
- **App URL (Streamlit Cloud):** `https://maxstock-2025.streamlit.app/`

---

## 🗄️ Database (Aiven MySQL)
**Secrets/ENV yang _wajib_ tersedia (nama persis):**
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_SSL_CA`  
  - `DB_SSL_CA` diisi **PEM certificate lengkap** dari Aiven (mulai `-----BEGIN CERTIFICATE-----` sampai `-----END CERTIFICATE-----`).

**Tabel utama** `stock_prices_history`:
```sql
CREATE TABLE IF NOT EXISTS stock_prices_history (
  Ticker     VARCHAR(32)  NOT NULL,
  Tanggal    DATE         NOT NULL,
  Open       DECIMAL(19,4)     NULL,
  High       DECIMAL(19,4)     NULL,
  Low        DECIMAL(19,4)     NULL,
  Close      DECIMAL(19,4)     NULL,
  Volume     BIGINT            NULL,
  OI         BIGINT            NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (Ticker, Tanggal)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

**Tabel baru (RAW) untuk upload EOD** `eod_prices_raw`:
```sql
CREATE TABLE IF NOT EXISTS eod_prices_raw (
  Ticker     VARCHAR(32)  NOT NULL,
  Tanggal    DATE         NOT NULL,
  Open       DECIMAL(19,4)     NULL,
  High       DECIMAL(19,4)     NULL,
  Low        DECIMAL(19,4)     NULL,
  Close      DECIMAL(19,4)     NULL,
  Volume     BIGINT            NULL,
  OI         BIGINT            NULL,
  SourceFile VARCHAR(255)      NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (Ticker, Tanggal)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

> **Anti-duplikat:** Gunakan `INSERT IGNORE` dan PK `(Ticker, Tanggal)`.

---

## 📑 Struktur Halaman Streamlit (`app/pages/`)
- `1_Harga_Saham.py` — Visualisasi harga & indikator.
- `2_Update_Data_Harga_Saham.py` — Unduh via yfinance → simpan DB.
- `3_Konsol_Database.py` — Eksekusi SQL langsung di DB.
- `4_Sinyal_MACD.py` — Scan MACD **bulk** (1 query) + link ke `/Harga_Saham?ticker=...`.
- `5_Upload_EOD.py` — Upload CSV **RAW** (`<date>`, `<ticker>`, `<open>`, `<high>`, `<low>`, `<close>`, `<volume>`, `<oi>`) ke **`eod_prices_raw`**, auto-map header & **anti-duplikat**.

**Entry point:** `app/app_main.py` (mapping manual halaman).  
Tambahkan ke menu jika perlu, mis:
```python
PAGE_FILES = {
  "Harga Saham": APP_DIR / "1_Harga_Saham.py",
  "Update Data Harga Saham": APP_DIR / "2_Update_Data_Harga_Saham.py",
  "Konsol Database": APP_DIR / "3_Konsol_Database.py",
  "Sinyal MACD": APP_DIR / "4_Sinyal_MACD.py",
  "Upload EOD (CSV)": APP_DIR / "5_Upload_EOD.py",
}
```

---

## 🚀 Otomasi Harian (GitHub Actions)
**Workflow:** `.github/workflows/daily-update.yml`  
Jalan **setiap hari kerja** pukul **07:00 WIB** (00:00 UTC).

```yaml
name: Daily Stock Update

on:
  workflow_dispatch: {}
  schedule:
    - cron: "0 0 * * 1-5"   # 07:00 WIB (00:00 UTC) Senin–Jumat

jobs:
  update:
    runs-on: ubuntu-latest
    env:
      DB_HOST: ${{ secrets.DB_HOST }}
      DB_PORT: ${{ secrets.DB_PORT }}
      DB_NAME: ${{ secrets.DB_NAME }}
      DB_USER: ${{ secrets.DB_USER }}
      DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
      DB_SSL_CA: ${{ secrets.DB_SSL_CA }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }

      - name: Print time
        run: |
          echo "UTC:" && date -u
          TZ=Asia/Jakarta date

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r app/requirements.txt

      - name: Run batch updater (5 hari perdagangan)
        run: |
          python app/batch_update.py --period 5d --suffix .JK --max-tickers 0
```

**Secrets GitHub yang harus ada:**
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_SSL_CA`  
  (isi **per-key**, **tanpa** `DB_HOST =`, paste PEM `DB_SSL_CA` multiline apa adanya)

---

## 🧪 Cara Ngetes
- **Manual run:** tab **Actions** → pilih workflow → **Run workflow**.  
- **Verifikasi DB:**
  ```sql
  SELECT MAX(Tanggal) AS Terbaru FROM stock_prices_history;
  ```
- **Jika lambat:** pakai `--period 5d`, batasi `--max-tickers`, dan gunakan koneksi pool (sudah di `db_utils.py`).

---

## 🔒 Keamanan
- **Jangan commit secrets**. Simpan di **Streamlit Secrets** & **GitHub Secrets**.
- Jika password pernah diekspos, **rotate** di Aiven lalu update di Secrets.

---

## ✅ To-Do / Next
Tulis target kamu di sini agar asisten eksekusi step-by-step:
- [ ] (contoh) Tambah halaman screening RSI
- [ ] (contoh) Split workflow jadi 2 batch (pukul 07:00 & 07:10 WIB)
- [ ] (contoh) Tambah chart volume profile di Harga_Saham

---

## 🔁 Cara Menggunakan Primer Ini di Sesi Baru
1. Buka **chat baru**.  
2. Paste *primer* ini sebagai pesan pertama.  
3. Tambahkan to-do / masalah yang ingin dikerjakan hari ini.
