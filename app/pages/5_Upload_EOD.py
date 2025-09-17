# app/pages/5_Upload_EOD.py
# -*- coding: utf-8 -*-
"""
Upload CSV EOD ke tabel BARU: eod_prices_raw
- Menerima header seperti: <date>, <ticker>, <open>, <high>, <low>, <close>, <volume>, <oi>
- Membersihkan & memetakan header otomatis
- Menghapus duplikat DALAM FILE (Ticker, Tanggal)
- INSERT IGNORE (batched) ke eod_prices_raw (PK: Ticker, Tanggal)
- Membuat tabel jika belum ada
"""
import io
import re
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from db_utils import get_db_connection

st.set_page_config(page_title="Upload EOD (RAW) → eod_prices_raw", page_icon="🗂️", layout="wide")
st.title("🗂️ Upload EOD (RAW) → eod_prices_raw")

st.caption(
    "Unggah file CSV EOD (header `<date>`, `<ticker>`, `<open>`, `<high>`, `<low>`, "
    "`<close>`, `<volume>`, `<oi>`). Data dibersihkan, duplikat dalam file dihapus, "
    "kemudian **INSERT IGNORE** ke tabel **eod_prices_raw** (PK: `Ticker,Tanggal`)."
)

# =========================
# Konfigurasi
# =========================
TABLE_NAME = "eod_prices_raw"
REQUIRED_COLS = ["Ticker", "Tanggal", "Close"]  # minimal
ALL_COLS = ["Ticker", "Tanggal", "Open", "High", "Low", "Close", "Volume", "OI", "SourceFile"]
BATCH_SIZE = 1000

with st.expander("Struktur kolom & contoh minimal", expanded=False):
    st.markdown(
        """
**Header didukung (case-insensitive; tanda `<` `>` diabaikan):**  
- `date` → **Tanggal**  
- `ticker`/`symbol` → **Ticker**  
- `open`, `high`, `low`, `close`, `volume`, `oi`

**Contoh minimal:**
```csv
<date>,<ticker>,<close>
2025-09-15,BBCA.JK,10000
```
        """
    )

# -------------------------------------------------------------------
# DDL: pastikan tabel baru ada
# -------------------------------------------------------------------
def create_eod_raw_table_if_not_exists():
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
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
    ) ENGINE=InnoDB
      DEFAULT CHARSET = utf8mb4
      COLLATE = utf8mb4_unicode_ci;
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(ddl)
        conn.commit()
    finally:
        cur.close(); conn.close()

# -------------------------------------------------------------------
# Parsing & pembersihan
# -------------------------------------------------------------------
def _normalize_header(name: str) -> str:
    """Hilangkan <>, spasi, karakter non-alfanumerik, lower-case."""
    s = name.strip().lower()
    s = s.replace("<", "").replace(">", "")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s

def _auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    # normalisasi header dulu
    df = df.rename(columns={c: _normalize_header(c) for c in df.columns})

    aliases = {
        "ticker":  ["ticker", "symbol", "kode", "code"],
        "tanggal": ["tanggal", "date", "tradedate"],
        "open":    ["open", "o"],
        "high":    ["high", "h"],
        "low":     ["low", "l"],
        "close":   ["close", "c", "closing"],
        "volume":  ["volume", "vol", "v"],
        "oi":      ["oi", "openinterest"],
        "sourcefile": ["sourcefile", "filename"],
    }

    # temukan mapping
    mapping: Dict[str, str] = {}
    for std, names in aliases.items():
        for n in names:
            if n in df.columns and std not in mapping.values():
                mapping[n] = std
                break

    # ubah ke nama final (Title Case kecuali Tanggal)
    rename = {}
    for src, std in mapping.items():
        if std == "tanggal":
            rename[src] = "Tanggal"
        else:
            rename[src] = std.capitalize()  # Ticker/Open/High/Low/Close/Volume/Oi/Sourcefile

    df = df.rename(columns=rename)
    # konsisten: gunakan "OI" & "SourceFile" sebagai final
    if "Oi" in df.columns: df = df.rename(columns={"Oi": "OI"})
    if "Sourcefile" in df.columns: df = df.rename(columns={"Sourcefile": "SourceFile"})
    return df

def _clean_and_validate(df: pd.DataFrame, source_name: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {
        "rows_raw": len(df),
        "rows_after_required": 0,
        "rows_after_parse": 0,
        "intra_file_duplicates_dropped": 0,
        "unique_keys": 0,
    }

    # map header
    df = _auto_map_columns(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang: {missing}. Kolom tersedia: {list(df.columns)}")

    # tambah kolom yang belum ada
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # koersi tipe
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df.dropna(subset=["Ticker", "Tanggal"])

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").round().astype("Int64")
    df["OI"] = pd.to_numeric(df["OI"], errors="coerce").round().astype("Int64")

    stats["rows_after_required"] = len(df)

    # tanggal ke date
    df["Tanggal"] = df["Tanggal"].dt.date

    # hapus duplikat dalam file
    before = len(df)
    df = df.drop_duplicates(subset=["Ticker", "Tanggal"], keep="last")
    stats["intra_file_duplicates_dropped"] = before - len(df)
    stats["rows_after_parse"] = len(df)
    stats["unique_keys"] = len(df)

    # set SourceFile
    if source_name:
        df["SourceFile"] = source_name

    df = df.sort_values(["Ticker", "Tanggal"]).reset_index(drop=True)
    df = df[ALL_COLS]
    return df, stats

# -------------------------------------------------------------------
# DB helpers
# -------------------------------------------------------------------
def _insert_ignore_bulk(conn, rows: List[Tuple]) -> int:
    if not rows:
        return 0
    sql = f"""
        INSERT IGNORE INTO {TABLE_NAME}
        (Ticker, Tanggal, Open, High, Low, Close, Volume, OI, SourceFile)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    cur = conn.cursor()
    inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i:i+BATCH_SIZE]
        cur.executemany(sql, chunk)
        inserted += cur.rowcount if cur.rowcount != -1 else 0
        conn.commit()
    cur.close()
    return inserted

# ========================= UI =========================
uploaded = st.file_uploader("Pilih file CSV EOD (RAW)", type=["csv"])
if uploaded is None:
    st.info("Unggah file `.csv` dulu. Contoh: EOD.csv (dengan header `<date>`, `<ticker>`, ...).")
    st.stop()

# buat tabel kalau belum ada
create_eod_raw_table_if_not_exists()

# Baca CSV (coba utf-8 lalu fallback latin-1)
content = uploaded.read()
try:
    df_raw = pd.read_csv(io.BytesIO(content))
except UnicodeDecodeError:
    df_raw = pd.read_csv(io.BytesIO(content), encoding="latin-1")

st.subheader("Pratinjau")
st.dataframe(df_raw.head(20), use_container_width=True)

try:
    df, stats = _clean_and_validate(df_raw, source_name=uploaded.name)
except Exception as e:
    st.error(f"Gagal memproses CSV: {e}")
    st.stop()

st.markdown("### Ringkasan sebelum upload")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Baris mentah", stats["rows_raw"])
c2.metric("Setelah validasi", stats["rows_after_required"])
c3.metric("Setelah parse/clean", stats["rows_after_parse"])
c4.metric("Duplikasi dalam file", stats["intra_file_duplicates_dropped"])
c5.metric("Kunci unik (Ticker,Tanggal)", stats["unique_keys"])

with st.expander("Lihat data siap upload", expanded=False):
    st.dataframe(df, use_container_width=True)

if st.button(f"🚀 Upload ke Tabel {TABLE_NAME} (tanpa duplikat)", type="primary", use_container_width=True):
    rows: List[Tuple] = []
    for _, r in df.iterrows():
        rows.append((
            r["Ticker"], r["Tanggal"],
            None if pd.isna(r["Open"]) else float(r["Open"]),
            None if pd.isna(r["High"]) else float(r["High"]),
            None if pd.isna(r["Low"])  else float(r["Low"]),
            None if pd.isna(r["Close"]) else float(r["Close"]),
            None if pd.isna(r["Volume"]) else int(r["Volume"]),
            None if pd.isna(r["OI"]) else int(r["OI"]),
            None if pd.isna(r["SourceFile"]) else str(r["SourceFile"]),
        ))

    with st.spinner("Mengunggah data (INSERT IGNORE)…"):
        conn = get_db_connection()
        try:
            inserted = _insert_ignore_bulk(conn, rows)
        finally:
            conn.close()

    st.success(f"Selesai upload ke tabel {TABLE_NAME}!")
    m1, m2 = st.columns(2)
    m1.metric("Berhasil INSERT", inserted)
    m2.metric("Duplikasi (lewat karena PK)", max(stats["unique_keys"] - inserted, 0))

    st.toast(f"Upload RAW selesai (tabel {TABLE_NAME}).", icon="✅")
