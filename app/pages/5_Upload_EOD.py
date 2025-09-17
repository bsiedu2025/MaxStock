# app/pages/5_Upload_EOD.py
# -*- coding: utf-8 -*-
import io
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from db_utils import get_db_connection

st.set_page_config(page_title="Upload EOD CSV", page_icon="📤", layout="wide")
st.title("📤 Upload EOD (End Of Day) ke Database")
st.caption(
    "Unggah file CSV berisi data EOD. Sistem akan **membersihkan data**, "
    "**menghapus duplikat dalam file**, dan melakukan **INSERT IGNORE** ke tabel "
    "`stock_prices_history` (PK: `Ticker, Tanggal`) agar **tidak terjadi duplikasi di DB**."
)

# =========================
# Konfigurasi
# =========================
REQUIRED_COLS = ["Ticker", "Tanggal", "Close"]  # kolom minimal
ALL_COLS = ["Ticker", "Tanggal", "Open", "High", "Low", "Close", "Volume"]
BATCH_SIZE = 1000

with st.expander("Format CSV yang diterima", expanded=False):
    st.markdown(
        """
**Header kolom yang didukung (case-insensitive, akan dipetakan otomatis):**
- `Ticker` (alias: `Symbol`)
- `Tanggal` (alias: `Date`, `TradeDate`) — format bebas, akan diparsing
- `Open`, `High`, `Low`, `Close`
- `Volume`

**Contoh minimal:**
```csv
Ticker,Tanggal,Close
BBCA.JK,2025-09-15,10000
BBRI.JK,2025-09-15,6000
```
        """
    )

def _auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map kolom input ke nama standar ALL_COLS secara case-insensitive."""
    mapping: Dict[str, str] = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}

    aliases = {
        "ticker": ["ticker", "symbol", "kode", "code"],
        "tanggal": ["tanggal", "date", "tradedate"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "closing"],
        "volume": ["volume", "vol", "v"],
    }

    for std, alist in aliases.items():
        for a in alist:
            if a in lower_cols:
                mapping[lower_cols[a]] = std.capitalize() if std != "tanggal" else "Tanggal"
                if std in ["open", "high", "low", "close", "volume"]:
                    mapping[lower_cols[a]] = std.capitalize()
                break

    df2 = df.rename(columns=mapping)
    return df2

def _clean_and_validate(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Bersihkan df, pastikan kolom wajib, normalisasi tipe data, hapus duplikat (Ticker,Tanggal)."""
    stats = {
        "rows_raw": len(df),
        "rows_after_required": 0,
        "rows_after_parse": 0,
        "intra_file_duplicates_dropped": 0,
        "unique_keys": 0,
    }

    # Peta kolom otomatis
    df = _auto_map_columns(df)

    # Pastikan kolom wajib minimal ada
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang: {missing}. Kolom tersedia: {list(df.columns)}")

    # Tambah kolom yang tidak ada dengan nilai NaN
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # Koersi tipe
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df.dropna(subset=["Ticker", "Tanggal"])

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").round().astype("Int64")

    stats["rows_after_required"] = len(df)

    df["Tanggal"] = df["Tanggal"].dt.date

    # Hapus duplikat dalam file
    before = len(df)
    df = df.drop_duplicates(subset=["Ticker", "Tanggal"], keep="last")
    stats["intra_file_duplicates_dropped"] = before - len(df)
    stats["rows_after_parse"] = len(df)
    stats["unique_keys"] = len(df)

    df = df.sort_values(["Ticker", "Tanggal"]).reset_index(drop=True)
    df = df[ALL_COLS]

    return df, stats

def _count_existing_keys(conn, pairs: List[Tuple[str, datetime]]) -> int:
    """Hitung berapa keys (Ticker,Tanggal) dari pairs yang sudah ada di DB (batched)."""
    if not pairs:
        return 0
    total = 0
    for i in range(0, len(pairs), 1000):
        chunk = pairs[i : i + 1000]
        placeholders = ",".join(["(%s,%s)"] * len(chunk))
        sql = f"""
            SELECT COUNT(*) AS n
            FROM stock_prices_history
            WHERE (Ticker, Tanggal) IN ({placeholders})
        """
        flat_params: List = []
        for t, d in chunk:
            flat_params.extend([t, d])
        cur = conn.cursor()
        cur.execute(sql, flat_params)
        n = cur.fetchone()[0]
        cur.close()
        total += n
    return total

def _insert_ignore_bulk(conn, rows: List[Tuple]) -> int:
    """INSERT IGNORE bulk (executemany). Return rows inserted (affected)."""
    if not rows:
        return 0
    sql = """
        INSERT IGNORE INTO stock_prices_history
        (Ticker, Tanggal, Open, High, Low, Close, Volume)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
    """
    inserted = 0
    cur = conn.cursor()
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i : i + BATCH_SIZE]
        cur.executemany(sql, chunk)
        inserted += cur.rowcount if cur.rowcount != -1 else 0
        conn.commit()
    cur.close()
    return inserted

uploaded = st.file_uploader("Pilih file CSV EOD", type=["csv"])
if uploaded is None:
    st.info("Unggah file `.csv` terlebih dulu. Contoh: `EOD.csv`.")
    st.stop()

# Baca CSV (coba utf-8 lalu fallback latin-1)
content = uploaded.read()
try:
    df_raw = pd.read_csv(io.BytesIO(content))
except UnicodeDecodeError:
    df_raw = pd.read_csv(io.BytesIO(content), encoding="latin-1")

st.subheader("Pratinjau")
st.dataframe(df_raw.head(20), use_container_width=True)

try:
    df, stats = _clean_and_validate(df_raw)
except Exception as e:
    st.error(f"Gagal memproses CSV: {e}")
    st.stop()

st.markdown("### Ringkasan sebelum upload")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Baris mentah", stats["rows_raw"])
c2.metric("Setelah validasi", stats["rows_after_required"])
c3.metric("Setelah parse/clean", stats["rows_after_parse"])
c4.metric("Duplikat dalam file", stats["intra_file_duplicates_dropped"])
c5.metric("Kunci unik (Ticker,Tanggal)", stats["unique_keys"])

with st.expander("Lihat data siap upload", expanded=False):
    st.dataframe(df, use_container_width=True)

if st.button("🚀 Upload ke Database (tanpa duplikat)", type="primary", use_container_width=True):
    rows: List[Tuple] = []
    for _, r in df.iterrows():
        rows.append((
            r["Ticker"], r["Tanggal"],
            None if pd.isna(r["Open"]) else float(r["Open"]),
            None if pd.isna(r["High"]) else float(r["High"]),
            None if pd.isna(r["Low"])  else float(r["Low"]),
            None if pd.isna(r["Close"]) else float(r["Close"]),
            None if pd.isna(r["Volume"]) else int(r["Volume"]),
        ))

    with st.spinner("Menghitung data yang sudah ada di DB…"):
        conn = get_db_connection()
        try:
            key_pairs = [(t, d) for t, d in zip(df["Ticker"].tolist(), df["Tanggal"].tolist())]
            existing = _count_existing_keys(conn, key_pairs)
        finally:
            conn.close()

    progress = st.empty()
    progress.info("Mengunggah data (INSERT IGNORE)…")
    conn = get_db_connection()
    try:
        inserted = _insert_ignore_bulk(conn, rows)
    finally:
        conn.close()
        progress.empty()

    skipped_from_db = max(stats["unique_keys"] - inserted, 0)

    st.success("Selesai upload!")
    m1, m2, m3 = st.columns(3)
    m1.metric("Berhasil INSERT", inserted)
    m2.metric("Sudah ada di DB (terlewati)", existing)
    m3.metric("Duplikasi dalam file (dihapus)", stats["intra_file_duplicates_dropped"])

    st.toast("Upload selesai tanpa duplikat (PK + INSERT IGNORE).", icon="✅")
