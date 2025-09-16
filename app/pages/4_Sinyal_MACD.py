# app/pages/4_Sinyal_MACD.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from ta.trend import MACD
from db_utils import get_db_connection

st.set_page_config(page_title="Sinyal MACD", page_icon="📉", layout="wide")
st.title("Sinyal Beli MACD (Histogram Hijau)")

st.caption(
    "Scan cepat: ambil semua harga penutupan **sekali query** (X hari terakhir) "
    "lalu hitung MACD per-ticker di pandas. Lebih cepat dibanding query per-ticker."
)

days = st.slider("Rentang hari historis", 60, 400, 200, step=20)
min_rows = st.slider("Minimal baris per ticker (agar bisa hitung MACD)", 30, 200, 60, step=10)

@st.cache_data(ttl=600, show_spinner="Memuat data harga dari database…")
def load_recent_close(days_: int) -> pd.DataFrame:
    sql = """
        SELECT Ticker, Tanggal, Close
        FROM stock_prices_history
        WHERE Tanggal >= CURDATE() - INTERVAL %s DAY
        ORDER BY Ticker, Tanggal
    """
    conn = get_db_connection()
    try:
        df = pd.read_sql(sql, conn, params=(days_,))
    finally:
        conn.close()
    if df.empty:
        return df
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    df = df.dropna(subset=["Tanggal"])
    return df

if st.button("🔍 Cari Sinyal MACD Terbaru", type="primary", use_container_width=True):
    df = load_recent_close(days)
    if df.empty:
        st.warning("Tidak ada data dalam rentang hari yang dipilih.")
        st.stop()

    df = df[["Ticker", "Tanggal", "Close"]].sort_values(["Ticker", "Tanggal"])

    progress = st.progress(0.0, text=f"Menganalisis 0/{df['Ticker'].nunique()} ticker…")
    found = []

    total = df["Ticker"].nunique()
    for i, (ticker, g) in enumerate(df.groupby("Ticker"), start=1):
        progress.progress(i / total, text=f"Menganalisis {i}/{total}: {ticker}")
        g = g.dropna(subset=["Close"])
        if len(g) < min_rows:
            continue

        g = g.set_index("Tanggal").sort_index()
        try:
            macd = MACD(close=g["Close"], window_slow=26, window_fast=12, window_sign=9)
            hist = macd.macd_diff()
            if len(hist) < 2:
                continue
            # Transisi histogram dari negatif → positif
            if (hist.iloc[-2] < 0) and (hist.iloc[-1] > 0):
                found.append(
                    {
                        "Ticker": ticker,
                        "Tanggal": g.index[-1].date(),
                        "Close_Terakhir": float(g["Close"].iloc[-1]),
                        "Hist_-1": float(hist.iloc[-2]),
                        "Hist_0": float(hist.iloc[-1]),
                    }
                )
        except Exception:
            continue

    st.subheader("Hasil")
    if not found:
        st.info("Belum ada sinyal beli (histogram hijau) pada periode ini.")
    else:
        out = pd.DataFrame(found).sort_values("Ticker").reset_index(drop=True)
        st.dataframe(out, use_container_width=True)
