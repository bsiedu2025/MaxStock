# app/pages/4_Sinyal_MACD.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from ta.trend import MACD
from db_utils import (
    get_distinct_tickers_from_price_history_with_suffix,
    fetch_stock_prices_from_db,
)

st.set_page_config(page_title="Sinyal MACD", page_icon="📉", layout="wide")
st.title("Sinyal MACD 📉")

tickers = get_distinct_tickers_from_price_history_with_suffix(".JK")
if not tickers:
    st.warning("Belum ada ticker di database. Silakan isi data lewat halaman 'Update Data Harga Saham'.")
    st.stop()

ticker = st.selectbox("Pilih ticker", tickers, index=0)
df = fetch_stock_prices_from_db(ticker)
if df is None or df.empty:
    st.warning(f"Data {ticker} kosong.")
    st.stop()

# Hitung MACD
macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
df_out = pd.DataFrame({
    "MACD": macd.macd(),
    "Signal": macd.macd_signal(),
    "Hist": macd.macd_diff(),
}, index=df.index)

st.line_chart(df_out[["MACD", "Signal"]])
st.bar_chart(df_out[["Hist"]])
