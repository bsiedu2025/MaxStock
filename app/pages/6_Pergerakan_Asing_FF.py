# app/pages/6_Pergerakan_Asing_FF.py
# Disusun mengikuti struktur Analisa_Foreign_Flow.py (Plotly, SQLAlchemy, KSEI-aware).

import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="📈 Analisa Foreign Flow", page_icon="📈", layout="wide")
st.title("📈 Analisa Foreign Flow")
st.caption("Harga (Line/Candle) + Foreign Flow + Partisipasi Asing/Ritel (pakai KSEI jika tersedia).")

engine = get_engine()

# ── Ambil daftar simbol (non-FF)

with engine.connect() as con:
    syms = pd.read_sql(
        "SELECT DISTINCT base_symbol FROM eod WHERE is_foreign_flow=0 ORDER BY base_symbol",
        con,
    )["base_symbol"].tolist()

if not syms:
    st.warning("Belum ada data harga. Silakan import EOD/History terlebih dahulu.")
    st.stop()

# ── Controls
colA, colB, colC = st.columns([2, 1, 1])
with colA:
    default_idx = syms.index("BBRI") if "BBRI" in syms else 0
    symbol = st.selectbox("Pilih Saham", syms, index=default_idx)
with colB:
    period = st.selectbox("Periode", ["1M", "3M", "6M", "1Y", "ALL"], index=1)
with colC:
    price_type = st.radio("Tipe Harga", ["Line", "Candle"], horizontal=True, index=1)

win = st.slider(
    "Window Partisipasi (hari)",
    min_value=5, max_value=60, value=20, step=1,
    help="Digunakan untuk menghitung Pa/Ri berbasis rolling saat data KSEI tidak tersedia."
)

# ── Cek keberadaan tabel ksei_daily
with engine.connect() as con:
    ksei_exists = bool(
        con.execute(
            text(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = 'ksei_daily'
                """
            )
        ).scalar()
    )

# ── Filter periode
date_filter = ""
if period != "ALL":
    if period.endswith("M"):
        n = int(period[:-1])
        date_filter = f"AND p.trade_date >= CURDATE() - INTERVAL {n} MONTH"
    else:
        n = int(period[:-1])
        date_filter = f"AND p.trade_date >= CURDATE() - INTERVAL {n} YEAR"

# ── SQL: pakai JOIN ke ksei_daily jika ada
if ksei_exists:
    sql = f"""
    SELECT
      p.trade_date,
      p.base_symbol,
      p.open, p.high, p.low, p.close,
      p.volume AS volume_price,
      COALESCE(f.foreign_net, 0) AS foreign_net,
      k.foreign_pct, k.retail_pct, k.total_volume, k.total_value
    FROM eod p
    LEFT JOIN eod f
      ON f.trade_date = p.trade_date
     AND f.base_symbol = p.base_symbol
     AND f.is_foreign_flow = 1
    LEFT JOIN ksei_daily k
      ON k.trade_date  = p.trade_date
     AND k.base_symbol = p.base_symbol
    WHERE p.base_symbol = :sym
      AND p.is_foreign_flow = 0
    {date_filter}
    ORDER BY p.trade_date
    """
else:
    sql = f"""
    SELECT
      p.trade_date,
      p.base_symbol,
      p.open, p.high, p.low, p.close,
      p.volume AS volume_price,
      COALESCE(f.foreign_net, 0) AS foreign_net,
      NULL AS foreign_pct, NULL AS retail_pct, NULL AS total_volume, NULL AS total_value
    FROM eod p
    LEFT JOIN eod f
      ON f.trade_date = p.trade_date
     AND f.base_symbol = p.base_symbol
     AND f.is_foreign_flow = 1
    WHERE p.base_symbol = :sym
      AND p.is_foreign_flow = 0
    {date_filter}
    ORDER BY p.trade_date
    """

with engine.connect() as con:
    df = pd.read_sql(text(sql), con, params={"sym": symbol})

if df.empty:
    st.warning("Data tidak tersedia untuk simbol ini pada periode terpilih.")
    st.stop()

# ── Feature engineering
df["trade_date"] = pd.to_datetime(df["trade_date"])
df["MA20"] = df["close"].rolling(20, min_periods=1).mean()

# ── Partisipasi: pakai KSEI jika ada persentase; kalau tidak, fallback proksi FF/Volume rolling
if df["foreign_pct"].notna().any():
    df["Pa_pct"] = df["foreign_pct"].astype(float).clip(0, 100)
    if df["retail_pct"].notna().any():
        df["Ri_pct"] = df["retail_pct"].astype(float).clip(0, 100)
    else:
        df["Ri_pct"] = (100 - df["Pa_pct"]).clip(0, 100)
else:
    vol_roll = df["volume_price"].abs().rolling(win, min_periods=1).sum()
    ff_roll_abs = df["foreign_net"].abs().rolling(win, min_periods=1).sum()
    df["Pa_pct"] = (100.0 * (ff_roll_abs / vol_roll)).clip(0, 100).fillna(0.0)
    df["Ri_pct"] = (100.0 - df["Pa_pct"]).clip(0, 100)

# ── Plot: 3 subplot (Harga | FF bar | Pa/Ri)
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    row_heights=[0.6, 0.25, 0.15],
    subplot_titles=(f"Harga — {symbol}", "Daily Foreign Flow (Net Volume)", "Partisipasi Asing & Ritel (%)")
)

# Row 1: Harga
if price_type == "Line":
    fig.add_trace(
        go.Scatter(x=df["trade_date"], y=df["close"], name="Close", mode="lines"),
        row=1, col=1
    )
else:
    fig.add_trace(
        go.Candlestick(
            x=df["trade_date"],
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Harga"
        ),
        row=1, col=1
    )

# MA20
fig.add_trace(
    go.Scatter(x=df["trade_date"], y=df["MA20"], name="MA20", mode="lines"),
    row=1, col=1
)

# Row 2: Foreign Net (bar)
fig.add_trace(
    go.Bar(x=df["trade_date"], y=df["foreign_net"], name="Foreign Net", marker_line_width=0, opacity=0.85),
    row=2, col=1
)

# Row 3: Pa/Ri
fig.add_trace(
    go.Scatter(x=df["trade_date"], y=df["Pa_pct"], name="Pa (%)", mode="lines"),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(x=df["trade_date"], y=df["Ri_pct"], name="Ri (%)", mode="lines"),
    row=3, col=1
)
fig.add_hline(y=50, line_dash="dot", line_width=1, row=3, col=1)

fig.update_layout(
    height=820,
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_rangeslider_visible=False,
    hovermode="x unified"
)
fig.update_yaxes(title_text="Harga", row=1, col=1)
fig.update_yaxes(title_text="Foreign Net", row=2, col=1)
fig.update_yaxes(title_text="%", range=[0, 100], row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# ── Ringkasan metrik
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Close", f"{df['close'].iloc[-1]:,.0f}")
c2.metric(f"Sum FF ({win}D)", f"{df['foreign_net'].tail(win).sum():,.0f}")
c3.metric("Pa(%) terakhir", f"{df['Pa_pct'].iloc[-1]:.2f}")
c4.metric("Ri(%) terakhir", f"{df['Ri_pct'].iloc[-1]:.2f}")

with st.expander("Tabel (akhir 250 baris)"):
    show_cols = [
        "trade_date", "open", "high", "low", "close",
        "MA20", "foreign_net", "volume_price", "Pa_pct", "Ri_pct"
    ]
    st.dataframe(df[show_cols].tail(250), use_container_width=True)
