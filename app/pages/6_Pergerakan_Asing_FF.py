# app/pages/6_Pergerakan_Asing_FF.py
# Self-contained: no external db module. Builds SQLAlchemy engine from Streamlit Secrets/ENV.
import os
import tempfile
from urllib.parse import quote_plus
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text, create_engine
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="📈 Analisa Foreign Flow", page_icon="📈", layout="wide")
st.title("📈 Analisa Foreign Flow")
st.caption("Harga (Line/Candle) + Foreign Flow + Partisipasi Asing/Ritel (pakai KSEI jika tersedia).")

# ── Build SQLAlchemy engine from secrets/ENV
def _build_engine():
    host = os.getenv("DB_HOST", st.secrets.get("DB_HOST", ""))
    port = int(os.getenv("DB_PORT", st.secrets.get("DB_PORT", 3306)))
    database = os.getenv("DB_NAME", st.secrets.get("DB_NAME", ""))
    user = os.getenv("DB_USER", st.secrets.get("DB_USER", ""))
    password = os.getenv("DB_PASSWORD", st.secrets.get("DB_PASSWORD", ""))
    ssl_ca = os.getenv("DB_SSL_CA", st.secrets.get("DB_SSL_CA", ""))

    pwd = quote_plus(str(password))

    connect_args = {}
    if ssl_ca and "BEGIN CERTIFICATE" in ssl_ca:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
        tmp.write(ssl_ca.encode("utf-8")); tmp.flush()
        connect_args["ssl_ca"] = tmp.name

    url = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{database}"
    eng = create_engine(url, connect_args=connect_args, pool_recycle=300, pool_pre_ping=True)
    return eng

engine = _build_engine()

# ── Helpers
def _table_exists(engine, table_name: str) -> bool:
    try:
        with engine.connect() as con:
            q = text("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = :t
            """)
            n = con.execute(q, {"t": table_name}).scalar()
            return bool(n)
    except Exception:
        return False

USE_EOD_TABLE = _table_exists(engine, "eod")
USE_KSEI = _table_exists(engine, "ksei_daily")

# ── Controls
with engine.connect() as con:
    if USE_EOD_TABLE:
        syms = pd.read_sql(
            "SELECT DISTINCT base_symbol FROM eod WHERE is_foreign_flow=0 ORDER BY base_symbol",
            con,
        )["base_symbol"].tolist()
    else:
        syms = pd.read_sql(
            """
            SELECT DISTINCT Ticker AS base_symbol
            FROM eod_prices_raw
            WHERE Ticker NOT LIKE '% FF'
            ORDER BY base_symbol
            """,
            con,
        )["base_symbol"].tolist()

if not syms:
    st.warning("Belum ada data harga. Silakan import EOD/History terlebih dahulu.")
    st.stop()

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

# ── Date filters
date_filter = ""
if period != "ALL":
    if period.endswith("M"):
        n = int(period[:-1])
        date_filter = f"AND p.trade_date >= CURDATE() - INTERVAL {n} MONTH"
    else:
        n = int(period[:-1])
        date_filter = f"AND p.trade_date >= CURDATE() - INTERVAL {n} YEAR"

# ── Main SQL
if USE_EOD_TABLE:
    if USE_KSEI:
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
else:
    # derive from eod_prices_raw
    date_filter_raw = ""
    if period != "ALL":
        if period.endswith("M"):
            n = int(period[:-1])
            date_filter_raw = f"AND Tanggal >= CURDATE() - INTERVAL {n} MONTH"
        else:
            n = int(period[:-1])
            date_filter_raw = f"AND Tanggal >= CURDATE() - INTERVAL {n} YEAR"

    if USE_KSEI:
        sql = f"""
        SELECT
            p.trade_date,
            p.base_symbol,
            p.open, p.high, p.low, p.close,
            p.volume_price,
            COALESCE(f.foreign_net, 0) AS foreign_net,
            k.foreign_pct, k.retail_pct, k.total_volume, k.total_value
        FROM
            (SELECT DATE(Tanggal) AS trade_date,
                    Ticker AS base_symbol,
                    `Open` AS open, `High` AS high, `Low` AS low, `Close` AS close,
                    Volume AS volume_price
             FROM eod_prices_raw
             WHERE Ticker = :sym
               AND Ticker NOT LIKE '% FF'
               {date_filter_raw}) AS p
        LEFT JOIN
            (SELECT DATE(Tanggal) AS trade_date,
                    TRIM(REPLACE(Ticker,' FF','')) AS base_symbol,
                    Volume AS foreign_net
             FROM eod_prices_raw
             WHERE TRIM(REPLACE(Ticker,' FF','')) = :sym
               AND Ticker LIKE '% FF'
               {date_filter_raw}) AS f
        ON f.trade_date = p.trade_date AND f.base_symbol = p.base_symbol
        LEFT JOIN ksei_daily k
          ON k.trade_date = p.trade_date AND k.base_symbol = p.base_symbol
        ORDER BY p.trade_date
        """
    else:
        sql = f"""
        SELECT
            p.trade_date,
            p.base_symbol,
            p.open, p.high, p.low, p.close,
            p.volume_price,
            COALESCE(f.foreign_net, 0) AS foreign_net,
            NULL AS foreign_pct, NULL AS retail_pct, NULL AS total_volume, NULL AS total_value
        FROM
            (SELECT DATE(Tanggal) AS trade_date,
                    Ticker AS base_symbol,
                    `Open` AS open, `High` AS high, `Low` AS low, `Close` AS close,
                    Volume AS volume_price
             FROM eod_prices_raw
             WHERE Ticker = :sym
               AND Ticker NOT LIKE '% FF'
               {date_filter_raw}) AS p
        LEFT JOIN
            (SELECT DATE(Tanggal) AS trade_date,
                    TRIM(REPLACE(Ticker,' FF','')) AS base_symbol,
                    Volume AS foreign_net
             FROM eod_prices_raw
             WHERE TRIM(REPLACE(Ticker,' FF','')) = :sym
               AND Ticker LIKE '% FF'
               {date_filter_raw}) AS f
        ON f.trade_date = p.trade_date AND f.base_symbol = p.base_symbol
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

# Partisipasi: gunakan KSEI jika ada; kalau tidak fallback proxy rolling
if df["foreign_pct"].notna().any():
    df["Pa_pct"] = pd.to_numeric(df["foreign_pct"], errors="coerce").clip(0, 100)
    if df["retail_pct"].notna().any():
        df["Ri_pct"] = pd.to_numeric(df["retail_pct"], errors="coerce").clip(0, 100)
    else:
        df["Ri_pct"] = (100 - df["Pa_pct"]).clip(0, 100)
else:
    vol_roll = pd.to_numeric(df["volume_price"], errors="coerce").abs().rolling(win, min_periods=1).sum()
    ff_roll_abs = pd.to_numeric(df["foreign_net"], errors="coerce").abs().rolling(win, min_periods=1).sum()
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
        go.Scatter(x=df["trade_date"], y=df["close"], name="Harga", mode="lines"),
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

# Row 2: Foreign Net (bar) with color by sign
colors = np.where(df["foreign_net"] > 0, "rgba(0,160,0,0.85)",
          np.where(df["foreign_net"] < 0, "rgba(220,0,0,0.85)", "rgba(160,160,160,0.6)"))
fig.add_trace(
    go.Bar(x=df["trade_date"], y=df["foreign_net"], name="Foreign Net",
           marker=dict(color=colors), marker_line_width=0, opacity=0.95),
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

# ── Bulanan KSEI (opsional jika data tersedia)
st.markdown("---")
st.subheader("📅 Ringkasan Bulanan KSEI")

if df["foreign_pct"].notna().any():
    df_mon = df.copy()
    df_mon["foreign_frac"] = (pd.to_numeric(df_mon["foreign_pct"], errors="coerce") / 100.0)
    df_mon["retail_frac"]  = 1.0 - df_mon["foreign_frac"]
    df_mon["vol_est_foreign"] = pd.to_numeric(df_mon["total_volume"], errors="coerce") * df_mon["foreign_frac"]
    df_mon["vol_est_retail"]  = pd.to_numeric(df_mon["total_volume"], errors="coerce") * df_mon["retail_frac"]
    df_mon["net_est_foreign_vol"] = (df_mon["vol_est_foreign"] - df_mon["vol_est_retail"])

    mon = (df_mon
        .set_index("trade_date")
        .resample("MS")
        .agg({
            "foreign_pct":"mean",
            "retail_pct":"mean",
            "total_volume":"sum",
            "total_value":"sum",
            "vol_est_foreign":"sum",
            "vol_est_retail":"sum",
            "net_est_foreign_vol":"sum",
        })
        .reset_index()
    )
    mon["Month"] = mon["trade_date"].dt.strftime("%Y-%m")

    sub = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Volume Estimasi: Asing vs Ritel (Bulanan)", "Rata-rata Pa (%) Bulanan"))
    sub.add_trace(go.Bar(x=mon["Month"], y=mon["vol_est_foreign"], name="Foreign (est.)",
                         marker_color="rgba(0,160,0,0.85)"), row=1, col=1)
    sub.add_trace(go.Bar(x=mon["Month"], y=mon["vol_est_retail"],  name="Retail (est.)",
                         marker_color="rgba(200,60,60,0.85)"), row=1, col=1)
    sub.update_yaxes(title_text="Volume (est.)", row=1, col=1)

    sub.add_trace(go.Scatter(x=mon["Month"], y=mon["foreign_pct"], mode="lines+markers",
                             name="Pa (%)", line=dict(width=2)), row=2, col=1)
    sub.update_yaxes(title_text="Pa (%)", range=[0,100], row=2, col=1)

    sub.update_layout(barmode="stack", height=650, hovermode="x unified",
                      showlegend=True, margin=dict(l=40,r=40,t=60,b=40))
    st.plotly_chart(sub, use_container_width=True)

    # Leaderboard pasar (per bulan)
    st.markdown("### 🏁 Leaderboard Pasar (per Bulan)")
    month_opts = mon["Month"].tolist()
    if month_opts and USE_KSEI:
        sel_mon = st.selectbox("Pilih Bulan", month_opts, index=len(month_opts)-1)
        y, m = sel_mon.split("-")
        start_d = f"{sel_mon}-01"
        end_d = f"{int(y)+1}-01-01" if int(m) == 12 else f"{y}-{int(m)+1:02d}-01"

        sql_lb = """
        SELECT
          base_symbol,
          SUM(total_volume)                                    AS total_volume,
          AVG(foreign_pct)                                     AS foreign_pct_avg,
          SUM(total_volume * (foreign_pct/100.0))              AS vol_est_foreign,
          SUM(total_volume * (1 - (foreign_pct/100.0)))        AS vol_est_retail,
          SUM(total_volume * ( (foreign_pct/100.0) - (1 - (foreign_pct/100.0)) )) AS net_est_foreign_vol
        FROM ksei_daily
        WHERE trade_date >= :start_d AND trade_date < :end_d
        GROUP BY base_symbol
        ORDER BY net_est_foreign_vol DESC
        LIMIT 30
        """
        with engine.connect() as con:
            lb_buy = pd.read_sql(text(sql_lb), con, params={"start_d": start_d, "end_d": end_d})
        with engine.connect() as con:
            lb_sell = pd.read_sql(text(sql_lb.replace("DESC","ASC")), con, params={"start_d": start_d, "end_d": end_d})

        colL, colR = st.columns(2)
        with colL:
            st.markdown("**Top Net BUY (Vol. Estimasi Asing)**")
            st.dataframe(lb_buy.assign(net_est_foreign_vol=lb_buy["net_est_foreign_vol"].round(0)),
                         use_container_width=True)
        with colR:
            st.markdown("**Top Net SELL (Vol. Estimasi Asing)**")
            st.dataframe(lb_sell.assign(net_est_foreign_vol=lb_sell["net_est_foreign_vol"].round(0)),
                         use_container_width=True)
    elif month_opts and not USE_KSEI:
        st.info("Leaderboard memerlukan tabel `ksei_daily`.")
else:
    st.info("Data KSEI tidak tersedia untuk simbol ini; ringkasan bulanan dinonaktifkan.")

# ── Ringkasan metrik & tabel
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Close", f"{df['close'].iloc[-1]:,.0f}")
c2.metric(f"Sum FF ({win}D)", f"{pd.to_numeric(df['foreign_net']).tail(win).sum():,.0f}")
c3.metric("Pa(%) terakhir", f"{pd.to_numeric(df['Pa_pct']).iloc[-1]:.2f}")
c4.metric("Ri(%) terakhir", f"{pd.to_numeric(df['Ri_pct']).iloc[-1]:.2f}")

with st.expander("Tabel (akhir 250 baris)"):
    show_cols = [
        "trade_date", "open", "high", "low", "close",
        "MA20", "foreign_net", "volume_price", "Pa_pct", "Ri_pct"
    ]
    cols_exist = [c for c in show_cols if c in df.columns]
    st.dataframe(df[cols_exist].tail(250), use_container_width=True)
