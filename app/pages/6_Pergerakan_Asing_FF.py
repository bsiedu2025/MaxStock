
# -*- coding: utf-8 -*-
# app/pages/6_Pergerakan_Asing_FF.py
# Analisa Foreign Flow + KSEI bulanan (ksei_month)
# Step #1: FF Intensity + Spike markers + AVWAP
# Step #2: Heatmap kategori (bulanan) & Shift Map
# NOTE: Tidak ada operasi tulis folder lokal.

import os
import tempfile
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="📈 Analisa Foreign Flow", page_icon="📈", layout="wide")
st.title("📈 Analisa Foreign Flow")
st.caption(
    "Harga + Foreign Flow + Partisipasi Asing/Ritel (KSEI bulanan `ksei_month`). "
    "Termasuk FF Intensity + AVWAP (Step 1) dan Heatmap/Shift Map kategori (Step 2)."
)

# ────────────────────────────────────────────────────────────────────────────────
# DB
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
    return create_engine(url, connect_args=connect_args, pool_recycle=300, pool_pre_ping=True)

engine = _build_engine()

def _table_exists(name: str) -> bool:
    try:
        with engine.connect() as con:
            q = text("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = :t
            """)
            return bool(con.execute(q, {"t": name}).scalar())
    except Exception:
        return False

USE_EOD_TABLE = _table_exists("eod")
USE_KSEI = _table_exists("ksei_month")

# ────────────────────────────────────────────────────────────────────────────────
# Controls
with engine.connect() as con:
    if USE_EOD_TABLE:
        syms = pd.read_sql(
            "SELECT DISTINCT base_symbol FROM eod WHERE is_foreign_flow=0 ORDER BY base_symbol", con
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
    st.warning("Belum ada data harga untuk dianalisis.")
    st.stop()

cA, cB, cC = st.columns([2, 1, 1])
with cA:
    idx = syms.index("BBRI") if "BBRI" in syms else 0
    symbol = st.selectbox("Pilih Saham", syms, index=idx)
with cB:
    period = st.selectbox("Periode", ["1M", "3M", "6M", "1Y", "ALL"], index=1)
with cC:
    price_type = st.radio("Tipe Harga", ["Line", "Candle"], horizontal=True, index=1)

win = st.slider(
    "Window Partisipasi (hari)", 5, 60, 20, 1,
    help="Jika KSEI tidak ada, Pa/Ri dihitung dari rolling |FF|/Volume."
)

# Step #1 toggles
st.markdown("### ⚙️ Opsi Analitik")
col1, col2, col3 = st.columns([1.2, 1, 1])
with col1:
    show_intensity = st.checkbox("Tampilkan FF Intensity & spike markers", value=True)
with col2:
    show_avwap = st.checkbox("Tampilkan AVWAP dari spike terbesar", value=True)
with col3:
    topN = st.number_input("Top-N spike", min_value=1, max_value=5, value=3, step=1)

def _date_filter(field: str) -> str:
    if period == "ALL":
        return ""
    if period.endswith("M"):
        n = int(period[:-1]); return f"AND {field} >= CURDATE() - INTERVAL {n} MONTH"
    n = int(period[:-1]); return f"AND {field} >= CURDATE() - INTERVAL {n} YEAR"

# ────────────────────────────────────────────────────────────────────────────────
# Query harga + FF + join KSEI from ksei_month (by YEAR-MONTH)
ksei_join = """
LEFT JOIN (
    SELECT
      base_symbol,
      trade_date,
      (COALESCE(local_total,0) + COALESCE(foreign_total,0)) AS total_volume,
      CASE WHEN COALESCE(local_total,0) + COALESCE(foreign_total,0) > 0
           THEN 100.0 * COALESCE(foreign_total,0) / (COALESCE(local_total,0) + COALESCE(foreign_total,0))
           ELSE NULL END AS foreign_pct,
      CASE WHEN COALESCE(local_total,0) + COALESCE(foreign_total,0) > 0
           THEN 100.0 - (100.0 * COALESCE(foreign_total,0) / (COALESCE(local_total,0) + COALESCE(foreign_total,0)))
           ELSE NULL END AS retail_pct,
      CASE WHEN price IS NOT NULL
           THEN price * (COALESCE(local_total,0) + COALESCE(foreign_total,0))
           ELSE NULL END AS total_value
    FROM ksei_month
) k
  ON k.base_symbol = p.base_symbol
 AND YEAR(k.trade_date) = YEAR(p.trade_date)
 AND MONTH(k.trade_date) = MONTH(p.trade_date)
"""

if USE_EOD_TABLE:
    sql_df = f"""
    SELECT
      p.trade_date, p.base_symbol,
      p.open, p.high, p.low, p.close,
      p.volume AS volume_price,
      COALESCE(f.foreign_net, 0) AS foreign_net,
      k.foreign_pct, k.retail_pct, k.total_volume, k.total_value
    FROM eod p
    LEFT JOIN eod f
      ON f.trade_date = p.trade_date AND f.base_symbol = p.base_symbol AND f.is_foreign_flow = 1
    {ksei_join if USE_KSEI else ""}
    WHERE p.base_symbol = :sym AND p.is_foreign_flow = 0
    {_date_filter("p.trade_date")}
    ORDER BY p.trade_date
    """
else:
    sql_df = f"""
    SELECT
        p.trade_date, p.base_symbol,
        p.open, p.high, p.low, p.close,
        p.volume_price,
        COALESCE(f.foreign_net, 0) AS foreign_net,
        k.foreign_pct, k.retail_pct, k.total_volume, k.total_value
    FROM
      (SELECT DATE(Tanggal) AS trade_date, Ticker AS base_symbol,
              `Open` AS open, `High` AS high, `Low` AS low, `Close` AS close,
              Volume AS volume_price
       FROM eod_prices_raw
       WHERE Ticker = :sym AND Ticker NOT LIKE '% FF' {_date_filter("Tanggal")}
      ) AS p
    LEFT JOIN
      (SELECT DATE(Tanggal) AS trade_date, TRIM(REPLACE(Ticker,' FF','')) AS base_symbol,
              Volume AS foreign_net
       FROM eod_prices_raw
       WHERE TRIM(REPLACE(Ticker,' FF','')) = :sym AND Ticker LIKE '% FF' {_date_filter("Tanggal")}
      ) AS f
      ON f.trade_date = p.trade_date AND f.base_symbol = p.base_symbol
    {ksei_join if USE_KSEI else ""}
    ORDER BY p.trade_date
    """

with engine.connect() as con:
    df = pd.read_sql(text(sql_df), con, params={"sym": symbol})

if df.empty:
    st.warning("Data tidak tersedia untuk simbol/periode ini.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# Feature engineering
df["trade_date"] = pd.to_datetime(df["trade_date"])
for col in ["open","high","low","close","volume_price","foreign_net"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["MA20"] = df["close"].rolling(20, min_periods=1).mean()

# Pa/Ri by KSEI (or fallback rolling)
if USE_KSEI and df["foreign_pct"].notna().any():
    df["Pa_pct"] = pd.to_numeric(df["foreign_pct"], errors="coerce").clip(0, 100)
    df["Ri_pct"] = pd.to_numeric(
        df["retail_pct"].where(df["retail_pct"].notna(), 100 - df["Pa_pct"]),
        errors="coerce",
    ).clip(0, 100)
else:
    vol_roll = df["volume_price"].abs().rolling(win, min_periods=1).sum()
    ff_roll  = df["foreign_net"].abs().rolling(win, min_periods=1).sum()
    df["Pa_pct"] = (100.0 * (ff_roll / vol_roll)).clip(0, 100).fillna(0.0)
    df["Ri_pct"] = (100.0 - df["Pa_pct"]).clip(0, 100)

# ── Step #1: FF Intensity & spikes
df["ADV20"] = df["volume_price"].rolling(20, min_periods=5).mean()
df["FF_intensity"] = df["foreign_net"] / df["ADV20"]
thr = np.nan
if show_intensity and df["FF_intensity"].notna().any():
    thr = np.nanpercentile(np.abs(df["FF_intensity"].dropna()), 95)
    df["is_spike"] = np.abs(df["FF_intensity"]) >= thr
else:
    df["is_spike"] = False

# Top-N spikes (anchor candidates) by |FF_intensity|
spike_df = df.loc[df["is_spike"]].copy()
spike_df["abs_ffi"] = np.abs(spike_df["FF_intensity"])
spike_df = spike_df.sort_values(["abs_ffi","trade_date"], ascending=[False, True]).head(int(topN))
anchor_indices = spike_df.index.tolist()

# ────────────────────────────────────────────────────────────────────────────────
# Plots utama (Harga + FF + Pa/Ri)
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    row_heights=[0.58, 0.27, 0.15],
    specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]],
    subplot_titles=(f"Harga — {symbol}", "Daily Foreign Flow (Net Volume) + FF Intensity", "Partisipasi Asing & Ritel (%)"),
)

# Harga
if price_type == "Line":
    fig.add_trace(go.Scatter(x=df["trade_date"], y=df["close"], name="Harga", mode="lines"), row=1, col=1)
else:
    fig.add_trace(
        go.Candlestick(
            x=df["trade_date"],
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Harga",
        ),
        row=1, col=1,
    )
fig.add_trace(go.Scatter(x=df["trade_date"], y=df["MA20"], name="MA20", mode="lines"), row=1, col=1)

# FF bar (primary y)
colors = np.where(df["foreign_net"] > 0, "rgba(0,160,0,0.9)",
         np.where(df["foreign_net"] < 0, "rgba(220,0,0,0.9)", "rgba(160,160,160,0.6)"))
fig.add_trace(
    go.Bar(x=df["trade_date"], y=df["foreign_net"], name="Foreign Net",
           marker=dict(color=colors), marker_line_width=0, opacity=0.95),
    row=2, col=1, secondary_y=False,
)

# FF Intensity line + spike markers (secondary y)
if show_intensity:
    fig.add_trace(
        go.Scatter(x=df["trade_date"], y=df["FF_intensity"], name="FF Intensity (FF / ADV20)",
                   mode="lines", opacity=0.9),
        row=2, col=1, secondary_y=True,
    )
    if df["is_spike"].any():
        spike_pos = df["is_spike"] & (df["FF_intensity"] > 0)
        spike_neg = df["is_spike"] & (df["FF_intensity"] < 0)
        if spike_pos.any():
            fig.add_trace(
                go.Scatter(x=df.loc[spike_pos, "trade_date"], y=df.loc[spike_pos, "FF_intensity"],
                           mode="markers", name="Spike +",
                           marker=dict(symbol="triangle-up", size=10)),
                row=2, col=1, secondary_y=True,
            )
        if spike_neg.any():
            fig.add_trace(
                go.Scatter(x=df.loc[spike_neg, "trade_date"], y=df.loc[spike_neg, "FF_intensity"],
                           mode="markers", name="Spike −",
                           marker=dict(symbol="triangle-down", size=10)),
                row=2, col=1, secondary_y=True,
            )
        if not np.isnan(thr):
            fig.add_hline(y=thr, line_dash="dot", line_width=1, row=2, col=1, secondary_y=True)
            fig.add_hline(y=-thr, line_dash="dot", line_width=1, row=2, col=1, secondary_y=True)

# AVWAP dari spike terbesar (overlay di harga)
def _anchored_vwap(close: pd.Series, vol: pd.Series, anchor_idx: int) -> pd.Series:
    pv = (close * vol).fillna(0).cumsum()
    cv = vol.fillna(0).cumsum()
    out = pd.Series(np.nan, index=close.index)
    for j in range(anchor_idx, len(close)):
        num = pv.iloc[j] - (pv.iloc[anchor_idx - 1] if anchor_idx > 0 else 0.0)
        den = cv.iloc[j] - (cv.iloc[anchor_idx - 1] if anchor_idx > 0 else 0.0)
        out.iloc[j] = num / den if den != 0 else np.nan
    return out

if show_avwap and len(anchor_indices) > 0:
    for i, idx0 in enumerate(anchor_indices, start=1):
        avwap = _anchored_vwap(df["close"], df["volume_price"], idx0)
        label = f"AVWAP spike#{i} ({pd.to_datetime(df.loc[idx0,'trade_date']).date()})"
        fig.add_trace(go.Scatter(x=df["trade_date"], y=avwap, name=label, mode="lines"), row=1, col=1)
        fig.add_vline(x=df.loc[idx0, "trade_date"], line_width=1, line_dash="dot", row=1, col=1)

# Pa/Ri
fig.add_trace(go.Scatter(x=df["trade_date"], y=df["Pa_pct"], name="Pa (%)", mode="lines"), row=3, col=1)
fig.add_trace(go.Scatter(x=df["trade_date"], y=df["Ri_pct"], name="Ri (%)", mode="lines"), row=3, col=1)
fig.add_hline(y=50, line_dash="dot", line_width=1, row=3, col=1)

fig.update_layout(
    height=900, margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_rangeslider_visible=False, hovermode="x unified",
)
fig.update_yaxes(title_text="Harga", row=1, col=1)
fig.update_yaxes(title_text="Foreign Net", row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text="FF Intensity", row=2, col=1, secondary_y=True)
fig.update_yaxes(title_text="%", range=[0, 100], row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# Ringkasan Spike
with st.expander("🔎 Ringkasan Spike FF Intensity (Top-N)"):
    spike_df2 = df.loc[df["is_spike"]].copy()
    if spike_df2.empty:
        st.info("Tidak ada spike pada periode ini.")
    else:
        show_cols = ["trade_date","foreign_net","ADV20","FF_intensity"]
        tmp = spike_df2[show_cols].copy()
        tmp = tmp.sort_values("FF_intensity", key=lambda s: np.abs(s), ascending=False)
        tmp["trade_date"] = pd.to_datetime(tmp["trade_date"]).dt.date.astype(str)
        st.dataframe(tmp, use_container_width=True)
        if not np.isnan(thr):
            st.caption(f"Ambang spike (|FF_intensity| p95): **{thr:.2f}**")

# ────────────────────────────────────────────────────────────────────────────────
# Ringkasan Bulanan KSEI (dari ksei_month)
st.markdown("---")
st.subheader("📅 Ringkasan Bulanan KSEI")

if USE_KSEI:
    show_all_ksei = st.checkbox("Tampilkan semua data KSEI (abaikan filter Periode)", value=True)
    chart_type = st.radio("Tipe grafik bulanan", ["Line", "Bar"], index=0, horizontal=True)

    params = {"sym": symbol}
    if show_all_ksei:
        cond = ""
    else:
        if period == "ALL":
            cond = ""
        elif period.endswith("M"):
            n = int(period[:-1]); cond = "AND trade_date >= DATE_SUB(CURDATE(), INTERVAL :n MONTH)"; params["n"] = n
        else:
            n = int(period[:-1]); cond = "AND trade_date >= DATE_SUB(CURDATE(), INTERVAL :n YEAR)"; params["n"] = n

    sql_k = f"""
        SELECT
          trade_date, base_symbol,
          (COALESCE(local_total,0) + COALESCE(foreign_total,0)) AS total_volume,
          CASE WHEN COALESCE(local_total,0) + COALESCE(foreign_total,0) > 0
               THEN 100.0 * COALESCE(foreign_total,0) / (COALESCE(local_total,0) + COALESCE(foreign_total,0))
               ELSE NULL END AS foreign_pct,
          CASE WHEN COALESCE(local_total,0) + COALESCE(foreign_total,0) > 0
               THEN 100.0 - (100.0 * COALESCE(foreign_total,0) / (COALESCE(local_total,0) + COALESCE(foreign_total,0)))
               ELSE NULL END AS retail_pct,
          CASE WHEN price IS NOT NULL
               THEN price * (COALESCE(local_total,0) + COALESCE(foreign_total,0))
               ELSE NULL END AS total_value
        FROM ksei_month
        WHERE base_symbol = :sym
        {cond}
        ORDER BY trade_date
    """
    with engine.connect() as con:
        kdf = pd.read_sql(text(sql_k), con, params=params)

    if not kdf.empty:
        kdf["trade_date"] = pd.to_datetime(kdf["trade_date"])
        kdf["Month"] = kdf["trade_date"].dt.strftime("%Y-%m")

        agg = (kdf.sort_values("trade_date")
                 .groupby("Month", as_index=False)
                 .agg({"total_volume": "sum", "foreign_pct": "mean"}))

        tmp = kdf.copy()
        tmp["foreign_frac"] = pd.to_numeric(tmp["foreign_pct"], errors="coerce") / 100.0
        tmp["vol_foreign_est"] = pd.to_numeric(tmp["total_volume"], errors="coerce") * tmp["foreign_frac"]
        tmp["vol_local_est"]   = pd.to_numeric(tmp["total_volume"], errors="coerce") * (1.0 - tmp["foreign_frac"])
        vol = (tmp.groupby("Month", as_index=False)
                  .agg({"vol_foreign_est": "sum", "vol_local_est": "sum"}))

        sub = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=("Volume Estimasi: Asing vs Lokal (Bulanan)", "Pa (%) Bulanan"),
        )

        if chart_type == "Line":
            sub.add_trace(go.Scatter(x=vol["Month"], y=vol["vol_foreign_est"],
                                     name="Foreign (est.)", mode="lines+markers"), row=1, col=1)
            sub.add_trace(go.Scatter(x=vol["Month"], y=vol["vol_local_est"],
                                     name="Lokal/Ritel (est.)", mode="lines+markers"), row=1, col=1)
        else:
            sub.add_trace(go.Bar(x=vol["Month"], y=vol["vol_foreign_est"], name="Foreign (est.)"), row=1, col=1)
            sub.add_trace(go.Bar(x=vol["Month"], y=vol["vol_local_est"],  name="Lokal/Ritel (est.)"), row=1, col=1)
            sub.update_layout(barmode="stack")

        sub.update_yaxes(title_text="Volume (est.)", row=1, col=1)
        sub.add_trace(go.Scatter(x=agg["Month"], y=agg["foreign_pct"],
                                 name="Pa (%)", mode="lines+markers"), row=2, col=1)
        sub.update_yaxes(title_text="Pa (%)", range=[0, 100], row=2, col=1)

        sub.update_layout(
            height=650, hovermode="x unified",
            showlegend=True, margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(sub, use_container_width=True)
    else:
        st.info("Belum ada baris `ksei_month` untuk simbol ini.")
else:
    st.info("Tabel `ksei_month` belum tersedia.")

# ────────────────────────────────────────────────────────────────────────────────
# 📊 DETAIL KATEGORI (paling bawah) — sumber: ksei_month
st.markdown("---")
st.subheader("📊 Detail Kategori KSEI (Semua Bulan)")

CATEGORY_LABEL = {
    "ID": "Individual (Perorangan)",
    "CP": "Corporate (Perusahaan/Corporate)",
    "MF": "Mutual Fund (Reksa Dana)",
    "IB": "Financial Institution (Lembaga Keuangan Lainnya)",
    "IS": "Insurance (Perusahaan Perasuransian)",
    "SC": "Securities Company (Perusahaan Efek/Sekuritas)",
    "PF": "Pension Fund (Dana Pensiun)",
    "FD": "Foundation (Yayasan)",
    "OT": "Others (Lainnya)",
}

if USE_KSEI:
    detail_side = st.radio("Sisi", ["Lokal", "Asing", "Keduanya"], index=2, horizontal=True)
    detail_chart = st.radio("Tipe grafik detail", ["Bar", "Line"], index=0, horizontal=True)
    use_all_detail = st.checkbox("Tampilkan semua bulan (abaikan filter Periode)", value=True)

    with st.expander("ℹ️ Keterangan Kategori (sumber: Panduan KSEI)"):
        k_rows = [f"- **{code}** — {CATEGORY_LABEL[code]}" for code in ["ID","CP","MF","IB","IS","SC","PF","FD","OT"]]
        st.markdown("\n".join(k_rows))

    params_d = {"sym": symbol}
    if use_all_detail:
        cond_d = ""
    else:
        if period == "ALL":
            cond_d = ""
        elif period.endswith("M"):
            n = int(period[:-1]); cond_d = "AND trade_date >= DATE_SUB(CURDATE(), INTERVAL :n MONTH)"; params_d["n"] = n
        else:
            n = int(period[:-1]); cond_d = "AND trade_date >= DATE_SUB(CURDATE(), INTERVAL :n YEAR)"; params_d["n"] = n

    sql_det = f"""
        SELECT
          trade_date, base_symbol,
          local_is, local_cp, local_pf, local_ib, local_id, local_mf, local_sc, local_fd, local_ot, local_total,
          foreign_is, foreign_cp, foreign_pf, foreign_ib, foreign_id, foreign_mf, foreign_sc, foreign_fd, foreign_ot, foreign_total
        FROM ksei_month
        WHERE base_symbol = :sym
        {cond_d}
        ORDER BY trade_date
    """
    with engine.connect() as con:
        kcat = pd.read_sql(text(sql_det), con, params=params_d)

    if kcat.empty:
        st.info("Belum ada data kategori di `ksei_month` untuk simbol ini.")
    else:
        kcat["trade_date"] = pd.to_datetime(kcat["trade_date"])
        kcat["Month"] = kcat["trade_date"].dt.strftime("%Y-%m")

        num_cols = [
            "local_is","local_cp","local_pf","local_ib","local_id","local_mf","local_sc","local_fd","local_ot","local_total",
            "foreign_is","foreign_cp","foreign_pf","foreign_ib","foreign_id","foreign_mf","foreign_sc","foreign_fd","foreign_ot","foreign_total"
        ]
        kcat[num_cols] = kcat[num_cols].apply(pd.to_numeric, errors="coerce")
        agg_cat = kcat.groupby("Month", as_index=False)[num_cols].sum()

        # Plot helper
        def _plot_categories(df_month: pd.DataFrame, cols: list, title: str, side_label: str):
            long = df_month.melt(id_vars="Month", value_vars=cols, var_name="category", value_name="volume").fillna(0.0)
            long["code"] = (long["category"]
                            .str.replace(r"^local_", "", regex=True)
                            .str.replace(r"^foreign_", "", regex=True)
                            .str.upper())

            fig = go.Figure()
            for code in sorted(long["code"].unique().tolist()):
                pretty = f"{code} — {CATEGORY_LABEL.get(code, code)}"
                y = long.loc[long["code"] == code, ["Month", "volume"]]
                if detail_chart == "Bar":
                    fig.add_bar(x=y["Month"], y=y["volume"], name=pretty)
                else:
                    fig.add_scatter(x=y["Month"], y=y["volume"], mode="lines+markers", name=pretty)

            fig.update_layout(
                title=title,
                barmode="stack" if detail_chart == "Bar" else None,
                height=420, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40),
            )
            fig.update_yaxes(title_text=f"Volume ({side_label})")
            st.plotly_chart(fig, use_container_width=True)

        # Lokal
        if detail_side in ("Lokal", "Keduanya"):
            _plot_categories(
                agg_cat,
                ["local_is","local_cp","local_pf","local_ib","local_id","local_mf","local_sc","local_fd","local_ot"],
                "Detail Kategori — **Lokal** (IS, CP, PF, IB, ID, MF, SC, FD, OT)",
                "Lokal",
            )
        # Asing
        if detail_side in ("Asing", "Keduanya"):
            _plot_categories(
                agg_cat,
                ["foreign_is","foreign_cp","foreign_pf","foreign_ib","foreign_id","foreign_mf","foreign_sc","foreign_fd","foreign_ot"],
                "Detail Kategori — **Asing** (IS, CP, PF, IB, ID, MF, SC, FD, OT)",
                "Asing",
            )

        with st.expander("📄 Tabel Ringkas Kategori (per Bulan)"):
            show_cols = ["Month",
                         "local_is","local_cp","local_pf","local_ib","local_id","local_mf","local_sc","local_fd","local_ot","local_total",
                         "foreign_is","foreign_cp","foreign_pf","foreign_ib","foreign_id","foreign_mf","foreign_sc","foreign_fd","foreign_ot","foreign_total"]
            st.dataframe(agg_cat[show_cols], use_container_width=True)

        # ───────────────────────────────────────────────────────────────────
        # STEP #2 — HEATMAP & SHIFT MAP
        st.markdown("---")
        st.subheader("🔥 Heatmap & Shift Map Kategori (Bulanan) — Step 2")

        hcol1, hcol2, hcol3 = st.columns([1.2, 1, 1])
        with hcol1:
            hm_side = st.selectbox("Sisi data untuk Heatmap/Shift Map", ["Lokal", "Asing"], index=0)
        with hcol2:
            hm_mode = st.selectbox("Normalisasi", ["% kontribusi (share)", "Volume absolut"], index=0)
        with hcol3:
            topK = st.number_input("Top-K kategori (Shift Map)", min_value=3, max_value=9, value=5, step=1,
                                   help="Kategori di luar Top-K akan digabung sebagai OTH.")

        if hm_side == "Lokal":
            cat_cols = ["local_is","local_cp","local_pf","local_ib","local_id","local_mf","local_sc","local_fd","local_ot"]
            total_col = "local_total"
        else:
            cat_cols = ["foreign_is","foreign_cp","foreign_pf","foreign_ib","foreign_id","foreign_mf","foreign_sc","foreign_fd","foreign_ot"]
            total_col = "foreign_total"

        wide = agg_cat[["Month"] + cat_cols + [total_col]].copy()
        wide = wide.sort_values("Month").reset_index(drop=True)

        # Shares
        share = wide.copy()
        denom = share[total_col].replace(0, np.nan)
        for c in cat_cols:
            share[c] = share[c] / denom
        share = share.drop(columns=[total_col])

        # Data untuk heatmap
        def _category_code(colname: str) -> str:
            return colname.split("_", 1)[1].upper()

        if hm_mode.startswith("%"):
            zdf = share
            ztitle = "Share (%)"
            zmin, zmax = 0.0, 1.0
        else:
            zdf = wide.drop(columns=[total_col])
            ztitle = "Volume"
            zmin, zmax = None, None

        # tidy → pivot untuk heatmap
        zlong = zdf.melt(id_vars="Month", value_vars=cat_cols, var_name="cat", value_name="val")
        zlong["code"] = zlong["cat"].map(_category_code)
        pivot = zlong.pivot(index="Month", columns="code", values="val").fillna(0.0)
        pivot = pivot[[c for c in ["IS","CP","PF","IB","ID","MF","SC","FD","OT"] if c in pivot.columns]]

        hm = go.Figure(data=go.Heatmap(
            z=pivot.values if hm_mode.startswith("%") else pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorbar=dict(title=ztitle),
            zmin=zmin, zmax=zmax,
            hovertemplate="Bulan=%{y}<br>Kategori=%{x}<br>Nilai=%{z}<extra></extra>"
        ))
        hm.update_layout(height=420, margin=dict(l=40, r=40, t=40, b=40), title="Heatmap Kategori per Bulan")
        st.plotly_chart(hm, use_container_width=True)

        # Shift Map (stacked area)
        smode_share = hm_mode.startswith("%")
        sm_df = (share if smode_share else wide.drop(columns=[total_col])).copy()

        # Pilih Top-K kategori berdasar rata-rata share/volume
        means = sm_df[cat_cols].mean(axis=0).sort_values(ascending=False)
        keep = means.head(int(topK)).index.tolist()
        other = [c for c in cat_cols if c not in keep]

        stack = pd.DataFrame({"Month": sm_df["Month"]})
        for c in keep:
            stack[_category_code(c)] = sm_df[c].astype(float)

        if other:
            stack["OTH"] = sm_df[other].sum(axis=1).astype(float)
            legend_order = [_category_code(c) for c in keep] + ["OTH"]
        else:
            legend_order = [_category_code(c) for c in keep]

        # Plot stacked area
        sm_fig = go.Figure()
        for i, key in enumerate(legend_order):
            ys = stack[key].values
            sm_fig.add_trace(
                go.Scatter(
                    x=stack["Month"], y=ys, name=key,
                    mode="lines",
                    stackgroup="one",
                    groupnorm="percent" if smode_share else None,
                    hovertemplate="Bulan=%{x}<br>%s=%{y:.2f}%s<extra></extra>" % (
                        key, "%" if smode_share else "", "" if smode_share else ""
                    ),
                )
            )
        sm_fig.update_layout(
            title="Shift Map Kategori (Top-K + OTH)",
            height=420, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40),
        )
        sm_fig.update_yaxes(title_text="Share (%)" if smode_share else "Volume", range=[0, 100] if smode_share else None)
        st.plotly_chart(sm_fig, use_container_width=True)

        # Top Movers MoM (berdasar share atau volume)
        st.markdown("**Top Movers MoM (berdasar pilihan normalisasi di atas)**")
        if len(stack) >= 2:
            last, prev = stack.iloc[-1], stack.iloc[-2]
            movers = []
            for key in legend_order:
                d = (last[key] - prev[key])
                movers.append((key, d))
            movers = sorted(movers, key=lambda x: abs(x[1]), reverse=True)
            tbl = pd.DataFrame(movers, columns=["Kategori", "Δ (MoM)"])
            if smode_share:
                tbl["Δ (MoM)"] = tbl["Δ (MoM)"] * 100.0
                tbl["Δ (MoM)"] = tbl["Δ (MoM)"].map(lambda v: f"{v:+.2f}%")
            else:
                tbl["Δ (MoM)"] = tbl["Δ (MoM)"].map(lambda v: f"{v:+,.0f}")
            st.dataframe(tbl, use_container_width=True)
        else:
            st.info("Butuh ≥ 2 bulan data untuk menghitung MoM.")

else:
    st.info("Tabel `ksei_month` belum tersedia untuk detail kategori.")

# ────────────────────────────────────────────────────────────────────────────────
# Roadmap progress (Step checklist)
st.markdown("---")
st.subheader("🗺️ Roadmap Fitur Analitik")
st.markdown("""
- [x] **1. FF Intensity + spike markers + AVWAP**
- [x] **2. Heatmap kategori (bulanan) & Shift Map**
- [ ] 3. Event study pasca spike (median & win-rate)
- [ ] 4. Agregasi sektor & Breadth pasar
- [ ] 5. Signals harian (otomasi GitHub Actions)
""")
