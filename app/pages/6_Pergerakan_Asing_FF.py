# -*- coding: utf-8 -*-
# app/pages/6_Pergerakan_Asing_FF.py
# Analisa Foreign Flow + KSEI bulanan (ksei_month)
# Step #1: FF Intensity + Spike markers + AVWAP
# Step #2: Heatmap kategori (bulanan) & Shift Map
# Step #3: Event study pasca spike (median & win-rate) + CAR
# Step #4: Agregasi sektor & Breadth pasar
# Patch: rangebreaks (Plotly) untuk kompres weekend/libur dan tanggal tanpa perdagangan

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
    "Termasuk Step 1–4: FF Intensity/AVWAP, Heatmap/Shift Map, Event Study, serta Agregasi Sektor & Breadth."
)

# ────────────────────────────────────────────────────────────────────────────────
# Helper: rangebreaks utk kompres weekend & tanggal libur/absent
def _rangebreaks_from_dates(dates_series):
    try:
        s = pd.to_datetime(dates_series, errors="coerce").dropna()
    except Exception:
        return []
    if s.empty:
        return []
    all_days = pd.date_range(s.min().normalize(), s.max().normalize(), freq="D")
    present = pd.to_datetime(pd.Series(s.unique())).dt.normalize()
    missing = all_days.difference(present)  # termasuk hari libur bursa
    return [dict(bounds=["sat","mon"]), dict(values=missing.to_pydatetime().tolist())]

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

            figc = go.Figure()
            for code in sorted(long["code"].unique().tolist()):
                pretty = f"{code} — {CATEGORY_LABEL.get(code, code)}"
                y = long.loc[long["code"] == code, ["Month", "volume"]]
                if detail_chart == "Bar":
                    figc.add_bar(x=y["Month"], y=y["volume"], name=pretty)
                else:
                    figc.add_scatter(x=y["Month"], y=y["volume"], mode="lines+markers", name=pretty)

            figc.update_layout(
                title=title,
                barmode="stack" if detail_chart == "Bar" else None,
                height=420, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40),
            )
            figc.update_yaxes(title_text=f"Volume ({side_label})")
            st.plotly_chart(figc, use_container_width=True)

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
            z=pivot.values,
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
            stack[c.split("_",1)[1].upper()] = sm_df[c].astype(float)

        if other:
            stack["OTH"] = sm_df[other].sum(axis=1).astype(float)
            legend_order = [c.split("_",1)[1].upper() for c in keep] + ["OTH"]
        else:
            legend_order = [c.split("_",1)[1].upper() for c in keep]

        sm_fig = go.Figure()
        for key in legend_order:
            ys = stack[key].values
            sm_fig.add_trace(
                go.Scatter(
                    x=stack["Month"], y=ys, name=key,
                    mode="lines",
                    stackgroup="one",
                    groupnorm="percent" if smode_share else None,
                    hovertemplate=(f"Bulan=%{{x}}<br>{key}=%{{y:.2f}}%<extra></extra>"
                                   if smode_share else
                                   f"Bulan=%{{x}}<br>{key}=%{{y:,.0f}}<extra></extra>"),
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

        # Top Movers MoM
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
# STEP #3 — EVENT STUDY PASCA SPIKE
st.markdown("---")
st.subheader("🧪 Event Study Pasca Spike — Step 3")

ev1, ev2, ev3 = st.columns([1.2, 1, 1])
with ev1:
    ev_side = st.selectbox("Jenis spike yang diuji", ["Spike + (FF>0)", "Spike − (FF<0)", "Keduanya"], index=0)
with ev2:
    horizons = st.multiselect("Horizon (hari)", [1, 3, 5, 10, 15, 20], default=[1, 3, 5, 10, 20])
with ev3:
    show_car = st.checkbox("Tampilkan CAR (cumulative avg. return)", value=True)

df = df.sort_values("trade_date").reset_index(drop=True)
close = pd.to_numeric(df["close"], errors="coerce")
max_h = max(horizons) if horizons else 0

fwd = {}
for h in horizons:
    fwd[h] = 100.0 * (close.shift(-h) / close - 1.0)

spikes_idx = df.index[df["is_spike"]].tolist()
pos_idx = [i for i in spikes_idx if df.loc[i, "FF_intensity"] > 0]
neg_idx = [i for i in spikes_idx if df.loc[i, "FF_intensity"] < 0]

def _event_summary(event_idx: list, label: str):
    if not event_idx:
        st.info(f"Tidak ada event untuk **{label}**.")
        return

    rows = []
    for h in horizons:
        vals = pd.Series(fwd[h]).iloc[event_idx].dropna().values
        if len(vals) == 0:
            rows.append((h, 0, np.nan, np.nan))
        else:
            med = np.nanmedian(vals)
            win = float(np.mean(vals > 0.0)) * 100.0
            rows.append((h, len(vals), med, win))

    summ = pd.DataFrame(rows, columns=["Horizon", "N", "Median (%)", "Win-rate (%)"])

    fig_ev = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ev.add_trace(go.Bar(x=summ["Horizon"], y=summ["Median (%)"], name="Median Return (%)"))
    fig_ev.add_trace(go.Scatter(x=summ["Horizon"], y=summ["Win-rate (%)"], name="Win-rate (%)", mode="lines+markers"), secondary_y=True)
    fig_ev.update_yaxes(title_text="Median Return (%)", secondary_y=False)
    fig_ev.update_yaxes(title_text="Win-rate (%)", range=[0,100], secondary_y=True)
    fig_ev.update_layout(title=f"Hasil Event Study — {label}", height=380, margin=dict(l=40,r=40,t=40,b=40), hovermode="x unified")
    st.plotly_chart(fig_ev, use_container_width=True)

    st.dataframe(summ, use_container_width=True)

    if show_car and max_h > 0:
        paths = []
        for idx0 in event_idx:
            base = close.iloc[idx0]
            if pd.isna(base):
                continue
            seq = []
            for h in range(0, max_h + 1):
                j = idx0 + h
                if j >= len(close) or pd.isna(close.iloc[j]):
                    seq.append(np.nan)
                else:
                    seq.append(100.0 * (close.iloc[j] / base - 1.0))
            paths.append(seq)
        if paths:
            arr = np.array(paths, dtype=float)
            car = np.nanmean(arr, axis=0)
            days = list(range(0, max_h + 1))
            car_fig = go.Figure()
            car_fig.add_trace(go.Scatter(x=days, y=car, mode="lines+markers", name="CAR Avg"))
            car_fig.update_layout(title=f"CAR Rata-rata — {label} (0..{max_h} hari)", height=320, margin=dict(l=40,r=40,t=40,b=40))
            car_fig.update_yaxes(title_text="CAR (%)")
            car_fig.update_xaxes(title_text="Hari sejak event")
            st.plotly_chart(car_fig, use_container_width=True)
        else:
            st.info("CAR: tidak cukup data setelah event untuk membentuk jalur.")

if ev_side == "Spike + (FF>0)":
    _event_summary(pos_idx, "Spike + (FF>0)")
elif ev_side == "Spike − (FF<0)":
    _event_summary(neg_idx, "Spike − (FF<0)")
else:
    c1, c2 = st.columns(2)
    with c1:
        _event_summary(pos_idx, "Spike + (FF>0)")
    with c2:
        _event_summary(neg_idx, "Spike − (FF<0)")

# ────────────────────────────────────────────────────────────────────────────────
# STEP #4 — AGREGASI SEKTOR & BREADTH PASAR
st.markdown("---")
st.subheader("🏭 Agregasi Sektor & 🫶 Breadth Pasar — Step 4")

u1, u2, u3 = st.columns([1.2, 1, 1])
with u1:
    uni_mode = st.selectbox("Universe", ["Semua saham (default)", "Custom list"], index=0)
with u2:
    breadth_basis = st.selectbox("Basis Breadth", ["Price return", "Foreign net"], index=0)
with u3:
    agg_mode = st.selectbox("Agregasi Sektor", ["Bulanan", "Harian"], index=0)

custom_list = []
if uni_mode == "Custom list":
    txt = st.text_input("Masukkan daftar ticker (pisah koma), contoh: BBRI, BBCA, ASII", "")
    if txt.strip():
        custom_list = [s.strip().upper() for s in txt.split(",") if s.strip()]

# Ambil data banyak simbol (periode sama)
def _date_cond(field: str) -> str:
    if period == "ALL":
        return ""
    if period.endswith("M"):
        n = int(period[:-1]); return f"AND {field} >= CURDATE() - INTERVAL {n} MONTH"
    n = int(period[:-1]); return f"AND {field} >= CURDATE() - INTERVAL {n} YEAR"

if USE_EOD_TABLE:
    sql_u = f"""
        SELECT p.trade_date, p.base_symbol, p.close, p.volume AS volume_price,
               COALESCE(f.foreign_net,0) AS foreign_net
        FROM eod p
        LEFT JOIN eod f
          ON f.trade_date = p.trade_date AND f.base_symbol = p.base_symbol AND f.is_foreign_flow = 1
        WHERE p.is_foreign_flow = 0
        {_date_cond("p.trade_date")}
    """
else:
    sql_u = f"""
        SELECT p.trade_date, p.base_symbol, p.close, p.volume_price,
               COALESCE(f.foreign_net,0) AS foreign_net
        FROM
        ( SELECT DATE(Tanggal) AS trade_date, Ticker AS base_symbol, `Close` AS close, Volume AS volume_price
          FROM eod_prices_raw
          WHERE Ticker NOT LIKE '% FF' {_date_cond("Tanggal")}
        ) p
        LEFT JOIN
        ( SELECT DATE(Tanggal) AS trade_date, TRIM(REPLACE(Ticker,' FF','')) AS base_symbol, Volume AS foreign_net
          FROM eod_prices_raw
          WHERE Ticker LIKE '% FF' {_date_cond("Tanggal")}
        ) f
        ON f.trade_date = p.trade_date AND f.base_symbol = p.base_symbol
    """

with engine.connect() as con:
    dfu = pd.read_sql(text(sql_u), con)

if dfu.empty:
    st.info("Universe kosong untuk periode ini.")
else:
    dfu["trade_date"] = pd.to_datetime(dfu["trade_date"])
    for c in ["close","volume_price","foreign_net"]:
        dfu[c] = pd.to_numeric(dfu[c], errors="coerce")
    dfu = dfu.dropna(subset=["close"])

    if custom_list:
        dfu = dfu[dfu["base_symbol"].isin(custom_list)].copy()
        if dfu.empty:
            st.warning("Ticker di custom list tidak ditemukan di periode ini.")
    # Hitung return harian per simbol
    dfu = dfu.sort_values(["base_symbol","trade_date"])
    dfu["ret"] = dfu.groupby("base_symbol")["close"].pct_change() * 100.0

    # Breadth Market
    bdf = dfu.dropna(subset=["ret"]).copy()
    if breadth_basis == "Price return":
        bdf["adv"] = (bdf["ret"] > 0).astype(int)
        bdf["dec"] = (bdf["ret"] < 0).astype(int)
        daily = bdf.groupby("trade_date")[["adv","dec"]].sum().reset_index()
        daily["unch"] = bdf.groupby("trade_date")["ret"].apply(lambda s: (s == 0).sum()).values
        daily["net_ad"] = daily["adv"] - daily["dec"]
    else:
        bdf["buy"] = (bdf["foreign_net"] > 0).astype(int)
        bdf["sell"] = (bdf["foreign_net"] < 0).astype(int)
        daily = bdf.groupby("trade_date")[["buy","sell"]].sum().reset_index()
        daily["unch"] = bdf.groupby("trade_date")["foreign_net"].apply(lambda s: (s == 0).sum()).values
        daily["net_ad"] = daily["buy"] - daily["sell"]

    # Plot breadth stacked & AD line
    fig_b = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                          row_heights=[0.6, 0.4])
    if breadth_basis == "Price return":
        fig_b.add_trace(go.Bar(x=daily["trade_date"], y=daily["adv"], name="Advancers"), row=1,col=1)
        fig_b.add_trace(go.Bar(x=daily["trade_date"], y=-daily["dec"], name="Decliners"), row=1,col=1)
        fig_b.add_trace(go.Bar(x=daily["trade_date"], y=daily["unch"], name="Unchanged"), row=1,col=1)
        fig_b.update_layout(barmode="relative")
        fig_b.update_yaxes(title_text="# Saham", row=1, col=1)
        fig_b.add_trace(go.Scatter(x=daily["trade_date"], y=daily["net_ad"].cumsum(),
                                   name="AD Line (cum Adv-Decl)",
                                   mode="lines"), row=2, col=1)
    else:
        fig_b.add_trace(go.Bar(x=daily["trade_date"], y=daily["buy"], name="# FF Buyers"), row=1,col=1)
        fig_b.add_trace(go.Bar(x=daily["trade_date"], y=-daily["sell"], name="# FF Sellers"), row=1,col=1)
        fig_b.add_trace(go.Bar(x=daily["trade_date"], y=daily["unch"], name="FF Flat"), row=1,col=1)
        fig_b.update_layout(barmode="relative")
        fig_b.update_yaxes(title_text="# Saham", row=1, col=1)
        fig_b.add_trace(go.Scatter(x=daily["trade_date"], y=daily["net_ad"].cumsum(),
                                   name="FF Breadth Line (cum Buyers-Sellers)",
                                   mode="lines"), row=2, col=1)

    fig_b.update_yaxes(title_text="Breadth (kumulatif)", row=2, col=1)
    # Rangebreaks breadth
    rb_b = _rangebreaks_from_dates(daily["trade_date"])
    fig_b.update_xaxes(rangebreaks=rb_b)

    fig_b.update_layout(height=560, margin=dict(l=40,r=40,t=40,b=40), hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_b, use_container_width=True)

    # ── Agregasi Sektor
    st.markdown("### 🏭 Agregasi Sektor")
    sector_df = None

    # 1) Coba ambil dari tabel metadata yang umum
    sector_tables = [
        ("symbols_meta", "base_symbol", "sector"),
        ("tickers_info", "base_symbol", "sector"),
        ("symbols", "symbol", "sector"),
        ("ref_symbols", "symbol", "sector"),
    ]
    for tname, sc_sym, sc_col in sector_tables:
        if _table_exists(tname):
            try:
                with engine.connect() as con:
                    sector_df = pd.read_sql(text(f"SELECT {sc_sym} AS base_symbol, {sc_col} AS sector FROM {tname}"), con)
                if not sector_df.empty:
                    break
            except Exception:
                sector_df = None

    # 2) Jika belum ada, allow user upload mapping
    if sector_df is None or sector_df.empty:
        st.info("Mapping sektor tidak ditemukan di DB. Kamu bisa upload CSV sederhana: kolom **symbol, sector**.")
        up = st.file_uploader("Upload mapping sektor (CSV)", type=["csv"])
        if up is not None:
            try:
                tmp = pd.read_csv(up)
                cols = [c.lower().strip() for c in tmp.columns]
                tmp.columns = cols
                if "symbol" in cols and "sector" in cols:
                    sector_df = tmp.rename(columns={"symbol":"base_symbol"})
                else:
                    st.error("CSV harus memiliki kolom: symbol, sector.")
            except Exception as e:
                st.error(f"Gagal membaca CSV sektor: {e}")

    if sector_df is not None and not sector_df.empty:
        sector_df["base_symbol"] = sector_df["base_symbol"].astype(str).str.upper()
        dfj = dfu.merge(sector_df, on="base_symbol", how="left")
        dfj["sector"] = dfj["sector"].fillna("UNKNOWN")

        if agg_mode == "Bulanan":
            dfj["Month"] = dfj["trade_date"].dt.to_period("M").astype(str)
            grp_keys = ["Month","sector"]
            xlabel = "Month"
        else:
            grp_keys = ["trade_date","sector"]
            xlabel = "Tanggal"

        # metrics per sector
        g = dfj.sort_values("trade_date").groupby(grp_keys)
        met = pd.DataFrame({
            "ff_net_sum": g["foreign_net"].sum(),
            "ret_median_pct": g["close"].apply(lambda s: s.pct_change().median(skipna=True) * 100.0),
            "n_symbols": g["base_symbol"].nunique(),
        }).reset_index()

        # Pivot untuk heatmap FF sector
        idx_col = grp_keys[0]
        piv = met.pivot(index=idx_col, columns="sector", values="ff_net_sum").fillna(0.0)
        # Keep deterministic sector order
        sec_order = sorted([c for c in piv.columns if c != "UNKNOWN"]) + (["UNKNOWN"] if "UNKNOWN" in piv.columns else [])
        piv = piv[sec_order]

        h2 = go.Figure(data=go.Heatmap(
            z=piv.values,
            x=piv.columns.tolist(),
            y=piv.index.tolist(),
            colorbar=dict(title="Σ Foreign Net"),
            hovertemplate=f"{xlabel}=%"+"{y}<br>Sektor=%{x}<br>Foreign Net=%{z:,.0f}<extra></extra>"
        ))
        h2.update_layout(height=420, margin=dict(l=40,r=40,t=40,b=40), title="Heatmap Σ Foreign Net per Sektor")
        st.plotly_chart(h2, use_container_width=True)

        # Bar ranking sektor periode terakhir
        last_key = piv.index.max()
        last_cut = met[met[idx_col] == last_key].copy()
        if not last_cut.empty:
            rank_fig = go.Figure()
            last_cut = last_cut.sort_values("ff_net_sum", ascending=True)
            rank_fig.add_trace(go.Bar(
                x=last_cut["ff_net_sum"],
                y=last_cut["sector"],
                orientation="h",
                name="Σ Foreign Net"
            ))
            rank_fig.update_layout(height=420, margin=dict(l=40, r=40, t=40, b=40),
                                   title=f"Ranking Sektor berdasar Σ Foreign Net — {xlabel} {last_key}")
            st.plotly_chart(rank_fig, use_container_width=True)

        # Tabel ringkas sektor
        with st.expander("📄 Tabel Agregasi Sektor"):
            st.dataframe(met.sort_values([idx_col,"ff_net_sum"], ascending=[True,False]), use_container_width=True)
    else:
        st.info("Belum ada mapping sektor, bagian ini menunggu CSV mapping atau tabel meta tersedia.")

# ────────────────────────────────────────────────────────────────────────────────
# Metrik ringkas & tabel harga (single symbol)
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Close", f"{pd.to_numeric(df['close']).iloc[-1]:,.0f}")
c2.metric(f"Sum FF ({win}D)", f"{pd.to_numeric(df['foreign_net']).tail(win).sum():,.0f}")
c3.metric("Pa(%) terakhir", f"{pd.to_numeric(df['Pa_pct']).iloc[-1]:.2f}")
c4.metric("Ri(%) terakhir", f"{pd.to_numeric(df['Ri_pct']).iloc[-1]:.2f}")

with st.expander("Tabel (akhir 250 baris)"):
    cols = [
        "trade_date", "open", "high", "low", "close", "MA20",
        "foreign_net", "volume_price", "ADV20", "FF_intensity",
        "Pa_pct", "Ri_pct", "foreign_pct", "retail_pct", "total_volume", "total_value",
    ]
    st.dataframe(df[[c for c in cols if c in df.columns]].tail(250), use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# Roadmap progress (Step checklist)
st.markdown("---")
st.subheader("🗺️ Roadmap Fitur Analitik")
st.markdown("""
- [x] **1. FF Intensity + spike markers + AVWAP**
- [x] **2. Heatmap kategori (bulanan) & Shift Map**
- [x] **3. Event study pasca spike (median & win-rate) + CAR**
- [x] **4. Agregasi sektor & Breadth pasar**
- [ ] 5. Signals harian (otomasi GitHub Actions)
""")
