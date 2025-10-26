# app/pages/10_Foreign_Flow_&_Movers.py
# Foreign Flow & Top Movers Dashboard
# - Range tanggal, filter ADVT20 & spread
# - Ranking Net Buy/Sell Asing (akumulasi)
# - Top Gainer/Loser (1D)
# - Export CSV

import io
from datetime import date, timedelta
from typing import Tuple, Optional

import pandas as pd
import streamlit as st
from db_utils import get_db_connection, get_db_name, check_secrets

# ─────────────────────────── UI SETUP ───────────────────────────
st.set_page_config(page_title="📊 Foreign Flow & Movers", page_icon="📊", layout="wide")
st.title("📊 Foreign Flow & Top Movers")

with st.expander("ℹ️ Cara pakai & definisi", expanded=False):
    st.markdown(
        """
- **Net Foreign Value** = `foreign_buy - foreign_sell` (akumulasi di rentang yang dipilih).
- **ADVT 20D** = rata-rata nilai transaksi 20 hari (proxy likuiditas).
- **Spread (bps)** = `(offer - bid) / mid * 10,000`. Semakin kecil semakin rapat.
- Pilih range tanggal → ranking **Net Buy/Sell Asing**.  
  Untuk **Top Gainer/Loser**, gunakan tanggal Akhir (default) atau pilih tanggal lain.
"""
    )

if not check_secrets(show_in_ui=True):
    st.stop()

st.caption(f"DB aktif: **{get_db_name()}**")

# ───────────────────── Helper koneksi aman ─────────────────────
def _ensure_alive(conn):
    try:
        # mysql-connector punya .reconnect(attempts, delay)
        conn.reconnect(attempts=2, delay=1)
    except Exception:
        pass

def _safe_close(conn):
    try:
        if hasattr(conn, "is_connected"):
            if conn.is_connected():
                conn.close()
        else:
            conn.close()
    except Exception:
        # swallow supaya tidak mengganggu UI
        pass

# ──────────────────────── DB PREP (views) ───────────────────────
DDL_V_DAILY_METRICS = """
CREATE OR REPLACE VIEW v_daily_metrics AS
SELECT
  trade_date,
  kode_saham,
  nama_perusahaan,
  sebelumnya       AS prev_close,
  penutupan        AS close_price,
  CASE WHEN sebelumnya IS NOT NULL AND sebelumnya <> 0
       THEN (penutupan/sebelumnya) - 1 END                 AS ret_1d,
  volume,
  nilai,
  frekuensi,
  foreign_buy,
  foreign_sell,
  (foreign_buy - foreign_sell)                              AS net_foreign_value,
  CASE WHEN nilai IS NOT NULL AND nilai <> 0
       THEN (foreign_buy - foreign_sell)/nilai END          AS net_foreign_ratio,
  bid,
  offer,
  CASE WHEN bid IS NOT NULL AND offer IS NOT NULL AND (bid+offer) <> 0
       THEN (offer - bid)/((offer + bid)/2) * 10000 END     AS spread_bps,
  non_regular_value,
  non_regular_volume,
  non_regular_frequency,
  CASE WHEN nilai IS NOT NULL AND nilai <> 0
       THEN non_regular_value / nilai END                   AS non_regular_value_pct,
  tradeable_shares,
  CASE WHEN tradeable_shares IS NOT NULL AND tradeable_shares > 0
            AND penutupan IS NOT NULL
       THEN nilai / (tradeable_shares * penutupan) END      AS turnover_est
FROM data_harian;
"""

DDL_V_ROLLING_20D = """
CREATE OR REPLACE VIEW v_rolling_20d AS
SELECT
  d.*,
  AVG(d.nilai)   OVER (PARTITION BY d.kode_saham ORDER BY d.trade_date
                       ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS advt_20,
  AVG(d.volume)  OVER (PARTITION BY d.kode_saham ORDER BY d.trade_date
                       ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS advv_20,
  STDDEV_SAMP(d.nilai)  OVER (PARTITION BY d.kode_saham ORDER BY d.trade_date
                              ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sd_val_20,
  STDDEV_SAMP(d.volume) OVER (PARTITION BY d.kode_saham ORDER BY d.trade_date
                              ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sd_vol_20,
  AVG(d.ret_1d)  OVER (PARTITION BY d.kode_saham ORDER BY d.trade_date
                       ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS avg_ret_20,
  STDDEV_SAMP(d.ret_1d) OVER (PARTITION BY d.kode_saham ORDER BY d.trade_date
                              ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sd_ret_20,
  LAG(d.close_price, 5)  OVER (PARTITION BY d.kode_saham ORDER BY d.trade_date) AS close_lag_5,
  LAG(d.close_price,20)  OVER (PARTITION BY d.kode_saham ORDER BY d.trade_date) AS close_lag_20,
  CASE WHEN LAG(d.close_price, 5)  OVER (PARTITION BY d.kode_saham ORDER BY d.trade_date) IS NOT NULL
       THEN d.close_price / LAG(d.close_price, 5)  OVER (PARTITION BY d.kode_saham ORDER BY d.trade_date) - 1
       END AS ret_5d,
  CASE WHEN LAG(d.close_price,20)  OVER (PARTITION BY d.kode_saham ORDER BY d.trade_date) IS NOT NULL
       THEN d.close_price / LAG(d.close_price,20)  OVER (PARTITION BY d.kode_saham ORDER BY d.trade_date) - 1
       END AS ret_20d
FROM v_daily_metrics d;
"""

DDL_V_ANOMALY_20D = """
CREATE OR REPLACE VIEW v_anomaly_20d AS
SELECT
  r.*,
  CASE WHEN r.sd_val_20 > 0 THEN (r.nilai   - r.advt_20)/r.sd_val_20 END AS z_val,
  CASE WHEN r.sd_vol_20 > 0 THEN (r.volume  - r.advv_20)/r.sd_vol_20 END AS z_vol,
  CASE WHEN r.sd_ret_20 > 0 THEN (r.ret_1d  - r.avg_ret_20)/r.sd_ret_20 END AS z_ret
FROM v_rolling_20d r;
"""

def ensure_views(conn):
    cur = conn.cursor()
    try:
        cur.execute(DDL_V_DAILY_METRICS)
        cur.execute(DDL_V_ROLLING_20D)
        cur.execute(DDL_V_ANOMALY_20D)
        conn.commit()
    except Exception as e:
        st.warning("Gagal membuat VIEW (mungkin hak akses terbatas). Halaman tetap jalan dengan tabel mentah.")
        st.caption(f"Detail: {e}")
    finally:
        try:
            cur.close()
        except Exception:
            pass

def get_date_bounds(conn) -> Tuple[date, date]:
    q = "SELECT MIN(trade_date), MAX(trade_date) FROM data_harian"
    cur = conn.cursor()
    cur.execute(q)
    row = cur.fetchone()
    cur.close()
    if not row or not row[0] or not row[1]:
        today = date.today()
        return today - timedelta(days=30), today
    return row[0], row[1]

# ───────────────────────── Data helpers ─────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_net_foreign_rank(start: date, end: date, min_advt: Optional[float], max_spread: Optional[float], topn: int) -> pd.DataFrame:
    conn = get_db_connection()
    try:
        _ensure_alive(conn)
        ensure_views(conn)
        sql = """
        SELECT
          d.kode_saham,
          ANY_VALUE(d.nama_perusahaan)                AS nama_perusahaan,
          SUM(d.net_foreign_value)                    AS cum_net_foreign,
          SUM(d.nilai)                                AS total_value,
          AVG(r.advt_20)                              AS avg_advt_20,
          AVG(d.spread_bps)                           AS avg_spread_bps
        FROM v_daily_metrics d
        LEFT JOIN v_rolling_20d r
               ON r.trade_date = d.trade_date AND r.kode_saham = d.kode_saham
        WHERE d.trade_date BETWEEN %s AND %s
        GROUP BY d.kode_saham
        HAVING 1=1
        """
        params = [start, end]
        if min_advt is not None and min_advt > 0:
            sql += " AND AVG(r.advt_20) >= %s"
            params.append(min_advt)
        if max_spread is not None and max_spread > 0:
            sql += " AND (AVG(d.spread_bps) <= %s OR AVG(d.spread_bps) IS NULL)"
            params.append(max_spread)
        sql += " ORDER BY cum_net_foreign DESC LIMIT %s"
        params.append(int(topn))
        df = pd.read_sql(sql, conn, params=params)
        return df
    finally:
        _safe_close(conn)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_top_movers(trade_dt: date, min_advt: Optional[float], max_spread: Optional[float], topn: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    conn = get_db_connection()
    try:
        _ensure_alive(conn)
        ensure_views(conn)
        base = """
        SELECT
          kode_saham, nama_perusahaan, ret_1d, nilai, volume,
          advt_20, spread_bps
        FROM v_rolling_20d
        WHERE trade_date = %s
        """
        params = [trade_dt]
        if min_advt is not None and min_advt > 0:
            base += " AND advt_20 >= %s"
            params.append(min_advt)
        if max_spread is not None and max_spread > 0:
            base += " AND (spread_bps <= %s OR spread_bps IS NULL)"
            params.append(max_spread)

        df_all = pd.read_sql(base, conn, params=params)
        df_gainer = df_all.sort_values(by="ret_1d", ascending=False).head(int(topn)).reset_index(drop=True)
        df_loser  = df_all.sort_values(by="ret_1d", ascending=True ).head(int(topn)).reset_index(drop=True)
        return df_gainer, df_loser
    finally:
        _safe_close(conn)

def fmt_money(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return x

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ───────────────────────────── Sidebar ─────────────────────────────
conn_for_bounds = get_db_connection()
_ensure_alive(conn_for_bounds)
ensure_views(conn_for_bounds)
min_d, max_d = get_date_bounds(conn_for_bounds)
_safe_close(conn_for_bounds)

colA, colB = st.columns([1,1])
with colA:
    start_date = st.date_input("Tanggal Mulai", value=max(min_d, max_d - timedelta(days=29)), min_value=min_d, max_value=max_d)
with colB:
    end_date   = st.date_input("Tanggal Akhir", value=max_d, min_value=min_d, max_value=max_d)

if start_date > end_date:
    st.error("Tanggal Mulai tidak boleh lebih besar dari Tanggal Akhir.")
    st.stop()

fcol1, fcol2, fcol3 = st.columns([1,1,1])
with fcol1:
    min_advt_20 = st.number_input("Min ADVT 20D (nilai rata-rata) — Rp", min_value=0, step=100_000_000, value=1_000_000_000)
with fcol2:
    max_spread_bps = st.number_input("Max Spread (bps) — opsional", min_value=0, step=5, value=50)
with fcol3:
    topn = st.number_input("Top N", min_value=5, max_value=100, step=5, value=20)

st.markdown("---")

# ───────────────────────────── Foreign Flow ─────────────────────────────
st.subheader("🟩 Top Net **Buy Asing** (Akumulasi)")
df_rank = fetch_net_foreign_rank(start_date, end_date, min_advt_20, max_spread_bps, topn*2)
df_buy  = df_rank.sort_values("cum_net_foreign", ascending=False).head(topn).reset_index(drop=True)
df_sell = df_rank.sort_values("cum_net_foreign", ascending=True ).head(topn).reset_index(drop=True)

c1, c2 = st.columns(2)
with c1:
    st.dataframe(
        df_buy.assign(
            cum_net_foreign_fmt=df_buy["cum_net_foreign"].map(fmt_money),
            total_value_fmt=df_buy["total_value"].map(fmt_money)
        )[["kode_saham","nama_perusahaan","cum_net_foreign","total_value","avg_advt_20","avg_spread_bps"]],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button("⬇️ Export CSV - Top Net Buy", data=to_csv_bytes(df_buy), file_name=f"net_buy_{start_date}_{end_date}.csv")

with c2:
    st.subheader("🟥 Top Net **Sell Asing** (Akumulasi)")
    st.dataframe(
        df_sell.assign(
            cum_net_foreign_fmt=df_sell["cum_net_foreign"].map(fmt_money),
            total_value_fmt=df_sell["total_value"].map(fmt_money)
        )[["kode_saham","nama_perusahaan","cum_net_foreign","total_value","avg_advt_20","avg_spread_bps"]],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button("⬇️ Export CSV - Top Net Sell", data=to_csv_bytes(df_sell), file_name=f"net_sell_{start_date}_{end_date}.csv")

st.markdown("---")

# ───────────────────────────── Movers 1D ─────────────────────────────
st.subheader("⚡ Top Movers (1D)")
mcol1, _ = st.columns([1,3])
with mcol1:
    movers_date = st.date_input("Tanggal untuk Movers (1D)", value=end_date, min_value=min_d, max_value=max_d, key="movers_date")

gainers, losers = fetch_top_movers(movers_date, min_advt_20, max_spread_bps, topn)

mc1, mc2 = st.columns(2)
with mc1:
    st.markdown(f"**Top Gainer — {movers_date}**")
    st.dataframe(
        gainers.assign(
            nilai_fmt=gainers["nilai"].map(fmt_money),
            volume_fmt=gainers["volume"].map(fmt_money)
        )[["kode_saham","nama_perusahaan","ret_1d","nilai","volume","advt_20","spread_bps"]],
        use_container_width=True, hide_index=True
    )
    st.download_button("⬇️ Export CSV - Top Gainer", data=to_csv_bytes(gainers), file_name=f"top_gainer_{movers_date}.csv")

with mc2:
    st.markdown(f"**Top Loser — {movers_date}**")
    st.dataframe(
        losers.assign(
            nilai_fmt=losers["nilai"].map(fmt_money),
            volume_fmt=losers["volume"].map(fmt_money)
        )[["kode_saham","nama_perusahaan","ret_1d","nilai","volume","advt_20","spread_bps"]],
        use_container_width=True, hide_index=True
    )
    st.download_button("⬇️ Export CSV - Top Loser", data=to_csv_bytes(losers), file_name=f"top_loser_{movers_date}.csv")

st.markdown("---")
st.caption("Tips: atur **Min ADVT 20D** untuk nyaring saham ilikuid; atur **Max Spread** buat fokus ke bid-ask yang rapat.")
