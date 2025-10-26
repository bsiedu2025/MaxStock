# app/pages/9_Import_Market_Data.py
# Import Ringkasan Saham Harian (CSV/XLSX) → data_harian
# - Support single & bulk upload
# - Dedup by (kode_saham, trade_date)
# - UPSERT (optional) or INSERT IGNORE

import io
import os
import math
from typing import List

import pandas as pd
import streamlit as st
from db_utils import get_db_connection, get_db_name, check_secrets

st.set_page_config(page_title="📥 Import Market Data Harian", page_icon="📥", layout="wide")
st.title("📥 Import Market Data Harian (CSV/XLSX) → `data_harian`")

with st.expander("ℹ️ Petunjuk & Catatan", expanded=False):
    st.markdown(
        "- Format kolom mengikuti file contoh **Ringkasan Saham-YYYYMMDD.csv/xlsx** (BEI).\n"
        "- Kunci unik: **(kode_saham, trade_date)** → tidak akan dobel saat upload berulang.\n"
        "- Bisa unggah **satu** atau **banyak file** sekaligus.\n"
        "- Pilih mode **Lewati Duplikat (INSERT IGNORE)** atau **Update Jika Ada (UPSERT)**."
    )

if not check_secrets(show_in_ui=True):
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────────
def ensure_table_data_harian(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS data_harian (
        trade_date           DATE         NOT NULL,
        kode_saham           VARCHAR(20)  NOT NULL,
        nama_perusahaan      VARCHAR(255) NULL,
        remarks              VARCHAR(255) NULL,

        sebelumnya           DECIMAL(19,4) NULL,
        open_price           DECIMAL(19,4) NULL,
        first_trade          DECIMAL(19,4) NULL,
        tertinggi            DECIMAL(19,4) NULL,
        terendah             DECIMAL(19,4) NULL,
        penutupan            DECIMAL(19,4) NULL,
        selisih              DECIMAL(19,4) NULL,

        volume               BIGINT       NULL,
        nilai                BIGINT       NULL,
        frekuensi            BIGINT       NULL,
        index_individual     DECIMAL(19,4) NULL,
        offer                DECIMAL(19,4) NULL,
        offer_volume         BIGINT       NULL,
        bid                  DECIMAL(19,4) NULL,
        bid_volume           BIGINT       NULL,
        listed_shares        BIGINT       NULL,
        tradeable_shares     BIGINT       NULL,
        weight_for_index     BIGINT       NULL,
        foreign_sell         BIGINT       NULL,
        foreign_buy          BIGINT       NULL,
        non_regular_volume   BIGINT       NULL,
        non_regular_value    BIGINT       NULL,
        non_regular_frequency BIGINT      NULL,

        source_file          VARCHAR(255) NULL,
        created_at           TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,

        PRIMARY KEY (kode_saham, trade_date),
        KEY idx_trade_date (trade_date),
        KEY idx_kode_saham (kode_saham)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """)
    cur.close()


# ──────────────────────────────────────────────────────────────────────────
# File parsing & normalization
# ──────────────────────────────────────────────────────────────────────────
ID_MONTH = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','Mei':'05','Jun':'06','Jul':'07','Agt':'08','Sep':'09','Okt':'10','Nov':'11','Des':'12'}

def parse_id_date(s: str):
    if s is None:
        return None
    s = str(s).strip()
    parts = s.split()
    if len(parts) == 3 and parts[1] in ID_MONTH:
        d = parts[0].zfill(2); m = ID_MONTH[parts[1]]; y = parts[2]
        return f"{y}-{m}-{d}"
    # fallback
    try:
        return pd.to_datetime(s, dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return None

def _read_csv_robust(uploaded_file) -> pd.DataFrame:
    # Baca isi bytes → coba beberapa kombinasi encoding & delimiter
    raw = uploaded_file.read()
    # reset pointer agar bisa dibaca ulang di tempat lain jika perlu
    def try_read(encoding: str, sep: str):
        try:
            text = raw.decode(encoding, errors='ignore')
            return pd.read_csv(io.StringIO(text), sep=sep)
        except Exception:
            return None
    # urutan percobaan
    for enc in ("utf-8", "utf-16", "latin1"):
        for sep in (",", ";", "\t"):
            df = try_read(enc, sep)
            if df is not None and len(df.columns) >= 3:
                return df
    # terakhir: coba pandas langsung tanpa decode manual
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)

def read_any(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(('.xlsx', '.xls')):
        try:
            import openpyxl  # wajib untuk xlsx
        except Exception:
            st.error(
                "File Excel terdeteksi (**.xlsx/.xls**), tetapi dependensi **openpyxl** belum terpasang.\n\n"
                "Tambahkan `openpyxl>=3.1.2` ke **requirements.txt**, lalu deploy ulang. "
                "Sementara waktu, kamu bisa upload **CSV** sebagai alternatif."
            )
            st.stop()
        # pastikan pointer di awal
        uploaded_file.seek(0)
        # force engine agar jelas
        return pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        uploaded_file.seek(0)
        return _read_csv_robust(uploaded_file)

def normalize_market_df(df: pd.DataFrame, source_file: str = None) -> pd.DataFrame:
    # mapping default sesuai contoh BEI
    get = lambda c: df[c] if c in df.columns else None

    out = pd.DataFrame()
    out['trade_date'] = (get('Tanggal Perdagangan Terakhir') or pd.Series([None]*len(df))).map(parse_id_date)
    out['kode_saham'] = (get('Kode Saham') or pd.Series([None]*len(df))).astype(str).str.upper().str.strip()
    out['nama_perusahaan'] = (get('Nama Perusahaan') or pd.Series([None]*len(df))).astype(str).str.strip()
    out['remarks'] = (get('Remarks') or pd.Series([None]*len(df)))

    def num(col):
        s = get(col)
        return pd.to_numeric(s, errors='coerce') if s is not None else None

    out['sebelumnya'] = num('Sebelumnya')
    out['open_price'] = num('Open Price')
    out['first_trade'] = num('First Trade')
    out['tertinggi'] = num('Tertinggi')
    out['terendah'] = num('Terendah')
    out['penutupan'] = num('Penutupan')
    out['selisih'] = num('Selisih')
    out['volume'] = num('Volume')
    out['nilai'] = num('Nilai')
    out['frekuensi'] = num('Frekuensi')
    out['index_individual'] = num('Index Individual')
    out['offer'] = num('Offer')
    out['offer_volume'] = num('Offer Volume')
    out['bid'] = num('Bid')
    out['bid_volume'] = num('Bid Volume')
    out['listed_shares'] = num('Listed Shares')
    out['tradeable_shares'] = num('Tradeble Shares') if 'Tradeble Shares' in df.columns else num('Tradeable Shares')
    out['weight_for_index'] = num('Weight For Index')
    out['foreign_sell'] = num('Foreign Sell')
    out['foreign_buy'] = num('Foreign Buy')
    out['non_regular_volume'] = num('Non Regular Volume')
    out['non_regular_value'] = num('Non Regular Value')
    out['non_regular_frequency'] = num('Non Regular Frequency')
    out['source_file'] = source_file

    # bersih & dedup
    out = out.dropna(subset=['trade_date','kode_saham'])
    out = out.drop_duplicates(subset=['kode_saham','trade_date'], keep='last')
    return out

# ──────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────
st.caption(f"DB aktif: **{get_db_name()}**")

files = st.file_uploader(
    "Pilih file harian (CSV/XLSX). Anda bisa pilih **satu** atau **banyak** file sekaligus.",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    key="uploader_market_daily",
)
preview_rows = st.number_input("Preview baris", 5, 50, 20, 5)

mode = st.radio(
    "Mode insert ke database:",
    options=["Lewati duplikat (INSERT IGNORE)", "Update jika sudah ada (UPSERT)"],
    index=0
)

parsed: List[pd.DataFrame] = []
if files:
    st.write(f"Total file terpilih: **{len(files)}**")
    for up in files:
        with st.expander(f"📄 Preview: {up.name}", expanded=False):
            up.seek(0)
            df_raw = read_any(up)
            st.dataframe(df_raw.head(preview_rows), use_container_width=True)
            df_norm = normalize_market_df(df_raw, source_file=up.name)
            st.caption(f"Baris siap simpan dari file ini: {len(df_norm):,}")
            st.dataframe(df_norm.head(preview_rows), use_container_width=True)
            parsed.append(df_norm)

if st.button("🚀 Simpan ke `data_harian`", type="primary") and parsed:
    all_df = pd.concat(parsed, ignore_index=True) if parsed else pd.DataFrame()
    if all_df.empty:
        st.warning("Tidak ada baris valid untuk disimpan.")
        st.stop()

    # Dedup in-batch
    before = len(all_df)
    all_df = all_df.drop_duplicates(subset=['kode_saham','trade_date'], keep='last')
    after = len(all_df)
    if after < before:
        st.info(f"Menghapus duplikat dalam batch: {before - after:,} baris. Sisa: {after:,}")

    conn = get_db_connection()
    try:
        ensure_table_data_harian(conn)
        cur = conn.cursor()

        order = [
            'trade_date','kode_saham','nama_perusahaan','remarks',
            'sebelumnya','open_price','first_trade','tertinggi','terendah','penutupan','selisih',
            'volume','nilai','frekuensi','index_individual','offer','offer_volume','bid','bid_volume',
            'listed_shares','tradeable_shares','weight_for_index','foreign_sell','foreign_buy',
            'non_regular_volume','non_regular_value','non_regular_frequency','source_file'
        ]
        all_df = all_df[order]

        if mode.startswith("Lewati"):
            sql = """
            INSERT IGNORE INTO data_harian
            (trade_date, kode_saham, nama_perusahaan, remarks,
             sebelumnya, open_price, first_trade, tertinggi, terendah, penutupan, selisih,
             volume, nilai, frekuensi, index_individual, offer, offer_volume, bid, bid_volume,
             listed_shares, tradeable_shares, weight_for_index, foreign_sell, foreign_buy,
             non_regular_volume, non_regular_value, non_regular_frequency, source_file)
            VALUES
            (%s,%s,%s,%s,
             %s,%s,%s,%s,%s,%s,%s,
             %s,%s,%s,%s,%s,%s,%s,%s,
             %s,%s,%s,%s,%s,
             %s,%s,%s,%s)
            """
        else:
            # UPSERT → update kolom non-key
            sql = """
            INSERT INTO data_harian
            (trade_date, kode_saham, nama_perusahaan, remarks,
             sebelumnya, open_price, first_trade, tertinggi, terendah, penutupan, selisih,
             volume, nilai, frekuensi, index_individual, offer, offer_volume, bid, bid_volume,
             listed_shares, tradeable_shares, weight_for_index, foreign_sell, foreign_buy,
             non_regular_volume, non_regular_value, non_regular_frequency, source_file)
            VALUES
            (%s,%s,%s,%s,
             %s,%s,%s,%s,%s,%s,%s,
             %s,%s,%s,%s,%s,%s,%s,%s,
             %s,%s,%s,%s,%s,
             %s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
               nama_perusahaan=VALUES(nama_perusahaan),
               remarks=VALUES(remarks),
               sebelumnya=VALUES(sebelumnya),
               open_price=VALUES(open_price),
               first_trade=VALUES(first_trade),
               tertinggi=VALUES(tertinggi),
               terendah=VALUES(terendah),
               penutupan=VALUES(penutupan),
               selisih=VALUES(selisih),
               volume=VALUES(volume),
               nilai=VALUES(nilai),
               frekuensi=VALUES(frekuensi),
               index_individual=VALUES(index_individual),
               offer=VALUES(offer),
               offer_volume=VALUES(offer_volume),
               bid=VALUES(bid),
               bid_volume=VALUES(bid_volume),
               listed_shares=VALUES(listed_shares),
               tradeable_shares=VALUES(tradeable_shares),
               weight_for_index=VALUES(weight_for_index),
               foreign_sell=VALUES(foreign_sell),
               foreign_buy=VALUES(foreign_buy),
               non_regular_volume=VALUES(non_regular_volume),
               non_regular_value=VALUES(non_regular_value),
               non_regular_frequency=VALUES(non_regular_frequency),
               source_file=VALUES(source_file)
            """

        batch_size = 5000
        total_rows = len(all_df)
        total_batches = math.ceil(total_rows / batch_size)
        pbar = st.progress(0, text="Menyimpan ke database...")
        affected_total = 0

        def to_sql_tuple(row):
            tup = []
            for v in row:
                if isinstance(v, float):
                    tup.append(v if pd.notna(v) else None)
                else:
                    # strings/ints already fine; convert NaN to None
                    try:
                        if pd.isna(v):
                            tup.append(None)
                        else:
                            tup.append(v)
                    except Exception:
                        tup.append(v)
            return tuple(tup)

        rows_buffer = []
        b = 0
        for row in all_df.itertuples(index=False, name=None):
            rows_buffer.append(to_sql_tuple(row))
            if len(rows_buffer) >= batch_size:
                cur.executemany(sql, rows_buffer)
                conn.commit()
                affected_total += cur.rowcount or 0
                rows_buffer.clear()
                b += 1
                pbar.progress(min(b / total_batches, 1.0), text=f"Menyimpan batch {b}/{total_batches}...")
        if rows_buffer:
            cur.executemany(sql, rows_buffer)
            conn.commit()
            affected_total += cur.rowcount or 0
            b += 1
            pbar.progress(1.0, text=f"Menyimpan batch {b}/{b}...")

        cur.close()
        pbar.empty()
        st.success(f"Selesai. Baris diproses: {total_rows:,}. Baris terpengaruh menurut DB: {affected_total:,}.")
    except Exception as e:
        st.exception(e)
    finally:
        try:
            conn.close()
        except Exception:
            pass

# Footer
st.markdown("---")
st.caption("Jika struktur header file Anda berbeda, kabari saya — kita bisa tambahkan mapping khusus.")
