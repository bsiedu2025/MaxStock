-- MaxStock: MySQL schema for RAW EOD uploads
-- Safe to run multiple times.
CREATE TABLE IF NOT EXISTS eod_prices_raw (
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Optional supporting indexes (speed up FF analysis and date filters)
CREATE INDEX IF NOT EXISTS idx_eod_tanggal ON eod_prices_raw (Tanggal);
CREATE INDEX IF NOT EXISTS idx_eod_ticker ON eod_prices_raw (Ticker);
