-- MaxStock: MySQL schema for KSEI daily participation data
CREATE TABLE IF NOT EXISTS ksei_daily (
  base_symbol  VARCHAR(32)  NOT NULL,
  trade_date   DATE         NOT NULL,
  foreign_pct  DECIMAL(5,2)     NULL,
  retail_pct   DECIMAL(5,2)     NULL,
  total_volume BIGINT            NULL,
  total_value  BIGINT            NULL,
  created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (base_symbol, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE INDEX IF NOT EXISTS idx_ksei_trade_date ON ksei_daily (trade_date);
CREATE INDEX IF NOT EXISTS idx_ksei_base_symbol ON ksei_daily (base_symbol);
