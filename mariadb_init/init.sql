-- D:\Docker\BrokerSummary\mariadb_init\init.sql
-- Skrip ini akan dijalankan secara otomatis saat container MariaDB pertama kali dibuat.
-- Ini akan membuat tabel dan mengisi data awal jika tabel kosong.

-- Menggunakan database yang telah dibuat oleh Docker Compose
USE Broker_DB;

-- Membuat tabel untuk data transaksi mentah
CREATE TABLE IF NOT EXISTS transactions_raw (
    Tanggal_transaksi TEXT,
    Kode_broker TEXT,
    Jenis_transaksi TEXT,
    Volume DECIMAL(15,2),
    Nilai_transaksi DECIMAL(20,2),
    Net_VAL DECIMAL(20,2),
    Net_LOT DECIMAL(15,2),
    INDEX idx_transaction_key (Tanggal_transaksi(10), Kode_broker(10), Jenis_transaksi(4))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Membuat tabel untuk informasi broker mentah
CREATE TABLE IF NOT EXISTS broker_info_raw (
    Kode VARCHAR(50) PRIMARY KEY, -- VARCHAR dengan panjang yang sesuai untuk PRIMARY KEY
    Nama_Broker TEXT,
    Jenis_Broker TEXT,
    Deskripsi TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- BARU: Membuat tabel untuk data historis harga saham
CREATE TABLE IF NOT EXISTS stock_prices_history (
    Ticker VARCHAR(20) NOT NULL,
    Tanggal DATE NOT NULL,
    Open DECIMAL(19,4), -- Menggunakan DECIMAL untuk presisi harga
    High DECIMAL(19,4),
    Low DECIMAL(19,4),
    Close DECIMAL(19,4),
    Volume BIGINT,
    -- Dividends DECIMAL(10,4), -- Opsional, yfinance dengan auto_adjust=True tidak menyertakannya di history utama
    -- Stock_Splits DECIMAL(10,4), -- Opsional
    PRIMARY KEY (Ticker, Tanggal) -- Kunci utama gabungan
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- Memasukkan data contoh awal ke tabel transactions_raw
INSERT IGNORE INTO transactions_raw (Tanggal_transaksi, Kode_broker, Jenis_transaksi, Volume, Nilai_transaksi, Net_VAL, Net_LOT) VALUES
('2023-03-10', 'SQ', 'BUY', 1100, 1127500000, 1127500000, 1100),
('2023-03-10', 'OD', 'SELL', 1000, 1025000000, -1025000000, -1000),
('2023-03-11', 'LG', 'BUY', 700, 717500000, 717500000, 700),
('2023-03-11', 'EP', 'SELL', 600, 615000000, -615000000, -600),
('2023-03-12', 'AZ', 'BUY', 500, 512500000, 512500000, 500);

-- Memasukkan data contoh awal ke tabel broker_info_raw
INSERT INTO broker_info_raw (Kode, Nama_Broker, Jenis_Broker, Deskripsi) VALUES
('SQ', 'PT. SEQUOIA SEKURITAS INDONESIA', 'ASING', 'Broker Asing'),
('OD', 'PT. CITIGROUP SEKURITIES INDONESIA', 'ASING', 'Broker Asing'),
('LG', 'PT. TRIMEGAH SEKURITAS INDONESIA TBK.', 'LOKAL', 'Broker Lokal'),
('EP', 'PT. MNC SEKURITAS', 'LOKAL', 'Broker Lokal'),
('AZ', 'PT. SUCOR SEKURITAS', 'LOKAL', 'Broker Lokal')
ON DUPLICATE KEY UPDATE Nama_Broker=VALUES(Nama_Broker), Jenis_Broker=VALUES(Jenis_Broker), Deskripsi=VALUES(Deskripsi);

COMMIT;
