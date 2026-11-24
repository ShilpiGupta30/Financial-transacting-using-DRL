-- Additive migration: simulation history only.
-- This script creates no database, does not alter Accounts/Transactions,
-- and does not modify financial records or balances.
USE resource_management_db;

CREATE TABLE IF NOT EXISTS SimulationHistory (
    SimulationID BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    ComparisonID CHAR(36) NULL,
    CreatedAt DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    Algorithm VARCHAR(32) NOT NULL,
    Parameters JSON NOT NULL,
    InputWorkload JSON NOT NULL,
    TransactionCount SMALLINT UNSIGNED NOT NULL,
    Makespan INT UNSIGNED NOT NULL,
    AverageWaitingTime DECIMAL(14, 3) NOT NULL,
    AverageTurnaroundTime DECIMAL(14, 3) NOT NULL,
    AverageResponseTime DECIMAL(14, 3) NOT NULL,
    CpuUtilization DECIMAL(8, 3) NULL,
    Throughput DECIMAL(14, 6) NULL,
    INDEX idx_SimulationHistory_CreatedAt (CreatedAt, SimulationID),
    INDEX idx_SimulationHistory_Algorithm (Algorithm, CreatedAt),
    INDEX idx_SimulationHistory_ComparisonID (ComparisonID)
);

CREATE TABLE IF NOT EXISTS SimulationHistoryResults (
    SimulationResultID BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    SimulationID BIGINT UNSIGNED NOT NULL,
    TransactionID INT NOT NULL,
    ProcessID INT NOT NULL,
    BurstTime INT UNSIGNED NOT NULL,
    CompletionTime INT UNSIGNED NOT NULL,
    WaitingTime INT UNSIGNED NOT NULL,
    TurnaroundTime INT UNSIGNED NOT NULL,
    ResponseTime INT UNSIGNED NOT NULL,
    UNIQUE KEY uq_SimulationHistoryResults_Transaction (SimulationID, TransactionID),
    INDEX idx_SimulationHistoryResults_Process (SimulationID, ProcessID),
    CONSTRAINT fk_SimulationHistoryResults_Simulation
        FOREIGN KEY (SimulationID) REFERENCES SimulationHistory (SimulationID)
        ON DELETE CASCADE
);
