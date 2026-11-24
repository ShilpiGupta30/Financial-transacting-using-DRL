-- Intelligent Resource Management - Database Schema (Phase 2)

CREATE DATABASE IF NOT EXISTS resource_management_db;
USE resource_management_db;

-- -----------------------------------------------------------------------------
-- Table: Accounts
-- Description: Financial accounts participating in transactions.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Accounts (
    AccountID INT AUTO_INCREMENT PRIMARY KEY,
    AccountNumber VARCHAR(20) NOT NULL UNIQUE,
    AccountHolderName VARCHAR(100) NOT NULL,
    Balance DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
    AccountStatus ENUM('ACTIVE', 'SUSPENDED', 'CLOSED') NOT NULL DEFAULT 'ACTIVE',
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_Balance CHECK (Balance >= 0.00)
);

-- Index for quick lookups by account number
CREATE INDEX idx_AccountNumber ON Accounts(AccountNumber);

-- -----------------------------------------------------------------------------
-- Table: Transactions
-- Description: Records of financial operations between accounts.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Transactions (
    TransactionID INT AUTO_INCREMENT PRIMARY KEY,
    TransactionType ENUM('DEPOSIT', 'WITHDRAWAL', 'TRANSFER', 'LOAN_PAYMENT') NOT NULL,
    SourceAccountID INT NOT NULL,
    DestinationAccountID INT NULL,
    Amount DECIMAL(15, 2) NOT NULL,
    Status ENUM('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'ROLLED_BACK') NOT NULL DEFAULT 'PENDING',
    Priority INT NOT NULL DEFAULT 0,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CompletedAt TIMESTAMP NULL,
    FOREIGN KEY (SourceAccountID) REFERENCES Accounts(AccountID),
    FOREIGN KEY (DestinationAccountID) REFERENCES Accounts(AccountID),
    CONSTRAINT chk_Amount CHECK (Amount > 0.00)
);

-- Indexes for performance and querying active transactions
CREATE INDEX idx_TransactionStatus ON Transactions(Status);
CREATE INDEX idx_SourceAccount ON Transactions(SourceAccountID);
CREATE INDEX idx_DestinationAccount ON Transactions(DestinationAccountID);
