-- Intelligent Resource Management - Seed Data (Phase 2)

USE resource_management_db;

-- Clear existing data (if re-running the seed script)
SET FOREIGN_KEY_CHECKS = 0;
TRUNCATE TABLE Transactions;
TRUNCATE TABLE Accounts;
SET FOREIGN_KEY_CHECKS = 1;

-- -----------------------------------------------------------------------------
-- Seed Accounts
-- -----------------------------------------------------------------------------
INSERT INTO Accounts (AccountNumber, AccountHolderName, Balance, AccountStatus) VALUES
('ACC-1001', 'Alice Smith', 10500.00, 'ACTIVE'),
('ACC-1002', 'Bob Johnson', 4200.50, 'ACTIVE'),
('ACC-1003', 'Charlie Davis', 800.00, 'ACTIVE'),
('ACC-1004', 'Diana Prince', 150000.00, 'ACTIVE'),
('ACC-1005', 'Evan Wright', 0.00, 'SUSPENDED');

-- -----------------------------------------------------------------------------
-- Seed Transactions
-- -----------------------------------------------------------------------------
INSERT INTO Transactions (TransactionType, SourceAccountID, DestinationAccountID, Amount, Status, Priority) VALUES
-- A completed deposit to Alice's account (Account 1)
('DEPOSIT', 1, NULL, 500.00, 'COMPLETED', 1),

-- A completed withdrawal from Bob's account (Account 2)
('WITHDRAWAL', 2, NULL, 200.00, 'COMPLETED', 1),

-- A pending transfer from Alice (1) to Bob (2)
('TRANSFER', 1, 2, 1000.00, 'PENDING', 5),

-- A running transfer from Charlie (3) to Diana (4)
('TRANSFER', 3, 4, 500.00, 'RUNNING', 2),

-- A failed loan payment from Diana (4)
('LOAN_PAYMENT', 4, NULL, 1500.00, 'FAILED', 1),

-- A rolled back withdrawal from Evan (5) because his account is suspended and has 0 balance
('WITHDRAWAL', 5, NULL, 100.00, 'ROLLED_BACK', 1);
