# Database Foundations

This directory contains the database schema and seed data for the Intelligent Resource Management project (Phase 2 MVP). 
MySQL provides the underlying storage and handles ACID properties for financial transactions executed by the C++ engine.

## 1. What the database contains
The database currently focuses on the core OLTP (Online Transaction Processing) requirements:
- **Accounts**: Simulates financial accounts with balances.
- **Transactions**: Logs the financial operations (deposits, transfers, etc.) processed by our scheduling system.

## 2. How to create the database
The database (`resource_management_db`) is automatically created when you execute the `schema.sql` script. You must have a MySQL server running locally.

## 3. How to run schema.sql
To initialize the database and create the tables, run the following command from the project root:

```bash
mysql -u root -p < database/schema/schema.sql
```
*(Enter your MySQL root password when prompted)*

## 4. How to run seed.sql
To populate the tables with synthetic demo data, run:

```bash
mysql -u root -p < database/seeds/seed.sql
```

## 5. What each table represents
- **`Accounts`**: 
  - Represents the financial entities holding funds.
  - Contains identifiers (`AccountID`, `AccountNumber`), holder information, the current `Balance` (enforced to be $\ge 0$ via constraints), and `AccountStatus` (e.g., ACTIVE, SUSPENDED).
- **`Transactions`**:
  - Represents the operations performed on accounts. 
  - Tracks the `TransactionType` (DEPOSIT, WITHDRAWAL, TRANSFER, LOAN_PAYMENT), the participating accounts (`SourceAccountID`, `DestinationAccountID`), the `Amount`, and the scheduling `Priority`. 
  - Crucially, it tracks the `Status` (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `ROLLED_BACK`) which changes as the C++ simulation schedules and executes the transaction against the database.
