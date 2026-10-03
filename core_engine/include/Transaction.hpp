#pragma once

enum class TransactionType {
    DEPOSIT,
    WITHDRAWAL,
    TRANSFER,
    LOAN_PAYMENT
};

struct Transaction {
    int transactionID;
    TransactionType type;
    int sourceAccountID;
    int destinationAccountID; // Can be -1 if not applicable
    double amount;
    int priority;
};
