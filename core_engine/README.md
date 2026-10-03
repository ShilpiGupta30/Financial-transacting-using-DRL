# C++ Core Engine - Transaction and PCB

This folder contains the core C++ simulation engine for financial transactions. 

## Process Control Block (PCB) Model

In an Operating System, a PCB represents the execution state of a process. For our simulator, each financial transaction (e.g., Deposit, Withdrawal, Transfer) is wrapped inside a PCB to allow the system to simulate OS-level scheduling algorithms.

### PCB Information
A PCB tracks the following:
- **Process ID & Transaction ID**: Unique identifiers tying the OS process to the financial operation.
- **Timing**: 
  - `Arrival Time`: When the transaction entered the system.
  - `Burst Time`: Total CPU cycles required to execute the transaction.
  - `Remaining Time`: Cycles left until completion.
- **Priority**: How important the process is.
- **Process State**: The current execution status.

### Process States
The PCB can move through the following standard OS states:
- **NEW**: Process created but not yet admitted to the ready queue.
- **READY**: Process waiting for CPU time.
- **RUNNING**: Process is currently executing on the CPU.
- **WAITING**: Process is blocked (e.g., waiting for an I/O operation or a resource lock).
- **COMPLETED**: Process execution finished successfully.
- **FAILED**: Process execution aborted (e.g., due to a deadlock or a failed business logic like insufficient funds).
