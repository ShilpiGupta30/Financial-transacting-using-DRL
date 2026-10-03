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

## Baseline CPU Schedulers

The core engine provides FCFS, Round Robin, and non-preemptive Priority scheduling. Each scheduler accepts the existing `PCB` objects and returns completion, waiting, and turnaround times in the same order as the input list. A scheduling run resets each PCB's remaining time and state before it begins.

Priority scheduling chooses the largest `PCB::priority` value first. When priorities are equal, it chooses the earlier arrival; input order breaks any remaining tie. Round Robin uses the configured positive time quantum and a FIFO ready queue.

## Basic Resource Management

The resource manager tracks named resources by ID, including their total and available units. A process allocates units through the manager; successful allocations are recorded against that PCB's process ID and reduce availability. A process can release only units it previously allocated, which returns those units to availability. Allocation and release requests with invalid quantities or insufficient holdings are rejected. Mutexes, semaphores, synchronization, and deadlock detection are not implemented yet.
