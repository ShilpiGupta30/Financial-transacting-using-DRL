# Financial-transacting-using-DRL
The project demonstrates how OS resource management can affect concurrent financial transaction processing. Transactions will be represented as processes and managed through CPU scheduling, resource allocation, synchronization, and deadlock detection. The DBMS will maintain ACID properties and concurrency control.

**Phase 2 establishes the foundational components of the system**, including a C++17 core engine, process management, CPU scheduling, resource allocation, synchronization mechanisms, unit tests, and an initial relational database schema.

The current implementation provides the foundation on which future phases can integrate a DRL-based scheduling agent, performance evaluation, and a monitoring dashboard.

## 2. Objectives

The main objectives are:

- Model financial transactions as processes with identifiable execution states.
- Implement CPU scheduling algorithms for managing concurrent transaction workloads.
- Manage resource allocation and release.
- Provide synchronization primitives for coordinating concurrent operations.
- Establish a relational database foundation for project data.
- Validate core functionality through modular C++ tests.
- Create an extensible architecture for future Deep Reinforcement Learning integration.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Core engine | C++17 |
| Build and compilation | GCC / g++ |
| Build configuration | CMake configuration provided |
| Version control | Git and GitHub |
| Database foundation | SQL |
| Testing | C++ test programs and assertions |
| Future intelligent scheduling | Deep Reinforcement Learning |

## 4. Implementations

### 4.1 Process Control Block (PCB)

The Process Control Block represents the execution information associated with a transaction process.

The PCB model provides a foundation for tracking process information and managing its lifecycle within the resource-management engine.

**Implementation areas:**
- PCB data model.
- Transaction-related process representation.
- Process-state and execution information.
- Unit tests for PCB functionality.

**Relevant files:**
- `core_engine/include/PCB.hpp`
- `core_engine/include/Transaction.hpp`
- `core_engine/src/transaction/PCB.cpp`
- `core_engine/tests/test_pcb.cpp`

### 4.2 CPU Scheduling

The scheduling module provides the foundation for deciding which transaction process should receive CPU time.

The engine includes CPU scheduling functionality intended to support different scheduling strategies and enable comparison of transaction execution behavior.

**Implementation areas:**
- Scheduler interfaces and implementation.
- Scheduling algorithm support.
- Scheduling behavior validation through tests.

**Relevant files:**
- `core_engine/include/Scheduler.hpp`
- `core_engine/src/scheduler/Scheduler.cpp`
- `core_engine/tests/test_scheduler.cpp`

The scheduling module can later serve as a baseline for evaluating a DRL-based scheduling policy.

### 4.3 Resource Management

The resource-management module provides functionality for managing resources required by transaction processes.

Its purpose is to establish a structured interface for resource allocation and release and to provide a foundation for handling resource constraints during concurrent execution.

**Implementation areas:**
- Resource representation.
- Resource manager interface.
- Resource allocation and release operations.
- Tests for resource-management behavior.

**Relevant files:**
- `core_engine/include/Resource.hpp`
- `core_engine/include/ResourceManager.hpp`
- `core_engine/src/resource/ResourceManager.cpp`
- `core_engine/tests/test_resource_manager.cpp`

### 4.4 Synchronization Mechanisms

Concurrent transaction processing requires coordination when multiple processes access shared resources.

The synchronization module introduces primitives and management interfaces intended to support coordinated access and reduce unsafe concurrent interactions.

**Implementation areas:**
- Mutex interface.
- Semaphore interface.
- Synchronization manager.
- Tests for synchronization functionality.

**Relevant files:**
- `core_engine/include/Mutex.hpp`
- `core_engine/include/Semaphore.hpp`
- `core_engine/include/SynchronizationManager.hpp`
- `core_engine/src/resource/SynchronizationManager.cpp`
- `core_engine/tests/test_synchronization.cpp`

These components establish a foundation for synchronization-aware transaction scheduling. Full thread safety and deadlock handling should be validated against the actual implementation and workload before being claimed as complete.

### 4.5 Database Foundation

An initial relational database foundation has been added to support the future persistence and organization of project data.

**Database components:**
- SQL schema definition.
- Seed data for initial development and testing.
- Database documentation.

**Relevant files:**
- `database/README.md`
- `database/schema/schema.sql`
- `database/seeds/seed.sql`

The database layer is a foundation for future integration. A live connection between the database and the C++ engine should be considered complete only after it has been implemented and tested.

## 5. Project Structure

The following is the expected project structure based on the modules implemented during Phase 2. Exact filenames and folders should be checked against the current repository.

```text
Intelligent_Resource_Management/
│
├── core_engine/
│   ├── include/
│   │   ├── PCB.hpp
│   │   ├── Transaction.hpp
│   │   ├── Scheduler.hpp
│   │   ├── Resource.hpp
│   │   ├── ResourceManager.hpp
│   │   ├── Mutex.hpp
│   │   ├── Semaphore.hpp
│   │   └── SynchronizationManager.hpp
│   │
│   ├── src/
│   │   ├── transaction/
│   │   │   └── PCB.cpp
│   │   ├── scheduler/
│   │   │   └── Scheduler.cpp
│   │   └── resource/
│   │       ├── ResourceManager.cpp
│   │       └── SynchronizationManager.cpp
│   │
│   ├── tests/
│   │   ├── test_pcb.cpp
│   │   ├── test_scheduler.cpp
│   │   ├── test_resource_manager.cpp
│   │   └── test_synchronization.cpp
│   │
│   ├── CMakeLists.txt
│   └── README.md
│
├── database/
│   ├── schema/
│   │   └── schema.sql
│   ├── seeds/
│   │   └── seed.sql
│   └── README.md
│
└── README.md
```

## 6. Testing and Validation

The core engine has been developed in modular form, with separate test programs for its major components.

The previously reported test results are summarized below.

| Module | Reported assertions |
|---|---:|
| PCB | 8 |
| CPU scheduling | 12 |
| Resource management | 19 |
| Synchronization | 25 |
| **Total** | **64** |

**Previously reported result:** 64 assertions across the four modules.

These figures describe the test results reported during development. They should be re-run on the current branch before being presented as the latest verified result. Assertion counts are not necessarily equivalent to the number of independent test cases.

### Build prerequisites

- A C++17-compatible compiler, such as GCC.
- CMake, if using the provided CMake build configuration.
- Git for cloning and managing the repository.

### Build using CMake

From the project root, run:

```bash
cmake -S core_engine -B core_engine/build
cmake --build core_engine/build
```

Run the relevant test executables or test runner according to the targets configured in `core_engine/CMakeLists.txt`.

If CMake is not installed or the build configuration has not yet registered test targets, follow the instructions in `core_engine/README.md` and compile the available test programs using their required source files.

## 7. Architecture Overview

The current foundational architecture can be represented as follows:

```text
Financial Transaction Workload
              |
              v
       Transaction / PCB
              |
              v
        CPU Scheduler
              |
              v
       Resource Manager
              |
              v
   Synchronization Mechanisms
              |
              v
     Transaction Execution
```

The database foundation provides a separate persistence layer that can be connected to the processing engine as integration work progresses.

The future DRL component can interact with the scheduling layer to select scheduling actions based on the observed system state and a defined reward function. That integration is a future development objective, not a claim that the current scheduler is already controlled by a trained agent.


## 9. Current Scope and Limitations

Phase 2 focuses on foundational implementation.

The following should not be considered complete solely on the basis of the current Phase 2 work:

- A trained or deployed Deep Reinforcement Learning agent.
- Demonstrated improvement over traditional scheduling algorithms.
- A fully integrated, live database connection.
- A completed monitoring dashboard.
- Production-grade financial transaction processing.
- Verified end-to-end concurrency safety, deadlock prevention, or recovery guarantees.

These areas can be addressed through subsequent implementation, integration, and testing.

## 10. Future Work

The planned next stages include:

1. **DRL environment:** Define transaction workloads, observations, actions, and rewards.
2. **Agent implementation:** Implement and train a suitable DRL algorithm.
3. **Baseline comparison:** Compare the learned policy with conventional scheduling strategies.
4. **Performance evaluation:** Measure throughput, waiting time, turnaround time, and resource utilization.
5. **Database integration:** Connect the database layer to the application where required.
6. **Monitoring dashboard:** Visualize transaction activity, scheduler decisions, and performance metrics.
7. **System integration:** Validate behavior across the engine, database, agent, and dashboard.
8. **Final evaluation:** Document experiments, limitations, results, and conclusions.

## 12. Conclusion

Phase 2 establishes the foundational software components for an intelligent resource-management system targeting concurrent financial transactions.

The current work covers a C++17 process model, CPU scheduling, resource-management interfaces, synchronization mechanisms, modular tests, and an initial SQL database foundation.

These components provide a starting point for the next stage: integrating and evaluating a Deep Reinforcement Learning-based resource-management policy against conventional scheduling approaches.

