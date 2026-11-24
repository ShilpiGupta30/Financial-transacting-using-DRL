#pragma once

#include "DeadlockDetector.hpp"
#include "ResourceManager.hpp"
#include "Scheduler.hpp"
#include "SynchronizationManager.hpp"

#include <string>
#include <vector>

struct ResourceEvent {
    int time;
    std::string type;
    int transactionID;
    int processID;
    std::string resource;
    int availableUnits;
    std::string detail;
};

struct ResourceSnapshot {
    int resourceID;
    std::string name;
    int totalUnits;
    int availableUnits;
    double utilizationPercent;
};

struct ResourceSimulationResult {
    std::string scenario;
    std::vector<SchedulingResult> results;
    std::vector<PCB> processes;
    std::vector<ResourceEvent> events;
    std::vector<ResourceSnapshot> resources;
    std::vector<int> deadlockedProcessIDs;
    int makespan = 0;
    int allocations = 0;
    int releases = 0;
    int resourceWaitEvents = 0;
    int lockContentionEvents = 0;
    bool deadlockDetected = false;
    bool deadlockRecovered = false;
    int recoveryVictimProcessID = 0;
};

// Deterministic, single-threaded OS-style resource simulation. All requests and
// lock conflicts are simulated; no financial transaction data is changed.
class ResourceSimulation {
public:
    ResourceSimulationResult runContention(const std::vector<Transaction>& transactions,
                                           const std::vector<int>& burstTimes) const;
    ResourceSimulationResult runDeadlockDemo(const std::vector<Transaction>& transactions) const;
};
