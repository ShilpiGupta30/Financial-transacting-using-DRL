#pragma once

#include "PCB.hpp"

#include <vector>

// Metrics are returned in the same order as the input PCB list.
struct SchedulingResult {
    int processID;
    int completionTime;
    int waitingTime;
    int turnaroundTime;
};

class FCFSScheduler {
public:
    // Non-preemptive: processes run in arrival order. Ties keep input order.
    std::vector<SchedulingResult> schedule(std::vector<PCB>& processes) const;
};

class RoundRobinScheduler {
public:
    explicit RoundRobinScheduler(int timeQuantum);

    // Ready processes run for at most timeQuantum before returning to the queue.
    std::vector<SchedulingResult> schedule(std::vector<PCB>& processes) const;

private:
    int timeQuantum;
};

class PriorityScheduler {
public:
    // Non-preemptive. A larger PCB priority value means higher priority.
    // Equal priorities are resolved by arrival time, then input order.
    std::vector<SchedulingResult> schedule(std::vector<PCB>& processes) const;
};
