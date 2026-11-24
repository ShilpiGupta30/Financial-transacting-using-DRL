#include "ResourceSimulation.hpp"

#include "Mutex.hpp"

#include <algorithm>
#include <deque>
#include <stdexcept>

namespace {
void validate(const std::vector<Transaction>& transactions, const std::vector<int>& bursts) {
    if (transactions.empty() || transactions.size() != bursts.size())
        throw std::invalid_argument("transactions and burst_times must have the same non-zero length");
    for (int burst : bursts) if (burst <= 0) throw std::invalid_argument("burst times must be positive");
}

void snapshot(ResourceSimulationResult& output, const ResourceManager& manager, int id,
              int busyUnitTime, int horizon) {
    const Resource* resource = manager.getResource(id);
    if (resource == nullptr) return;
    const double denominator = static_cast<double>(resource->totalUnits) * horizon;
    output.resources.push_back(ResourceSnapshot{id, resource->name, resource->totalUnits,
        resource->availableUnits, denominator > 0 ? 100.0 * busyUnitTime / denominator : 0.0});
}
}

ResourceSimulationResult ResourceSimulation::runContention(
        const std::vector<Transaction>& transactions, const std::vector<int>& burstTimes) const {
    validate(transactions, burstTimes);
    ResourceSimulationResult out;
    out.scenario = "resource_contention";
    out.processes.reserve(transactions.size());
    for (std::size_t i = 0; i < transactions.size(); ++i) {
        out.processes.push_back(PCB::createFromTransaction(transactions[i], 0, burstTimes[i]));
        out.processes.back().changeState(ProcessState::READY);
    }
    ResourceManager resources;
    SynchronizationManager sync;
    resources.addResource(Resource{1, "SHARED_TRANSACTION_RESOURCE", 1});
    sync.addMutex(Mutex{1, "TRANSACTION_CRITICAL_SECTION"});
    std::vector<int> remaining = burstTimes, waiting(transactions.size(), 0);
    std::vector<int> completion(transactions.size(), 0);
    std::deque<std::size_t> ready;
    for (std::size_t i = 0; i < transactions.size(); ++i) ready.push_back(i);
    int tick = 0;
    while (!ready.empty()) {
        const std::size_t ownerIndex = ready.front(); ready.pop_front();
        PCB& owner = out.processes[ownerIndex];
        if (!resources.allocate(owner, 1, 1) || !sync.acquireMutex(owner, "TRANSACTION_CRITICAL_SECTION"))
            throw std::runtime_error("resource simulation failed to acquire an available resource");
        ++out.allocations;
        out.events.push_back({tick, "allocated", owner.transactionID, owner.processID,
                              "SHARED_TRANSACTION_RESOURCE", 0, "one unit allocated"});

        // Other ready transactions make explicit competing requests while the owner holds
        // the capacity-one resource and mutex. Failed requests enter WAITING without work.
        for (std::size_t i = 0; i < out.processes.size(); ++i) {
            if (i == ownerIndex || remaining[i] == 0) continue;
            PCB& contender = out.processes[i];
            const bool resourceDenied = !resources.allocate(contender, 1, 1);
            const bool lockDenied = !sync.acquireMutex(contender, "TRANSACTION_CRITICAL_SECTION");
            if (resourceDenied) {
                ++out.resourceWaitEvents; ++waiting[i];
                contender.changeState(ProcessState::WAITING);
                out.events.push_back({tick, "resource_wait", contender.transactionID, contender.processID,
                    "SHARED_TRANSACTION_RESOURCE", 0, "resource owned by another transaction"});
            }
            if (lockDenied) {
                ++out.lockContentionEvents;
                out.events.push_back({tick, "lock_contention", contender.transactionID, contender.processID,
                    "TRANSACTION_CRITICAL_SECTION", 0, "critical section is locked"});
            }
        }

        owner.changeState(ProcessState::RUNNING);
        --remaining[ownerIndex];
        owner.updateRemainingTime(1);
        out.events.push_back({tick, "executed", owner.transactionID, owner.processID,
                              "SHARED_TRANSACTION_RESOURCE", 0, "one CPU unit completed"});
        if (!sync.releaseMutex(owner, "TRANSACTION_CRITICAL_SECTION") || !resources.release(owner, 1, 1))
            throw std::runtime_error("resource simulation failed to release an owned resource");
        ++out.releases;
        out.events.push_back({tick + 1, "released", owner.transactionID, owner.processID,
                              "SHARED_TRANSACTION_RESOURCE", 1, "resource and mutex released"});
        ++tick;
        for (std::size_t i = 0; i < out.processes.size(); ++i)
            if (out.processes[i].state == ProcessState::WAITING) out.processes[i].changeState(ProcessState::READY);
        if (remaining[ownerIndex] == 0) {
            owner.changeState(ProcessState::COMPLETED);
            completion[ownerIndex] = tick;
        } else {
            owner.changeState(ProcessState::READY);
            ready.push_back(ownerIndex);
        }
    }
    out.makespan = tick;
    for (std::size_t i = 0; i < out.processes.size(); ++i) {
        out.results.push_back({out.processes[i].processID, completion[i], waiting[i], completion[i]});
    }
    snapshot(out, resources, 1, tick, tick);
    return out;
}

ResourceSimulationResult ResourceSimulation::runDeadlockDemo(
        const std::vector<Transaction>& transactions) const {
    if (transactions.size() != 2)
        throw std::invalid_argument("deadlock_demo requires exactly two transactions");
    ResourceSimulationResult out;
    out.scenario = "deadlock_demo";
    for (const Transaction& transaction : transactions) {
        out.processes.push_back(PCB::createFromTransaction(transaction, 0, 1));
        out.processes.back().changeState(ProcessState::READY);
    }
    ResourceManager resources;
    resources.addResource(Resource{1, "RESOURCE_A", 1});
    resources.addResource(Resource{2, "RESOURCE_B", 1});
    DeadlockDetector detector;
    for (const PCB& process : out.processes) detector.addProcess(process);
    PCB& first = out.processes[0]; PCB& second = out.processes[1];
    resources.allocate(first, 1, 1); ++out.allocations;
    out.events.push_back({0, "allocated", first.transactionID, first.processID, "RESOURCE_A", 0, "first lock acquired"});
    resources.allocate(second, 2, 1); ++out.allocations;
    out.events.push_back({1, "allocated", second.transactionID, second.processID, "RESOURCE_B", 0, "second lock acquired"});
    first.changeState(ProcessState::WAITING); detector.addDependency(first, second); ++out.resourceWaitEvents;
    out.events.push_back({2, "resource_wait", first.transactionID, first.processID, "RESOURCE_B", 0, "held by second process"});
    second.changeState(ProcessState::WAITING); detector.addDependency(second, first); ++out.resourceWaitEvents;
    out.events.push_back({2, "resource_wait", second.transactionID, second.processID, "RESOURCE_A", 0, "held by first process"});
    const DeadlockResult deadlock = detector.detectDeadlock();
    out.deadlockDetected = deadlock.hasDeadlock();
    out.deadlockedProcessIDs = deadlock.processIDs;
    if (!out.deadlockDetected) throw std::runtime_error("expected demonstration cycle was not detected");
    out.events.push_back({2, "deadlock_detected", 0, 0, "", 0, "wait-for graph contains a cycle"});

    // Deterministic recovery: abort the second process, release its held unit, remove
    // it from the wait-for graph, then let the first process acquire and release B.
    out.recoveryVictimProcessID = second.processID;
    second.changeState(ProcessState::FAILED);
    resources.release(second, 2, 1); ++out.releases;
    out.events.push_back({3, "released", second.transactionID, second.processID, "RESOURCE_B", 1, "recovery released the victim's unit"});
    detector.removeProcess(second.processID);
    first.changeState(ProcessState::READY);
    out.events.push_back({3, "deadlock_recovered", second.transactionID, second.processID,
        "RESOURCE_B", 1, "second process aborted; its resource released"});
    resources.allocate(first, 2, 1); ++out.allocations;
    out.events.push_back({3, "allocated", first.transactionID, first.processID, "RESOURCE_B", 0, "survivor acquired the released unit"});
    out.deadlockRecovered = !detector.hasDeadlock();
    first.updateRemainingTime(1);
    out.events.push_back({3, "executed", first.transactionID, first.processID, "CPU", 0, "survivor completed one CPU unit while holding its resources"});
    resources.release(first, 1, 1); ++out.releases;
    out.events.push_back({4, "released", first.transactionID, first.processID, "RESOURCE_A", 1, "survivor released resource A"});
    resources.release(first, 2, 1); ++out.releases;
    out.events.push_back({4, "released", first.transactionID, first.processID, "RESOURCE_B", 1, "survivor released resource B"});
    first.changeState(ProcessState::COMPLETED);
    out.makespan = 4;
    out.results.push_back({first.processID, 4, 3, 4});
    out.results.push_back({second.processID, 3, 1, 3});
    snapshot(out, resources, 1, out.makespan, out.makespan);
    snapshot(out, resources, 2, 3, out.makespan);
    return out;
}
