#include "Scheduler.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <vector>

namespace {

void prepareProcesses(std::vector<PCB>& processes) {
    for (const PCB& process : processes) {
        if (process.arrivalTime < 0 || process.burstTime < 0) {
            throw std::invalid_argument("Arrival and burst times must be non-negative");
        }
    }

    // A scheduling run starts every supplied PCB from its original burst time.
    for (PCB& process : processes) {
        process.remainingTime = process.burstTime;
        process.changeState(ProcessState::NEW);
    }
}

std::vector<SchedulingResult> makeResults(std::size_t count) {
    return std::vector<SchedulingResult>(count, SchedulingResult{0, 0, 0, 0});
}

void recordCompletion(const PCB& process,
                      std::size_t inputIndex,
                      int completionTime,
                      std::vector<SchedulingResult>& results) {
    results[inputIndex] = SchedulingResult{
        process.processID,
        completionTime,
        completionTime - process.arrivalTime - process.burstTime,
        completionTime - process.arrivalTime
    };
}

std::vector<std::size_t> arrivalOrder(const std::vector<PCB>& processes) {
    std::vector<std::size_t> order(processes.size());
    for (std::size_t index = 0; index < processes.size(); ++index) {
        order[index] = index;
    }

    std::stable_sort(order.begin(), order.end(), [&processes](std::size_t left, std::size_t right) {
        return processes[left].arrivalTime < processes[right].arrivalTime;
    });
    return order;
}

} // namespace

std::vector<SchedulingResult> FCFSScheduler::schedule(std::vector<PCB>& processes) const {
    prepareProcesses(processes);
    std::vector<SchedulingResult> results = makeResults(processes.size());
    const std::vector<std::size_t> order = arrivalOrder(processes);
    int currentTime = 0;

    for (std::size_t index : order) {
        PCB& process = processes[index];
        if (currentTime < process.arrivalTime) {
            currentTime = process.arrivalTime;
        }

        process.changeState(ProcessState::READY);
        process.changeState(ProcessState::RUNNING);
        currentTime += process.remainingTime;
        process.updateRemainingTime(process.remainingTime);
        process.changeState(ProcessState::COMPLETED);
        recordCompletion(process, index, currentTime, results);
    }

    return results;
}

RoundRobinScheduler::RoundRobinScheduler(int quantum) : timeQuantum(quantum) {
    if (timeQuantum <= 0) {
        throw std::invalid_argument("Round Robin time quantum must be positive");
    }
}

std::vector<SchedulingResult> RoundRobinScheduler::schedule(std::vector<PCB>& processes) const {
    prepareProcesses(processes);
    std::vector<SchedulingResult> results = makeResults(processes.size());
    const std::vector<std::size_t> order = arrivalOrder(processes);
    std::queue<std::size_t> readyQueue;
    std::size_t nextArrival = 0;
    std::size_t completed = 0;
    int currentTime = 0;

    while (completed < processes.size()) {
        while (nextArrival < order.size() &&
               processes[order[nextArrival]].arrivalTime <= currentTime) {
            const std::size_t index = order[nextArrival++];
            processes[index].changeState(ProcessState::READY);
            readyQueue.push(index);
        }

        if (readyQueue.empty()) {
            // No process is ready; jump to the next arrival rather than ticking idle time.
            currentTime = processes[order[nextArrival]].arrivalTime;
            continue;
        }

        const std::size_t index = readyQueue.front();
        readyQueue.pop();
        PCB& process = processes[index];
        process.changeState(ProcessState::RUNNING);

        const int runTime = std::min(timeQuantum, process.remainingTime);
        currentTime += runTime;
        process.updateRemainingTime(runTime);

        // Arrivals during this time slice join the queue before the preempted process.
        while (nextArrival < order.size() &&
               processes[order[nextArrival]].arrivalTime <= currentTime) {
            const std::size_t arrivingIndex = order[nextArrival++];
            processes[arrivingIndex].changeState(ProcessState::READY);
            readyQueue.push(arrivingIndex);
        }

        if (process.remainingTime == 0) {
            process.changeState(ProcessState::COMPLETED);
            recordCompletion(process, index, currentTime, results);
            ++completed;
        } else {
            process.changeState(ProcessState::READY);
            readyQueue.push(index);
        }
    }

    return results;
}

std::vector<SchedulingResult> PriorityScheduler::schedule(std::vector<PCB>& processes) const {
    prepareProcesses(processes);
    std::vector<SchedulingResult> results = makeResults(processes.size());
    std::vector<bool> completed(processes.size(), false);
    std::size_t completedCount = 0;
    int currentTime = 0;

    while (completedCount < processes.size()) {
        std::size_t selected = processes.size();

        for (std::size_t index = 0; index < processes.size(); ++index) {
            if (completed[index] || processes[index].arrivalTime > currentTime) {
                continue;
            }

            if (selected == processes.size() ||
                processes[index].priority > processes[selected].priority ||
                (processes[index].priority == processes[selected].priority &&
                 processes[index].arrivalTime < processes[selected].arrivalTime)) {
                selected = index;
            }
        }

        if (selected == processes.size()) {
            int nextTime = processes.size() > 0 ? processes[0].arrivalTime : 0;
            for (std::size_t index = 0; index < processes.size(); ++index) {
                if (!completed[index] &&
                    (nextTime <= currentTime || processes[index].arrivalTime < nextTime)) {
                    nextTime = processes[index].arrivalTime;
                }
            }
            currentTime = nextTime;
            continue;
        }

        PCB& process = processes[selected];
        process.changeState(ProcessState::READY);
        process.changeState(ProcessState::RUNNING);
        currentTime += process.remainingTime;
        process.updateRemainingTime(process.remainingTime);
        process.changeState(ProcessState::COMPLETED);
        recordCompletion(process, selected, currentTime, results);
        completed[selected] = true;
        ++completedCount;
    }

    return results;
}
