#pragma once

#include "Mutex.hpp"
#include "Semaphore.hpp"

#include <map>
#include <string>

class SynchronizationManager {
public:
    bool addMutex(const Mutex& mutex);
    bool addSemaphore(const Semaphore& semaphore);

    const Mutex* getMutex(const std::string& name) const;
    const Semaphore* getSemaphore(const std::string& name) const;

    bool acquireMutex(const PCB& process, const std::string& name);
    bool releaseMutex(const PCB& process, const std::string& name);

    // Counting semaphores have no owner; process identifies the caller of wait/signal.
    bool waitSemaphore(const PCB& process, const std::string& name);
    bool signalSemaphore(const PCB& process, const std::string& name);

private:
    std::map<std::string, Mutex> mutexes;
    std::map<std::string, Semaphore> semaphores;
};
