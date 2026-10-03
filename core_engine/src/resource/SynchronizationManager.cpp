#include "SynchronizationManager.hpp"

bool SynchronizationManager::addMutex(const Mutex& mutex) {
    for (const auto& registered : mutexes) {
        if (registered.second.getID() == mutex.getID()) {
            return false;
        }
    }

    return mutexes.emplace(mutex.getName(), mutex).second;
}

bool SynchronizationManager::addSemaphore(const Semaphore& semaphore) {
    for (const auto& registered : semaphores) {
        if (registered.second.getID() == semaphore.getID()) {
            return false;
        }
    }

    return semaphores.emplace(semaphore.getName(), semaphore).second;
}

const Mutex* SynchronizationManager::getMutex(const std::string& name) const {
    const auto mutex = mutexes.find(name);
    return mutex == mutexes.end() ? nullptr : &mutex->second;
}

const Semaphore* SynchronizationManager::getSemaphore(const std::string& name) const {
    const auto semaphore = semaphores.find(name);
    return semaphore == semaphores.end() ? nullptr : &semaphore->second;
}

bool SynchronizationManager::acquireMutex(const PCB& process, const std::string& name) {
    auto mutex = mutexes.find(name);
    return mutex != mutexes.end() && mutex->second.acquire(process);
}

bool SynchronizationManager::releaseMutex(const PCB& process, const std::string& name) {
    auto mutex = mutexes.find(name);
    return mutex != mutexes.end() && mutex->second.release(process);
}

bool SynchronizationManager::waitSemaphore(const PCB&, const std::string& name) {
    auto semaphore = semaphores.find(name);
    return semaphore != semaphores.end() && semaphore->second.acquire();
}

bool SynchronizationManager::signalSemaphore(const PCB&, const std::string& name) {
    auto semaphore = semaphores.find(name);
    return semaphore != semaphores.end() && semaphore->second.release();
}
