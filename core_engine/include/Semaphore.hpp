#pragma once

#include <stdexcept>
#include <string>

class Semaphore {
public:
    Semaphore(int id, const std::string& semaphoreName, int maximum, int initialCount)
        : semaphoreID(id), name(semaphoreName), maximumCount(maximum),
          availableCount(initialCount) {
        if (semaphoreID < 0 || name.empty() || maximumCount < 0 ||
            initialCount < 0 || initialCount > maximumCount) {
            throw std::invalid_argument("Invalid semaphore ID, name, or count");
        }
    }

    int getID() const { return semaphoreID; }
    const std::string& getName() const { return name; }
    int getMaximumCount() const { return maximumCount; }
    int getAvailableCount() const { return availableCount; }

    bool acquire() {
        if (availableCount == 0) {
            return false;
        }
        --availableCount;
        return true;
    }

    bool release() {
        if (availableCount == maximumCount) {
            return false;
        }
        ++availableCount;
        return true;
    }

private:
    int semaphoreID;
    std::string name;
    int maximumCount;
    int availableCount;
};
