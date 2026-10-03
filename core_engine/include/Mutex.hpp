#pragma once

#include "PCB.hpp"

#include <optional>
#include <stdexcept>
#include <string>

class Mutex {
public:
    Mutex(int id, const std::string& mutexName) : mutexID(id), name(mutexName) {
        if (mutexID < 0 || name.empty()) {
            throw std::invalid_argument("Mutex ID must be non-negative and name must not be empty");
        }
    }

    int getID() const { return mutexID; }
    const std::string& getName() const { return name; }
    bool isLocked() const { return ownerProcessID.has_value(); }
    std::optional<int> getOwnerProcessID() const { return ownerProcessID; }

    // This is a simple, non-recursive simulation mutex.
    bool acquire(const PCB& process) {
        if (isLocked()) {
            return false;
        }
        ownerProcessID = process.processID;
        return true;
    }

    bool release(const PCB& process) {
        if (!isLocked() || ownerProcessID != process.processID) {
            return false;
        }
        ownerProcessID.reset();
        return true;
    }

private:
    int mutexID;
    std::string name;
    std::optional<int> ownerProcessID;
};
