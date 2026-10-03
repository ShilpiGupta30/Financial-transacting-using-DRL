#pragma once

#include "PCB.hpp"
#include "Resource.hpp"

#include <map>

class ResourceManager {
public:
    // Adds a resource with all units available. Returns false for invalid or duplicate IDs.
    bool addResource(const Resource& resource);

    // Returns true when the resource exists and has at least the requested positive units.
    bool hasAvailableUnits(int resourceID, int units) const;

    // Allocation and release are recorded against the PCB's processID.
    bool allocate(PCB& process, int resourceID, int units);
    bool release(PCB& process, int resourceID, int units);

    // Returns nullptr for an unknown ID. Resource availability can be inspected through it.
    const Resource* getResource(int resourceID) const;
    int getAllocatedUnits(const PCB& process, int resourceID) const;

private:
    std::map<int, Resource> resources;
    std::map<int, std::map<int, int>> allocationsByProcess;
};
