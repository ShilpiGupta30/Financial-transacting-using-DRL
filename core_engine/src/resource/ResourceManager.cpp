#include "ResourceManager.hpp"

bool ResourceManager::addResource(const Resource& resource) {
    if (resource.resourceID < 0 || resource.name.empty() || resource.totalUnits < 0 ||
        resource.availableUnits != resource.totalUnits) {
        return false;
    }

    return resources.emplace(resource.resourceID, resource).second;
}

bool ResourceManager::hasAvailableUnits(int resourceID, int units) const {
    if (units <= 0) {
        return false;
    }

    const Resource* resource = getResource(resourceID);
    return resource != nullptr && resource->availableUnits >= units;
}

bool ResourceManager::allocate(PCB& process, int resourceID, int units) {
    if (!hasAvailableUnits(resourceID, units)) {
        return false;
    }

    Resource& resource = resources.at(resourceID);
    resource.availableUnits -= units;
    allocationsByProcess[process.processID][resourceID] += units;
    return true;
}

bool ResourceManager::release(PCB& process, int resourceID, int units) {
    if (units <= 0) {
        return false;
    }

    auto processAllocations = allocationsByProcess.find(process.processID);
    if (processAllocations == allocationsByProcess.end()) {
        return false;
    }

    auto resourceAllocation = processAllocations->second.find(resourceID);
    auto resource = resources.find(resourceID);
    if (resourceAllocation == processAllocations->second.end() || resource == resources.end() ||
        resourceAllocation->second < units || resource->second.availableUnits + units > resource->second.totalUnits) {
        return false;
    }

    resourceAllocation->second -= units;
    resource->second.availableUnits += units;

    if (resourceAllocation->second == 0) {
        processAllocations->second.erase(resourceAllocation);
    }
    if (processAllocations->second.empty()) {
        allocationsByProcess.erase(processAllocations);
    }

    return true;
}

const Resource* ResourceManager::getResource(int resourceID) const {
    const auto resource = resources.find(resourceID);
    return resource == resources.end() ? nullptr : &resource->second;
}

int ResourceManager::getAllocatedUnits(const PCB& process, int resourceID) const {
    const auto processAllocations = allocationsByProcess.find(process.processID);
    if (processAllocations == allocationsByProcess.end()) {
        return 0;
    }

    const auto resourceAllocation = processAllocations->second.find(resourceID);
    return resourceAllocation == processAllocations->second.end() ? 0 : resourceAllocation->second;
}
