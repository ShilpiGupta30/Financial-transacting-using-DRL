#pragma once

#include <string>

struct Resource {
    int resourceID;
    std::string name;
    int totalUnits;
    int availableUnits;

    Resource(int id, const std::string& resourceName, int units)
        : resourceID(id), name(resourceName), totalUnits(units), availableUnits(units) {}
};
