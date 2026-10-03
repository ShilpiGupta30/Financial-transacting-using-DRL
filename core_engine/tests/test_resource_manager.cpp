#include "ResourceManager.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int testsPassed = 0;
int testsFailed = 0;

void expect(bool condition, const std::string& description) {
    if (condition) {
        ++testsPassed;
        std::cout << "[PASS] " << description << '\n';
    } else {
        ++testsFailed;
        std::cout << "[FAIL] " << description << '\n';
    }
}

PCB makeProcess(int processID) {
    Transaction transaction{processID * 10, TransactionType::TRANSFER, 1, 2, 10.0, 1};
    return PCB(processID, transaction, 0, 5);
}

} // namespace

int main() {
    ResourceManager manager;
    Resource cpu(1, "CPU slots", 5);
    expect(cpu.totalUnits == 5 && cpu.availableUnits == 5, "Resource is created with total units available");
    expect(manager.addResource(cpu), "Resource can be registered");
    expect(manager.getResource(1) != nullptr && manager.getResource(1)->availableUnits == 5,
           "Initial availability can be inspected");

    PCB firstProcess = makeProcess(1);
    PCB secondProcess = makeProcess(2);

    expect(manager.allocate(firstProcess, 1, 3), "Process successfully allocates resource units");
    expect(manager.getResource(1)->availableUnits == 2, "Availability decreases after allocation");
    expect(manager.getAllocatedUnits(firstProcess, 1) == 3, "Allocation is recorded against its PCB");
    expect(!manager.allocate(secondProcess, 1, 3), "Allocation fails when units are insufficient");
    expect(manager.getResource(1)->availableUnits == 2, "Failed allocation leaves availability unchanged");

    expect(manager.release(firstProcess, 1, 2), "Process successfully releases allocated units");
    expect(manager.getResource(1)->availableUnits == 4, "Availability increases after release");
    expect(manager.getAllocatedUnits(firstProcess, 1) == 1, "Remaining allocation is tracked");
    expect(!manager.release(firstProcess, 1, 2), "Release rejects more units than this PCB holds");
    expect(!manager.release(secondProcess, 1, 1), "Release rejects units held by a different PCB");
    expect(manager.getResource(1)->availableUnits == 4, "Invalid releases leave availability unchanged");

    expect(manager.addResource(Resource(2, "Memory blocks", 8)), "A second resource can be registered");
    expect(manager.allocate(firstProcess, 2, 4), "One PCB can hold units from multiple resources");
    expect(manager.allocate(secondProcess, 1, 1), "A second PCB can allocate available units");
    expect(manager.getResource(1)->availableUnits == 3 && manager.getResource(2)->availableUnits == 4,
           "Resources maintain independent availability");
    expect(manager.getAllocatedUnits(firstProcess, 2) == 4 &&
           manager.getAllocatedUnits(secondProcess, 1) == 1,
           "Multiple PCBs keep separate resource allocations");

    std::cout << "\nResource manager tests: " << testsPassed << " passed, " << testsFailed << " failed.\n";
    return testsFailed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
