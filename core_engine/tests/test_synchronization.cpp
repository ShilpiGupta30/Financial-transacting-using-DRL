#include "SynchronizationManager.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

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
    PCB firstProcess = makeProcess(1);
    PCB secondProcess = makeProcess(2);

    Mutex mutex(1, "account-lock");
    expect(!mutex.isLocked() && !mutex.getOwnerProcessID().has_value(), "Mutex starts unlocked");
    expect(mutex.acquire(firstProcess), "PCB acquires an unlocked mutex");
    expect(mutex.isLocked(), "Mutex reports locked after acquisition");
    expect(mutex.getOwnerProcessID() == std::optional<int>(firstProcess.processID),
           "Acquiring PCB is reported as mutex owner");
    expect(!mutex.acquire(secondProcess), "Another PCB cannot acquire a locked mutex");
    expect(mutex.release(firstProcess), "Mutex owner can release the mutex");
    expect(!mutex.isLocked() && !mutex.getOwnerProcessID().has_value(), "Mutex reports unlocked after release");
    expect(!mutex.release(secondProcess), "Non-owner cannot release the mutex");

    Semaphore semaphore(2, "worker-slots", 3, 2);
    expect(semaphore.getMaximumCount() == 3 && semaphore.getAvailableCount() == 2,
           "Semaphore starts with configured count and maximum");
    expect(semaphore.acquire() && semaphore.getAvailableCount() == 1, "Semaphore acquire decreases count");
    expect(semaphore.acquire() && semaphore.getAvailableCount() == 0, "Semaphore acquire succeeds while count remains");
    expect(!semaphore.acquire() && semaphore.getAvailableCount() == 0, "Semaphore acquire fails at zero");
    expect(semaphore.release() && semaphore.getAvailableCount() == 1, "Semaphore release increases count");
    expect(semaphore.release() && semaphore.release() && semaphore.getAvailableCount() == 3,
           "Semaphore count can return to its maximum");
    expect(!semaphore.release() && semaphore.getAvailableCount() == 3, "Release above maximum is rejected");

    SynchronizationManager manager;
    expect(manager.addMutex(Mutex(10, "manager-lock")), "Manager registers a mutex");
    expect(manager.addSemaphore(Semaphore(11, "manager-slots", 2, 1)), "Manager registers a semaphore");
    expect(manager.getMutex("manager-lock") != nullptr && manager.getMutex("missing-lock") == nullptr,
           "Manager retrieves mutexes and rejects unknown names");
    expect(manager.getSemaphore("manager-slots") != nullptr &&
           manager.getSemaphore("missing-slots") == nullptr,
           "Manager retrieves semaphores and rejects unknown names");
    expect(manager.acquireMutex(firstProcess, "manager-lock"), "PCB acquires mutex through manager");
    expect(manager.releaseMutex(firstProcess, "manager-lock"), "PCB releases mutex through manager");
    expect(manager.waitSemaphore(firstProcess, "manager-slots"), "PCB waits on semaphore through manager");
    expect(manager.signalSemaphore(firstProcess, "manager-slots"), "PCB signals semaphore through manager");
    expect(!manager.acquireMutex(firstProcess, "missing-lock") &&
           !manager.releaseMutex(firstProcess, "missing-lock"), "Manager rejects invalid mutex names");
    expect(!manager.waitSemaphore(firstProcess, "missing-slots") &&
           !manager.signalSemaphore(firstProcess, "missing-slots"), "Manager rejects invalid semaphore names");

    std::cout << "\nSynchronization tests: " << testsPassed << " passed, " << testsFailed << " failed.\n";
    return testsFailed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
