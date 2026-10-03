cmake_minimum_required(VERSION 3.10)
project(IntelligentResourceManager)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Placeholder for future source files
set(SOURCES
    src/main.cpp
    src/transaction/PCB.cpp
    src/scheduler/Scheduler.cpp
    src/resource/ResourceManager.cpp
)

add_executable(IntelligentResourceManager ${SOURCES})

# Basic PCB Tests
add_executable(test_pcb tests/test_pcb.cpp src/transaction/PCB.cpp)

# Baseline scheduling algorithm tests
add_executable(test_scheduler
    tests/test_scheduler.cpp
    src/scheduler/Scheduler.cpp
    src/transaction/PCB.cpp
)

add_executable(test_resource_manager
    tests/test_resource_manager.cpp
    src/resource/ResourceManager.cpp
    src/transaction/PCB.cpp
)

enable_testing()
add_test(NAME pcb_tests COMMAND test_pcb)
add_test(NAME scheduler_tests COMMAND test_scheduler)
add_test(NAME resource_manager_tests COMMAND test_resource_manager)

# Future: Add Google Test, MySQL connector paths, etc.
