cmake_minimum_required(VERSION 3.10)
project(IntelligentResourceManager)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Placeholder for future source files
set(SOURCES
    src/main.cpp
    src/transaction/PCB.cpp
)

add_executable(IntelligentResourceManager ${SOURCES})

# Basic PCB Tests
add_executable(test_pcb tests/test_pcb.cpp src/transaction/PCB.cpp)

# Future: Add Google Test, MySQL connector paths, etc.
