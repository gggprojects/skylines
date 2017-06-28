
include_directories(../src/skylines_engine)

add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks ${LIBS} skylines_engine)
set_property(TARGET benchmarks PROPERTY FOLDER "benchmarks")