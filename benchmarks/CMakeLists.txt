
include_directories(../src/skylines_engine)

add_executable(benchmarks_new benchmarks_new.cpp)
target_link_libraries(benchmarks_new ${LIBS} skylines_engine)
set_property(TARGET benchmarks_new PROPERTY FOLDER "benchmarks")