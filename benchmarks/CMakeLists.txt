
include_directories(../src/skylines_engine)

add_executable(benchmarks_all benchmarks_all.cpp)
target_link_libraries(benchmarks_all ${LIBS} skylines_engine)
set_property(TARGET benchmarks_all PROPERTY FOLDER "benchmarks")

add_executable(benchmarks_gpu_compare benchmarks_gpu_compare.cpp)
target_link_libraries(benchmarks_gpu_compare ${LIBS} skylines_engine)
set_property(TARGET benchmarks_gpu_compare PROPERTY FOLDER "benchmarks")