
include_directories(../../src/skylines_engine)

add_executable(cpu_stbf_bbtest cpu_stbf_bbtest.cpp)
target_link_libraries(cpu_stbf_bbtest ${LIBS} skylines_engine)
set_property(TARGET cpu_stbf_bbtest PROPERTY FOLDER "tests/blackbox")