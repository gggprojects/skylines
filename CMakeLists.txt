cmake_minimum_required (VERSION 3.5.2)

set(SOLUTION_NAME "Skylines")

project(${SOLUTION_NAME})

#testing enabled
enable_testing()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#prevent cmake to generate ZERO_CHECK project
set(CMAKE_SUPPRESS_REGENERATION true)

# Cmake Inclusions
include(cmake/util.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# =============================================================================
# Include third party libraries
# =============================================================================

# Boost
include_directories("submodule/apis_vs2015/boost-1.65.0")
link_directories("submodule/apis_vs2015/boost-1.65.0/lib64-msvc-14.0")

# Gtest and GMock
include_directories("submodule/apis_vs2015/googletest-distribution/include")
link_directories("submodule/apis_vs2015/googletest-distribution/lib")
set(LIBS ${LIBS} gtest gtest_main gmock gmock_main)

# Celero
include_directories("submodule/apis_vs2015/celero-master/include")
link_directories("submodule/apis_vs2015/celero-master/lib")
set(LIBS ${LIBS} debug celerod optimized celero)

# Rapidjson
include_directories("submodule/apis_vs2015/rapidjson-master/include")

# Spdlog
include_directories("submodule/apis_vs2015/spdlog-master/include")

#qt
set(CMAKE_PREFIX_PATH "submodule/apis_vs2015/Qt/msvc2015_64")
set(CMAKE_INCLUDE_CURRENT_DIR ON) # Find includes in corresponding build directories
find_package(Qt5Widgets)
find_package(Qt5Core)
find_package(OpenGL)

include_directories(${Qt5Widgets_INCLUDE_DIRS})

#opengl
set(LIBS ${LIBS} ${OPENGL_LIBRARIES})

#freeglut
include_directories("submodule/apis_vs2015/freeglut-3.0/include")
link_directories("submodule/apis_vs2015/freeglut-3.0/lib/x64")
set(LIBS ${LIBS} freeglut)

# CSV parser
include_directories(SYSTEM "submodule/apis_vs2015/fast-cpp-csv-parser-master/include")

#CUDA
find_package(CUDA QUIET REQUIRED)
#LIST(APPEND CUDA_NVCC_FLAGS --gpu-architecture=compute_32 --gpu-code=sm_62 --use_fast_math) #gpu-architecture == arch gpu-code == code
#LIST(APPEND CUDA_NVCC_FLAGS --gpu-architecture=compute_35 --gpu-code=sm_35)
LIST(APPEND CUDA_NVCC_FLAGS --gpu-architecture=compute_61 --gpu-code=sm_61) #GTX 1080
LIST(APPEND CUDA_NVCC_FLAGS_DEBUG -g -G)
CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src/skylines_engine)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "" FORCE)
include_directories(${CUDA_INCLUDE_DIRS})

#dll post copy depencies. Debug purposes. In Linux we use stow
if(WIN32)
	set(DEPENDENCY_LIBS_AND_FILES
	   #cuda
	   "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_100.dll"
	   
	   #freeglut
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/freeglut-3.0/bin/x64/freeglut.dll"
	)
    set(DEPENDENCY_LIBS_AND_FILES_DEBUG_SPECIFIC
       #gtest
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/googletest-distribution/bin/Debug/gtest.dll"
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/googletest-distribution/bin/Debug/gtest_main.dll"
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/googletest-distribution/bin/Debug/gmock.dll"
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/googletest-distribution/bin/Debug/gmock_main.dll"

       #qt
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/Qt/msvc2015_64/bin/Qt5Cored.dll"
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/Qt/msvc2015_64/bin/Qt5Guid.dll"
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/Qt/msvc2015_64/bin/Qt5Widgetsd.dll"

       #celero
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/celero-master/bin/celerod.dll"
    )
    set(DEPENDENCY_LIBS_AND_FILES_RELEASE_SPECIFIC
       #gtest
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/googletest-distribution/bin/Release/gtest.dll"
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/googletest-distribution/bin/Release/gtest_main.dll"
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/googletest-distribution/bin/Release/gmock.dll"
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/googletest-distribution/bin/Release/gmock_main.dll"

       #qt
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/Qt/msvc2015_64/bin/Qt5Core.dll"
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/Qt/msvc2015_64/bin/Qt5Gui.dll"
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/Qt/msvc2015_64/bin/Qt5Widgets.dll"

       #celero
       "${CMAKE_CURRENT_SOURCE_DIR}/submodule/apis_vs2015/celero-master/bin/celero.dll"
    )
endif()

add_definitions(-D_VARIADIC_MAX=10 -DGTEST_LINKED_AS_SHARED_LIBRARY=1 -DNOMINMAX)
if(WIN32)
    add_definitions(/MP) #MP --> Multiprocessor compiler
endif()


# Projects
add_subdirectory(src)
add_subdirectory(benchmarks)

add_subdirectory(tests/blackbox)

