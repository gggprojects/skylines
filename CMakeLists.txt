cmake_minimum_required (VERSION 3.5.2)

set(SOLUTION_NAME "Skylines")

project(${SOLUTION_NAME})

#testing enabled
enable_testing()

# Avoid minRelSize/RelWithDebugInfo configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Cmake Inclusions
include(cmake/util.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# =============================================================================
# Include third party libraries
# =============================================================================
set(APIS_PATH "$ENV{APIS_PATH_UdG}")

# Boost
include_directories("${APIS_PATH}/boost-1.65.0/include")
link_directories("${APIS_PATH}/boost-1.65.0/lib")

# Gtest and GMock
include_directories("${APIS_PATH}/googletest-distribution/include")
link_directories("${APIS_PATH}/googletest-distribution/lib")
set(LIBS ${LIBS} gtest gtest_main gmock gmock_main)

# Celero
include_directories("${APIS_PATH}/celero-master/include")
link_directories("${APIS_PATH}/celero-master/lib")
set(LIBS ${LIBS} debug celerod optimized celero)

# Rapidjson
include_directories("${APIS_PATH}/rapidjson-master/include")

# Spdlog
include_directories("${APIS_PATH}/spdlog-master/include")

#qt
set(CMAKE_INCLUDE_CURRENT_DIR ON) # Find includes in corresponding build directories
find_package(Qt5Widgets)
find_package(Qt5Core)
find_package(OpenGL)

include_directories(${Qt5Widgets_INCLUDE_DIRS})

#opengl
set(LIBS ${LIBS} ${OPENGL_LIBRARIES})

# CSV parser
include_directories(SYSTEM "${APIS_PATH}/fast-cpp-csv-parser-master/include")

#CUDA
find_package(CUDA QUIET REQUIRED)
list(APPEND CUDA_NVCC_FLAGS --cl-version=2017)
set(CUDA_HOST_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.11.25503/bin/HostX64/x64/cl.exe" CACHE STRING "" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "" FORCE)
CUDA_INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON CACHE BOOL "" FORCE)

#dll post copy depencies. Debug purposes. In Linux we use stow
if(WIN32)
	set(DEPENDENCY_LIBS_AND_FILES
	   #cuda
	   "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_90.dll"
	)
    set(DEPENDENCY_LIBS_AND_FILES_DEBUG_SPECIFIC
       #gtest
       "${APIS_PATH}/googletest-distribution/lib/Debug/gtest.dll"
       "${APIS_PATH}/googletest-distribution/lib/Debug/gtest_main.dll"
       "${APIS_PATH}/googletest-distribution/lib/Debug/gmock.dll"
       "${APIS_PATH}/googletest-distribution/lib/Debug/gmock_main.dll"

       #qt
       "$ENV{CMAKE_PREFIX_PATH}/bin/Qt5Cored.dll"
       "$ENV{CMAKE_PREFIX_PATH}/bin/Qt5Guid.dll"
       "$ENV{CMAKE_PREFIX_PATH}/bin/Qt5Widgetsd.dll"

       #celero
       "${APIS_PATH}/celero-master/bin/celerod.dll"
    )
    set(DEPENDENCY_LIBS_AND_FILES_RELEASE_SPECIFIC
       #gtest
       "${APIS_PATH}/googletest-distribution/lib/Release/gtest.dll"
       "${APIS_PATH}/googletest-distribution/lib/Release/gtest_main.dll"
       "${APIS_PATH}/googletest-distribution/lib/Release/gmock.dll"
       "${APIS_PATH}/googletest-distribution/lib/Release/gmock_main.dll"

       #qt
       "$ENV{CMAKE_PREFIX_PATH}/bin/Qt5Core.dll"
       "$ENV{CMAKE_PREFIX_PATH}/bin/Qt5Gui.dll"
       "$ENV{CMAKE_PREFIX_PATH}/bin/Qt5Widgets.dll"

       #celero
       "${APIS_PATH}/celero-master/bin/celero.dll"
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

