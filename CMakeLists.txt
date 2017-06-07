cmake_minimum_required (VERSION 3.5.2)

set(SOLUTION_NAME "Skylines")

project(${SOLUTION_NAME})

#testing enabled
enable_testing()

# Avoid minRelSize/RelWithDebugInfo configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

# Cmake Inclusions
include(cmake/util.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# =============================================================================
# Include third party libraries
# =============================================================================
set(APIS_PATH "$ENV{APIS_PATH_UdG}")

# Boost
include_directories("${APIS_PATH}/boost-1.60.0/include")
link_directories("${APIS_PATH}/boost-1.60.0/lib")

# Gtest and GMock
include_directories("${APIS_PATH}/googletest_with_gmock/include")
link_directories("${APIS_PATH}/googletest_with_gmock/lib")
set(LIBS ${LIBS} gtest gtest_main gmock gmock_main)

# Rapidjson
include_directories("${APIS_PATH}/rapidjson-master/include")

# Spdlog
include_directories("${APIS_PATH}/spdlog-master/include")

#qt
set(CMAKE_INCLUDE_CURRENT_DIR ON) # Find includes in corresponding build directories
# Find libraries
find_package(Qt5Widgets)
find_package(Qt5Core)
# OpenGL & GLEW library
find_package(OpenGL)

#dll post copy depencies. Debug purposes. In Linux we use stow
if(WIN32)
    set(DEPENDENCY_LIBS_AND_FILES_DEBUG_SPECIFIC
       #gtest
       "${APIS_PATH}/googletest_with_gmock/bin/Debug/gtest.dll"
       "${APIS_PATH}/googletest_with_gmock/bin/Debug/gtest_main.dll"
       "${APIS_PATH}/googletest_with_gmock/bin/Debug/gmock.dll"
       "${APIS_PATH}/googletest_with_gmock/bin/Debug/gmock_main.dll"

       #qt
       "C:/Qt/5.9/msvc2015_64/bin/Qt5Cored.dll"
       "C:/Qt/5.9/msvc2015_64/bin/Qt5Guid.dll"
       "C:/Qt/5.9/msvc2015_64/bin/Qt5Widgetsd.dll"
    )
    set(DEPENDENCY_LIBS_AND_FILES_RELEASE_SPECIFIC
       #gtest
       "${APIS_PATH}/googletest_with_gmock/bin/Release/gtest.dll"
       "${APIS_PATH}/googletest_with_gmock/bin/Release/gtest_main.dll"
       "${APIS_PATH}/googletest_with_gmock/bin/Release/gmock.dll"
       "${APIS_PATH}/googletest_with_gmock/bin/Release/gmock_main.dll"

       #qt
       "C:/Qt/5.9/msvc2015_64/bin/Qt5Core.dll"
       "C:/Qt/5.9/msvc2015_64/bin/Qt5Gui.dll"
       "C:/Qt/5.9/msvc2015_64/bin/Qt5Widgets.dll"
    )
endif()

add_definitions(-D_VARIADIC_MAX=10 -DGTEST_LINKED_AS_SHARED_LIBRARY=1 -DNOMINMAX)
if(WIN32)
    add_definitions(/MP) #MP --> Multiprocessor compiler
endif()


# Projects
include_directories(src)
add_subdirectory(src)

# #testing
# option(BUILD_AUTOMATIC_TESTS "Build automatic tests" ON)
# if(BUILD_AUTOMATIC_TESTS)
    # add_subdirectory(tests/unit)
    # add_subdirectory(tests/blackbox)
    # add_subdirectory(tests/benchmark)
# endif()
# option(BUILD_MANUAL_TESTS "Build manual tests" ON)
# if(BUILD_MANUAL_TESTS)
    # add_subdirectory(tests/smoke)
# endif()

