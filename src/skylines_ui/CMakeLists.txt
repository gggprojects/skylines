#we store in variable UIS all .ui
file(GLOB_RECURSE UIS "*.ui")
#wraper to generate .h files from .ui files. The new .h files are stored in UIS_HEADERS variable
qt5_wrap_ui(UIS_HEADERS ${UIS})
#we set the visual studio virtual folders
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} FILES ${UIS})

# #we store in variable RESOURCES all .qrc
# FILE(GLOB_RECURSE RESOURCES "*.qrc")
# #wraper to generate .h files from .qrc files. The new .h files are stored in RESOURCES_HEADERS variable
# qt5_add_resources(RESOURCES_HEADERS ${RESOURCES})

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

#we store in variable SRCS all .cpp and .h and .cu files
file(GLOB_RECURSE SRCS *.hpp *.cpp)
#we set the visual studio virtual folders
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} FILES ${SRCS})

include_directories(../skylines_engine)
add_executable(skylines_ui "${SRCS}" ${UIS_HEADERS})
target_link_libraries(skylines_ui PRIVATE Qt5::Widgets ${OPENGL_LIBRARIES} skylines_engine)