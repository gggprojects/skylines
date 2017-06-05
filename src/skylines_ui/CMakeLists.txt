#we store in variable UIS all .ui
file(GLOB_RECURSE UIS "*.ui")
#wraper to generate .h files from .ui files. The new .h files are stored in UIS_HEADERS variable
qt5_wrap_ui(UIS_HEADERS ${UIS})
#we set the visual studio virtual folders
#set_virtual_folders("${UIS}")

# #we store in variable RESOURCES all .qrc
# FILE(GLOB_RECURSE RESOURCES "*.qrc")
# #wraper to generate .h files from .qrc files. The new .h files are stored in RESOURCES_HEADERS variable
# qt5_add_resources(RESOURCES_HEADERS ${RESOURCES})

#we store in variable SRCS all .cpp and .h and .cu files
file(GLOB_RECURSE SRCS *.hpp *.cpp)
#we set the visual studio virtual folders
set_project_virtual_folders("${SRCS}")

add_executable(skylines_ui "${SRCS}" ${UIS_HEADERS})
target_link_libraries(skylines_ui PRIVATE Qt5::Widgets)