#we store in variable SRCS all .cpp and .h and .cu files
file(GLOB_RECURSE SRCS *.hpp *.cpp *.cu *.cuh)
#we set the visual studio virtual folders
set_project_virtual_folders("${SRCS}")

include_directories(${Qt5Widgets_INCLUDE_DIRS})

add_library(skylines_engine SHARED "${SRCS}")
target_link_libraries(skylines_engine PRIVATE ${LIBS} ${OPENGL_LIBRARIES})

######################################################
# For VS environment purposes we copy the needed dll
# to the target folder
######################################################

############################################
# debug and release specific libs
############################################
foreach(f ${DEPENDENCY_LIBS_AND_FILES})
  add_custom_command( TARGET skylines_engine PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${f}" "$(OutDir)")
endforeach()

############################################
# debug specific libs
############################################

#several libs                    
foreach(f ${DEPENDENCY_LIBS_AND_FILES_DEBUG_SPECIFIC})
  add_custom_command( TARGET skylines_engine PRE_BUILD
                      COMMAND ${CMAKE_COMMAND}
                      -DBUILD_TYPE=${CMAKE_CFG_INTDIR}
                      -DOUT_DIR=$(OutDir)
                      -DFILE="${f}"
                      -DCURRENT_DLL="Debug"
                      -P ${CMAKE_CURRENT_LIST_DIR}/../../cmake/post_copy_dlls.cmake)
endforeach()

############################################
# release specific libs
############################################                
foreach(f ${DEPENDENCY_LIBS_AND_FILES_RELEASE_SPECIFIC})
  add_custom_command( TARGET skylines_engine PRE_BUILD
                      COMMAND ${CMAKE_COMMAND}
                      -DBUILD_TYPE=${CMAKE_CFG_INTDIR}
                      -DOUT_DIR=$(OutDir)
                      -DFILE="${f}"
                      -DCURRENT_DLL="Release"
                      -P ${CMAKE_CURRENT_LIST_DIR}/../../cmake/post_copy_dlls.cmake)
endforeach()
