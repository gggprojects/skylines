# =============================================================================
# Virtual Folders
# =============================================================================
# Projects and source code is not located in the same subfolder. Thus, we need
# to set virtual folders. Those, have the same tree folder structure. Virtual
# Folders are a feature for Visual Studio and their definition shouldn't affect
# builds the other platforms including nmake
# =============================================================================

macro(set_target_virtual_folder SRCS TARGET_NAME)
  #add the version resource file to the root folder
  source_group("" FILES ${VERSION_RC})
  foreach(f ${SRCS})
      # Get the relative path of the file
      file(RELATIVE_PATH SRCGR ${PROJECT_SOURCE_DIR} ${f})
      string(FIND "${SRCGR}" "/" POS)
      # Extract the folder if exists, (Remove the filename part)
      string(REGEX REPLACE "(.*)(/[^/]*)$" "\\1" SRCGR ${SRCGR})

      if(${POS} GREATER -1)
        string(REPLACE / \\ SRCGR ${SRCGR})
        set_property(TARGET ${TARGET_NAME} PROPERTY FOLDER ${SRCGR})
      else()
        source_group("" FILES ${f})
      endif()
  endforeach()
endmacro()

# =============================================================================
# Add tests
# =============================================================================
# TODO, documentation for this function
# =============================================================================

function(ADD_UNIT_TEST)
    set(options OPTIONAL FAST)
    cmake_parse_arguments(GTEST "" "TEST_NAME;RUN_DIR" "SRCS;DEPS" ${ARGN})

    SET(SOURCES_COMPLETE_PATH "")
    foreach (src ${GTEST_SRCS})
      SET(SOURCES_COMPLETE_PATH ${SOURCES_COMPLETE_PATH} ${${SOLUTION_NAME}_SOURCE_DIR}${GTEST_SRC_FOLDER}/${src})
    endforeach()

    set_project_virtual_folders("${SOURCES_COMPLETE_PATH}")
    add_executable(${GTEST_TEST_NAME} ${SOURCES_COMPLETE_PATH})
    target_link_libraries(${GTEST_TEST_NAME} ${LIBS} ${GTEST_DEPS})
    set_target_virtual_folder("${SOURCES_COMPLETE_PATH}" ${GTEST_TEST_NAME})

    add_test(NAME ${GTEST_TEST_NAME} COMMAND ${GTEST_TEST_NAME})
endfunction()

function(ADD_SMOKE_TEST)
    set(options OPTIONAL FAST)
    cmake_parse_arguments(GTEST "" "TEST_NAME" "SRCS;DEPS" ${ARGN})

    SET(SOURCES_COMPLETE_PATH "")
    foreach (src ${GTEST_SRCS})
      SET(SOURCES_COMPLETE_PATH ${SOURCES_COMPLETE_PATH} ${${SOLUTION_NAME}_SOURCE_DIR}${GTEST_SRC_FOLDER}/${src})
    endforeach()

    set_project_virtual_folders("${SOURCES_COMPLETE_PATH}")
    add_executable(${GTEST_TEST_NAME} ${SOURCES_COMPLETE_PATH})
    target_link_libraries(${GTEST_TEST_NAME} ${LIBS} ${GTEST_DEPS})
    set_target_virtual_folder("${SOURCES_COMPLETE_PATH}" ${GTEST_TEST_NAME})
endfunction()

