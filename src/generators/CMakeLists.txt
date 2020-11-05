cmake_minimum_required(VERSION 3.5)
project(sltp VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Uncomment for debug build. Alternatively, build with `cmake -DCMAKE_BUILD_TYPE=Debug .`
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "-g -DDEBUG")
set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG")  # -Wall -Wextra


set(SOURCE_FILES
        src/featuregen.cpp
        src/sltp/dl/cache
        src/sltp/dl/elements
        src/sltp/dl/factory
        src/sltp/dl/types.hxx
        src/sltp/utils.hxx
        src/sltp/algorithms
        src/common/utils
        src/common/helpers
        src/common/base
        src/common/types
)


set(CNFGEN_SOURCE_FILES
        src/cnfgen.cpp
        src/cnf/transition_classification
        src/cnf/types
        src/cnf/generator
        src/cnf/cnfwriter.h
        src/common/utils
        src/common/helpers
        src/common/base
        src/common/types
        src/blai/sample.h
        src/blai/matrix.h
        src/blai/transitions.h
        src/blai/utils.h
)

add_executable(featuregen ${SOURCE_FILES})
add_executable(cnfgen ${CNFGEN_SOURCE_FILES})
set_target_properties(featuregen PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(cnfgen PROPERTIES LINKER_LANGUAGE CXX)



set(PROJECT_ROOT .)
#MESSAGE(WARNING "Your FS_ROOT home directory is: ${PROJECT_ROOT}" )

include_directories(${PROJECT_ROOT}/src)

# Allow custom boost installations
if(DEFINED ENV{BOOST_ROOT})
    message(STATUS "BOOST_ROOT variable set to '$ENV{BOOST_ROOT}'")
    set(BOOST_ROOT $ENV{HOME}/local)
endif()


find_package(Boost 1.65 COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(featuregen LINK_PUBLIC ${Boost_LIBRARIES})
target_link_libraries(cnfgen LINK_PUBLIC ${Boost_LIBRARIES})
