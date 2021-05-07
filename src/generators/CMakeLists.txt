cmake_minimum_required(VERSION 3.5)
project(sltp VERSION 0.1 LANGUAGES CXX)

set(PROJECT_ROOT .)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Uncomment for debug build. Alternatively, build with `cmake -DCMAKE_BUILD_TYPE=Debug .`
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "-g -DDEBUG")
set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG")  # -Wall -Wextra


set(COMMON_FILES
    src/blai/matrix.cxx
    src/common/utils.h
    src/common/utils.cxx
    src/common/helpers
    src/common/base
    src/common/types
)

set(SOURCE_FILES
    src/featuregen.cpp
    src/sltp/dl/cache
    src/sltp/dl/elements
    src/sltp/dl/factory
    src/sltp/dl/types.hxx
    src/sltp/algorithms
)


set(CNFGEN_SOURCE_FILES
    src/cnfgen.cpp
    src/cnf/d2l.cxx
    src/cnf/sd2l.cxx
    src/cnf/dtl.cxx
    src/cnf/encoding_factory.h
    src/cnf/types.cxx
    src/cnf/generator
    src/cnf/cnfwriter.h
    src/cnf/options.h
    src/cnf/options.cxx
    src/cnf/sample.h
    src/cnf/sample.cxx
    src/cnf/solver.h
    src/cnf/solver.cxx
)

add_executable(featuregen ${SOURCE_FILES} ${COMMON_FILES})
add_executable(cnfgen ${CNFGEN_SOURCE_FILES} ${COMMON_FILES})


set_target_properties(featuregen PROPERTIES
        LINKER_LANGUAGE CXX
        RUNTIME_OUTPUT_DIRECTORY_RELEASE  ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(cnfgen PROPERTIES
        LINKER_LANGUAGE CXX
        RUNTIME_OUTPUT_DIRECTORY_RELEASE  ${CMAKE_CURRENT_SOURCE_DIR})

#MESSAGE(WARNING "Your PROJECT_ROOT directory is: ${PROJECT_ROOT}" )
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
