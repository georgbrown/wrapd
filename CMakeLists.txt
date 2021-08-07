cmake_minimum_required(VERSION 3.1)
project(admm-elastic C CXX)

############################################################
#
#	Build Options
#
############################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/output")
add_definitions(-DWRAPD_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output")
add_definitions(-DWRAPD_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
option(MCL_BUILD_SAMPLES "Build mcl library samples" OFF)

# Lots of warnings
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-long-long")
endif()

############################################################
#
#	Libraries
#
############################################################

# Keep a variable of linked libs and link at end
set(LINKED_LIBS "")

# OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Eigen3 (header only)
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/Eigen3")
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# Check to see if PARDISO is available
find_library(PARDISO
  libpardiso600-GNU720-X86-64.so
  libpardiso600-GNU800-X86-64.so
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/deps/pardiso
)

if (${PARDISO} MATCHES "NOTFOUND")
  message("\nNOT USING PARDISO -- algorithm performance will be much slower than reported in the paper!\n")
else()
  message("Using PARDISO library at ${PARDISO}")
  add_definitions(-DUSE_PARDISO)
  set(ADDITIONAL_LIBRARIES ${PARDISO})
  set(LINKED_LIBS "${LINKED_LIBS};${PARDISO}")
endif()

# mcloptlib (header only), in deps/
set( MCLOPTLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/mcloptlib" )
include_directories( ${MCLOPTLIB_DIR}/include )

# Add project sources
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/src )
set( WRAPD_SRCS
	src/Math.hpp
	src/GlobalStepBehavior.hpp
	src/AlgorithmData.hpp
    src/AlgorithmData.cpp
	src/ADMM.hpp
	src/ADMM.cpp
	src/RuntimeData.hpp
	src/Settings.hpp
	src/Settings.cpp
	src/Solver.hpp
	src/Solver.cpp
	src/System.hpp
	src/System.cpp
	src/Lame.hpp
	src/Elements.hpp
	src/ModelSettings.hpp
	src/TetElements.hpp
	src/TetElements.cpp
)

# Finally, create the library
add_library( wrapd ${WRAPD_SRCS} )

set(LINKED_LIBS "${LINKED_LIBS};blas;lapack;gfortran;")

############################################################
#
#	Binaries
#
############################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
enable_testing()
add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/samples )
