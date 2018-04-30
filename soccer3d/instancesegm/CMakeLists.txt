cmake_minimum_required(VERSION 3.5)
project(instancesegm)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Eigen
set(MIN_EIGEN_VERSION 3.2.0)
find_package(Eigen ${MIN_EIGEN_VERSION} REQUIRED)
if (EIGEN_FOUND)
    message("-- Found Eigen version ${EIGEN_VERSION}: ${EIGEN_INCLUDE_DIRS}")
endif (EIGEN_FOUND)

find_package( OpenCV REQUIRED )
find_package(Boost 1.45.0 COMPONENTS filesystem regex program_options)
find_package(OpenMP)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


set(CMAKE_CXX_STANDARD 11)

include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp core.cpp core.h)
add_executable(instancesegm ${SOURCE_FILES})
target_link_libraries (instancesegm ${OpenCV_LIBS} ${Boost_LIBRARIES})
