cmake_minimum_required (VERSION 2.8.8)
project (tf_example)

# set variables for external dependencies
set(EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/external" CACHE PATH "Location where external dependencies will installed")
set(DOWNLOAD_LOCATION "${EXTERNAL_DIR}/src" CACHE PATH "Location where external projects will be downloaded")
mark_as_advanced(EXTERNAL_DIR DOWNLOAD_LOCATION)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

include(Protobuf)
include(Eigen)

set(PROJECT_INCLUDE_DIRS ${EXTERNAL_DIR}/include)
set(PROJECT_DEPENDENCIES Eigen Protobuf)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -W")

include_directories(
  ${PROJECT_INCLUDE_DIRS}
  /data/learn/ml/tensorflow
  /data/learn/ml/tensorflow/bazel-genfiles
  /data/learn/ml/tensorflow/contrib/makefile/gen/protobuf/include
)

set(LINK_DEPENDS Eigen Protobuf)

add_subdirectory(example)
