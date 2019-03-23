CMAKE_MINIMUM_REQUIRED(VERSION 3.7)
SET(CMAKE_CXX_STANDARD 11)

if (UNIX)
  ADD_DEFINITIONS(-Wno-sign-compare -Wno-enum-compare)
endif (UNIX)

PROJECT(HSDB)

INCLUDE("${CMAKE_CURRENT_LIST_DIR}/cmake/Cryptopp.cmake")
INCLUDE("${CMAKE_CURRENT_LIST_DIR}/cmake/Libraries.cmake")
INCLUDE("${CMAKE_CURRENT_LIST_DIR}/cmake/Compile.cmake")
