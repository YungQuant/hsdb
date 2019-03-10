CMAKE_MINIMUM_REQUIRED(VERSION 3.7)
SET(CMAKE_CXX_STANDARD 11)

# These definitions make sure UNIX OSes don't throw errors on build
if (UNIX)
  ADD_DEFINITIONS(-Wno-sign-compare -Wno-enum-compare)
endif (UNIX)

PROJECT(HSDB)

# Preset OpenSSL directories
IF (NOT OPENSSL_ROOT_DIR)
  MESSAGE("-- Setting default OpenSSL Root Directory")
  SET(OPENSSL_ROOT_DIR "/usr/local/opt/openssl")
ENDIF (NOT OPENSSL_ROOT_DIR)

IF (NOT OPENSSL_LIBRARIES)
  MESSAGE("-- Setting default OpenSSL Library Directory")
  SET(OPENSSL_LIBRARIES "/usr/local/opt/openssl/lib")
ENDIF (NOT OPENSSL_LIBRARIES)

MESSAGE("-- OPENSSL ROOT PATH \t" ${OPENSSL_ROOT_DIR})
MESSAGE("-- OPENSSL LIBRARIES \t" ${OPENSSL_LIBRARIES})

FIND_PACKAGE(Boost COMPONENTS program_options thread chrono system REQUIRED)
FIND_PACKAGE(cpprestsdk REQUIRED)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

MESSAGE("-- BOOST PATH \t\t" ${Boost_INCLUDE_DIR})

SET(
  REQUIRED_LIBRARIES
  cpprestsdk::cpprest
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${Boost_CHRONO_LIBRARY}
)

# Create executable
ADD_EXECUTABLE(hsdb src/hsdb.cpp)

# Link libraries
TARGET_LINK_LIBRARIES(
    hsdb
    PRIVATE
    ${REQUIRED_LIBRARIES}
)

# Link definitions
TARGET_COMPILE_DEFINITIONS(
  hsdb
  PRIVATE
  ${REQUIRED_DEFINITIONS}
)
