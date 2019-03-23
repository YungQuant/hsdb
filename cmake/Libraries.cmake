IF (CRYPTOPP_FOUND)
  INCLUDE_DIRECTORIES(${CRYPTOPP_INCLUDE_DIRS})
  SET(
    REQUIRED_LIBRARIES
    ${REQUIRED_LIBRARIES}
    ${CRYPTOPP_LIBRARIES}
  )
ENDIF (CRYPTOPP_FOUND)

FIND_PACKAGE(Boost COMPONENTS program_options thread chrono system REQUIRED)
FIND_PACKAGE(cpprestsdk REQUIRED)
FIND_PACKAGE(OPENSSL REQUIRED)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  SET(CURL_INCLUDE_DIR "/usr/local/opt/curl/include")
  SET(CURL_LIBRARIES "/usr/local/opt/curl/lib/libcurl.dylib")
else()
  FIND_PACKAGE(CURL REQUIRED)
endif()

INCLUDE_DIRECTORIES(${CURL_INCLUDE_DIR})
MESSAGE("-- CURL PATH \t\t" ${CURL_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${OPENSSL_INCLUDE_DIR})
MESSAGE("-- OPENSSL PATH \t\t" ${OPENSSL_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
MESSAGE("-- BOOST PATH \t\t" ${Boost_INCLUDE_DIR})

SET(
  REQUIRED_LIBRARIES
  ${REQUIRED_LIBRARIES}
  cpprestsdk::cpprest
  ${CURL_LIBRARIES}
  ${OPENSSL_LIBRARIES}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${Boost_CHRONO_LIBRARY}
)

IF (NOT HSDB_LIBRARY_DIR)
  MESSAGE("-- Setting default HSDB_LIBRARY_DIR Include Directory")
  SET(HSDB_LIBRARY_DIR "${CMAKE_CURRENT_LIST_DIR}/../lib")
ENDIF (NOT HSDB_LIBRARY_DIR)

MESSAGE("-- HSDB_LIBRARY_DIR\t" ${HSDB_LIBRARY_DIR})

INCLUDE_DIRECTORIES(${HSDB_LIBRARY_DIR})
LINK_DIRECTORIES(${HSDB_LIBRARY_DIR})