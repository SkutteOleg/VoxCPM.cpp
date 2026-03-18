add_library(Catch2WithMain STATIC extras/catch_amalgamated.cpp)
target_include_directories(Catch2WithMain PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extras)
add_library(Catch2::Catch2WithMain ALIAS Catch2WithMain)
