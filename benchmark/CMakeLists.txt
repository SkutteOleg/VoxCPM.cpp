add_executable(voxcpm_benchmark voxcpm_benchmark.cpp)
target_include_directories(voxcpm_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/json/include
    ${json_SOURCE_DIR}/include
)
target_link_libraries(voxcpm_benchmark PRIVATE voxcpm)

