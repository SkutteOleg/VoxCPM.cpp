# Examples CMakeLists.txt

add_executable(voxcpm_tts voxcpm_tts.cpp)
target_link_libraries(voxcpm_tts PRIVATE voxcpm)

add_executable(voxcpm_quantize voxcpm_quantize.cpp)
target_link_libraries(voxcpm_quantize PRIVATE voxcpm)

add_executable(voxcpm_imatrix voxcpm_imatrix.cpp)
target_link_libraries(voxcpm_imatrix PRIVATE voxcpm)

add_executable(voxcpm-server ../servers/voxcpm_server.cpp)
target_include_directories(voxcpm-server PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/cpp-httplib
    ${CMAKE_SOURCE_DIR}/third_party/json/include
    ${json_SOURCE_DIR}/include
)
target_link_libraries(voxcpm-server PRIVATE voxcpm Threads::Threads)
