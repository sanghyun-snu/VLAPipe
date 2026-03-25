cmake_minimum_required(VERSION 3.20)
project(pi0_sidecar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

set(SIDECAR_PROTO ${CMAKE_CURRENT_SOURCE_DIR}/proto/gpu_ipc_sidecar.proto)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

add_custom_command(
  OUTPUT
    ${GENERATED_DIR}/gpu_ipc_sidecar.pb.cc
    ${GENERATED_DIR}/gpu_ipc_sidecar.pb.h
    ${GENERATED_DIR}/gpu_ipc_sidecar.grpc.pb.cc
    ${GENERATED_DIR}/gpu_ipc_sidecar.grpc.pb.h
  COMMAND protobuf::protoc
  ARGS
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
    --cpp_out=${GENERATED_DIR}
    --grpc_out=${GENERATED_DIR}
    --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    ${SIDECAR_PROTO}
  DEPENDS ${SIDECAR_PROTO}
  VERBATIM
)

add_library(gpu_ipc_sidecar_proto
  ${GENERATED_DIR}/gpu_ipc_sidecar.pb.cc
  ${GENERATED_DIR}/gpu_ipc_sidecar.grpc.pb.cc
)
target_include_directories(gpu_ipc_sidecar_proto PUBLIC ${GENERATED_DIR})
target_link_libraries(gpu_ipc_sidecar_proto
  PUBLIC
    gRPC::grpc++
    protobuf::libprotobuf
)

add_executable(prefix_sidecar src/prefix_sidecar_main.cc)
target_include_directories(prefix_sidecar PRIVATE ${GENERATED_DIR})
target_link_libraries(prefix_sidecar PRIVATE gpu_ipc_sidecar_proto)

add_executable(suffix_sidecar src/suffix_sidecar_main.cc)
target_include_directories(suffix_sidecar PRIVATE ${GENERATED_DIR})
target_link_libraries(suffix_sidecar PRIVATE gpu_ipc_sidecar_proto)

