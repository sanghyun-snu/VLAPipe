#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "gpu_ipc_sidecar.grpc.pb.h"

namespace {

std::string MakeKey(const openpi::native::gpuipc::ResolveHandleRequest& request) {
  return request.request_id() + ":" + std::to_string(request.layer_idx()) + ":" + request.tensor_name();
}

std::string MakeKey(const openpi::native::gpuipc::PublishHandleRequest& request) {
  return request.request_id() + ":" + std::to_string(request.layer_idx()) + ":" + request.tensor_name();
}

class SidecarService final : public openpi::native::gpuipc::GpuIpcSidecar::Service {
 public:
  grpc::Status PublishHandle(grpc::ServerContext* /*context*/,
                             const openpi::native::gpuipc::PublishHandleRequest* request,
                             openpi::native::gpuipc::PublishHandleResponse* response) override {
    std::lock_guard<std::mutex> lock(mu_);
    handles_[MakeKey(*request)] = request->handle();
    response->set_ok(true);
    response->set_message("stored");
    return grpc::Status::OK;
  }

  grpc::Status ResolveHandle(grpc::ServerContext* /*context*/,
                             const openpi::native::gpuipc::ResolveHandleRequest* request,
                             openpi::native::gpuipc::ResolveHandleResponse* response) override {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = handles_.find(MakeKey(*request));
    if (it == handles_.end()) {
      response->set_found(false);
      response->set_message("not found");
      return grpc::Status::OK;
    }
    response->set_found(true);
    *response->mutable_handle() = it->second;
    response->set_message("ok");
    return grpc::Status::OK;
  }

  grpc::Status Healthz(grpc::ServerContext* /*context*/,
                       const openpi::native::gpuipc::HealthzRequest* /*request*/,
                       openpi::native::gpuipc::HealthzResponse* response) override {
    response->set_ok(true);
    response->set_message("prefix_sidecar_ready");
    return grpc::Status::OK;
  }

 private:
  std::mutex mu_;
  std::unordered_map<std::string, openpi::native::gpuipc::GpuIpcHandle> handles_;
};

}  // namespace

int main(int argc, char** argv) {
  std::string bind_address = "0.0.0.0:55062";
  if (argc >= 2) {
    bind_address = argv[1];
  }

  SidecarService service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(bind_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (!server) {
    std::cerr << "[pi0_sidecar] failed to start prefix sidecar at " << bind_address << std::endl;
    return 1;
  }
  std::cout << "[pi0_sidecar] prefix sidecar listening at " << bind_address << std::endl;
  server->Wait();
  return 0;
}

