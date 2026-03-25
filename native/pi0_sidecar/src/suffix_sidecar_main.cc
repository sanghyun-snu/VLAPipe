#include <grpcpp/grpcpp.h>

#include <cstdlib>
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
  explicit SidecarService(std::string upstream_address) : upstream_address_(std::move(upstream_address)) {
    if (!upstream_address_.empty()) {
      upstream_channel_ = grpc::CreateChannel(upstream_address_, grpc::InsecureChannelCredentials());
      upstream_stub_ = openpi::native::gpuipc::GpuIpcSidecar::NewStub(upstream_channel_);
    }
  }

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
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = handles_.find(MakeKey(*request));
      if (it != handles_.end()) {
        response->set_found(true);
        *response->mutable_handle() = it->second;
        response->set_message("ok");
        return grpc::Status::OK;
      }
    }

    if (upstream_stub_ != nullptr) {
      grpc::ClientContext upstream_context;
      openpi::native::gpuipc::ResolveHandleResponse upstream_response;
      grpc::Status upstream_status = upstream_stub_->ResolveHandle(&upstream_context, *request, &upstream_response);
      if (upstream_status.ok() && upstream_response.found()) {
        std::lock_guard<std::mutex> lock(mu_);
        handles_[MakeKey(*request)] = upstream_response.handle();
        response->set_found(true);
        *response->mutable_handle() = upstream_response.handle();
        response->set_message("ok_from_upstream");
        return grpc::Status::OK;
      }
    }

    response->set_found(false);
    response->set_message("not found");
    return grpc::Status::OK;
  }

  grpc::Status Healthz(grpc::ServerContext* /*context*/,
                       const openpi::native::gpuipc::HealthzRequest* /*request*/,
                       openpi::native::gpuipc::HealthzResponse* response) override {
    response->set_ok(true);
    response->set_message("suffix_sidecar_ready");
    return grpc::Status::OK;
  }

 private:
  std::string upstream_address_;
  std::shared_ptr<grpc::Channel> upstream_channel_;
  std::unique_ptr<openpi::native::gpuipc::GpuIpcSidecar::Stub> upstream_stub_;
  std::mutex mu_;
  std::unordered_map<std::string, openpi::native::gpuipc::GpuIpcHandle> handles_;
};

}  // namespace

int main(int argc, char** argv) {
  std::string bind_address = "0.0.0.0:55061";
  if (argc >= 2) {
    bind_address = argv[1];
  }
  const char* upstream_env = std::getenv("PI0_GPU_IPC_UPSTREAM");
  std::string upstream_address = upstream_env == nullptr ? "" : std::string(upstream_env);
  if (!upstream_address.empty()) {
    std::cout << "[pi0_sidecar] suffix sidecar upstream=" << upstream_address << std::endl;
  }

  SidecarService service(upstream_address);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(bind_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (!server) {
    std::cerr << "[pi0_sidecar] failed to start suffix sidecar at " << bind_address << std::endl;
    return 1;
  }
  std::cout << "[pi0_sidecar] suffix sidecar listening at " << bind_address << std::endl;
  server->Wait();
  return 0;
}

