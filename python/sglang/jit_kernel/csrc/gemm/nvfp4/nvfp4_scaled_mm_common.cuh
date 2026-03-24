/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <unordered_map>

using namespace host;

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

#define CUTLASS_CHECK(status)                                                        \
  {                                                                                  \
    cutlass::Status error = status;                                                  \
    RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

using namespace cute;

inline uint32_t next_pow_2(uint32_t x) noexcept {
  if (x <= 1) return 1;
  return 1u << (32 - __builtin_clz(x - 1));
}

struct WorkspaceKey {
  int device_id;
  uintptr_t stream;
  auto operator==(const WorkspaceKey&) const -> bool = default;
};

struct WorkspaceKeyHash {
  auto operator()(const WorkspaceKey& key) const -> size_t {
    size_t h1 = std::hash<int>{}(key.device_id);
    size_t h2 = std::hash<uintptr_t>{}(key.stream);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

struct WorkspaceState {
  void* ptr = nullptr;
  size_t bytes = 0;
};

inline auto get_cached_workspace(size_t required_bytes, int device_id, cudaStream_t stream) -> void* {
  if (required_bytes == 0) {
    return nullptr;
  }

  thread_local std::unordered_map<WorkspaceKey, WorkspaceState, WorkspaceKeyHash> cache;
  WorkspaceKey key{device_id, reinterpret_cast<uintptr_t>(stream)};
  auto& ws = cache[key];

  if (ws.ptr != nullptr && ws.bytes >= required_bytes) {
    return ws.ptr;
  }

  RuntimeDeviceCheck(cudaSetDevice(device_id));
  if (ws.ptr != nullptr) {
    RuntimeDeviceCheck(cudaFreeAsync(ws.ptr, stream));
    ws.ptr = nullptr;
    ws.bytes = 0;
  }
  RuntimeDeviceCheck(cudaMallocAsync(&ws.ptr, required_bytes, stream));
  ws.bytes = required_bytes;
  return ws.ptr;
}

inline int getSMVersion(int device_id) {
  int sm_major = 0;
  int sm_minor = 0;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device_id));
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device_id));
  return sm_major * 10 + sm_minor;
}
