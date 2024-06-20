/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/shape_util.h"

namespace xla {
namespace memory_space_assignment {

enum class MemoryTransferDirection {
  kUnsupported,
  kDefaultToAlternate,
  kAlternateToDefault,
};

// A wrapper class around runtime simulator.
class RuntimeSimulator {
 public:
  explicit RuntimeSimulator(CostAnalysis* cost_analysis)
      : cost_analysis_(cost_analysis) {}
  virtual ~RuntimeSimulator() = default;
  // This function is used to predict the effectiveness of the memory space
  // assignment solution. Specifically, it returns the estimated execution time
  // (in seconds) of the HLO module for the given memory space assignment (i.e.,
  // ```allocations```).
  // This function provides a basic estimate without considering the effect of
  // async copies.
  float SimulateElapsedTimeWithoutAsyncCopies(
      const HloLiveRange& hlo_live_range,
      const AllocationSequence& allocations);

  // This function, compared with SimulateElapsedTimeWithoutAsyncCopies
  // function, provides a more accurate estimated execution time, as it
  // simulates the default memory communication to estimate the overhead of
  // async copies.
  // To simulate the overhead of async copies, we need to maintain two queues to
  // track the memory access requests that read/write the default memory.
  // Specifically, for any async copy instruction (copy-start), we push it to
  // the corresponding queue. When the copy-done instruction is executed, we pop
  // it (and all the previous copy-start instructions) from the queue and
  // calculate the execution time of the async copy. Instead of the copy-done
  // instruction, we also try to drain the queue during computation instruction,
  // in which default memory is not accessed. The most important feature for the
  // memory model is sharing the bandwidth. Specifically, if both queues are not
  // empty, we can only use half of the bandwidth to transfer the request for
  // each of them in parallel.
  float SimulateElapsedTime(const HloModule* hlo_module,
                            const HloLiveRange& hlo_live_range,
                            const AllocationSequence& allocations,
                            int64_t alternate_memory_space,
                            float default_memory_bytes_per_second);

  // This is an auxiliary function which is used for simulating the execution
  // time of async copies. This function simulates the execution time of
  // transferring ```bytes_to_transfer``` bytes while sharing the bandwidth with
  // memory access requests in ```memory_access_queue_to_share_bandwidth```. The
  // bandwidth is shared equally: When the
  // memory_access_queue_to_share_bandwidth is not empty, we can only use half
  // of the bandwidth to transfer the request, and use the other half to
  // transfer the memory requests in the queue. When the queue is drained, we
  // can use the full bandwidth to transfer the request.
  static float SimulateAsyncCopyTransfer(
      float bytes_to_transfer,
      std::queue<const HloInstruction*>& memory_access_queue_to_share_bandwidth,
      absl::flat_hash_map<const HloInstruction*, float>&
          remaining_size_of_buffers,
      float default_memory_bytes_per_second);

  // This is an auxiliary function which simulates the process of draining the
  // memory access queue in a given time window. There are two queues which will
  // share the bandwidth: ```read_queue``` and ```write_queue``` which track the
  // memory access requests that read/write the default memory. When both of the
  // queues are not empty, the front requests from both queues equally share the
  // bandwidth. When one of the queue is empty, the other queue can use the full
  // bandwidth.

  static void ProcessAsyncCopyInTimeWindow(
      float time_windows, std::queue<const HloInstruction*>& read_queue,
      std::queue<const HloInstruction*>& write_queue,
      absl::flat_hash_map<const HloInstruction*, float>&
          remaining_size_of_buffers,
      float default_memory_bytes_per_second);

 private:
  // This function parses the memory space assignment solution and initializes
  // the maps that record, for each instruction, which outputs and operands are
  // stored in alternate memory. These maps are used to estimate the runtime of
  // the HLO module.
  void InitializeAlternateMemoryMap(const AllocationSequence& allocations);

  MemoryTransferDirection GetAsyncCopyDirection(
      const HloInstruction* async_copy, int64_t kAlternateMemorySpace);

  const CostAnalysis* cost_analysis_;
  CostAnalysis::Cache cost_analysis_cache_;
  absl::flat_hash_map<const HloInstruction*, std::vector<ShapeIndex>>
      outputs_in_alternate_memory_map_;
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<std::pair<int64_t, ShapeIndex>>>
      operands_in_alternate_memory_map_;
};
}  // namespace memory_space_assignment
}  // namespace xla
#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_
