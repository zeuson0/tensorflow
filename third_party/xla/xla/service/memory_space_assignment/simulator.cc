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

#include "xla/service/memory_space_assignment/simulator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/layout.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace memory_space_assignment {

void RuntimeSimulator::InitializeAlternateMemoryMap(
    const AllocationSequence& allocations) {
  outputs_in_alternate_memory_map_.clear();
  operands_in_alternate_memory_map_.clear();
  for (auto& allocation : allocations) {
    if (!allocation->is_copy_allocation()) {
      if (allocation->memory_space() == MemorySpace::kAlternate) {
        const HloInstruction* defining_instruction =
            allocation->defining_position().instruction;
        outputs_in_alternate_memory_map_[defining_instruction].push_back(
            allocation->defining_position().index);
      }
    }
    for (auto& hlo_use : allocation->uses()) {
      const HloInstruction* use_instruction = hlo_use.instruction;
      operands_in_alternate_memory_map_[use_instruction].push_back(
          std::make_pair(hlo_use.operand_number, hlo_use.operand_index));
    }
  }
}

MemoryTransferDirection RuntimeSimulator::GetAsyncCopyDirection(
    const HloInstruction* async_copy, int64_t alternate_memory_space) {
  CHECK_EQ(async_copy->opcode(), HloOpcode::kCopyStart);

  int64_t operand_memory_space =
      async_copy->operand(0)->shape().layout().memory_space();

  // Get all users
  std::optional<int64_t> output_memory_space;
  for (const HloInstruction* user : async_copy->users()) {
    if (user->opcode() == HloOpcode::kCopyDone) {
      output_memory_space.emplace(user->shape().layout().memory_space());
      break;
    }
  }
  if (!output_memory_space.has_value()) {
    return MemoryTransferDirection::kUnsupported;
  }

  if (operand_memory_space == xla::Layout::kDefaultMemorySpace &&
      output_memory_space == alternate_memory_space) {
    return MemoryTransferDirection::kDefaultToAlternate;
  }
  if (operand_memory_space == alternate_memory_space &&
      output_memory_space == xla::Layout::kDefaultMemorySpace) {
    return MemoryTransferDirection::kAlternateToDefault;
  }
  return MemoryTransferDirection::kUnsupported;
}

float RuntimeSimulator::SimulateElapsedTimeWithoutAsyncCopies(
    const HloLiveRange& hlo_live_range, const AllocationSequence& allocations) {
  InitializeAlternateMemoryMap(allocations);
  const auto& instruction_sequence =
      hlo_live_range.flattened_instruction_sequence().instructions();
  float total_elapsed = 0.0;
  for (const HloInstruction* instruction : instruction_sequence) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      continue;
    }

    auto output_it = outputs_in_alternate_memory_map_.find(instruction);
    const std::vector<ShapeIndex>& outputs_in_alternate_memory =
        (output_it != outputs_in_alternate_memory_map_.end())
            ? output_it->second
            : std::vector<ShapeIndex>();

    auto operand_it = operands_in_alternate_memory_map_.find(instruction);
    const std::vector<std::pair<int64_t, ShapeIndex>>&
        operands_in_alternate_memory =
            (operand_it != operands_in_alternate_memory_map_.end())
                ? operand_it->second
                : std::vector<std::pair<int64_t, ShapeIndex>>();

    float instruction_elapsed_per_invoke =
        cost_analysis_->GetInstructionElapsedInAlternateMemory(
            *instruction, operands_in_alternate_memory,
            outputs_in_alternate_memory);
    float total_trip_count = cost_analysis_->CalculateNestTripCount(
        instruction, &cost_analysis_cache_);
    // Calculate total elapsed time by summing up the overall elapsed time of
    // each instruction.
    total_elapsed += total_trip_count * instruction_elapsed_per_invoke;
  }
  return total_elapsed;
}

float RuntimeSimulator::SimulateAsyncCopyTransfer(
    float bytes_to_transfer,
    std::queue<const HloInstruction*>& memory_access_queue_to_share_bandwidth,
    absl::flat_hash_map<const HloInstruction*, float>&
        remaining_size_of_buffers,
    float default_memory_bytes_per_second) {
  float remaining_bytes = bytes_to_transfer;
  float elapsed_time = 0.0;
  while (!memory_access_queue_to_share_bandwidth.empty() &&
         remaining_bytes > 0) {
    const HloInstruction* front_async_copy =
        memory_access_queue_to_share_bandwidth.front();
    float smaller_buffer_size = std::min(
        remaining_bytes, remaining_size_of_buffers.at(front_async_copy));
    // The bandwidth is shared, so the request can only use half of the
    // bandwidth.
    elapsed_time +=
        smaller_buffer_size / (0.5 * default_memory_bytes_per_second);
    remaining_bytes -= smaller_buffer_size;
    remaining_size_of_buffers.at(front_async_copy) -= smaller_buffer_size;
    if (remaining_size_of_buffers.at(front_async_copy) <= 0) {
      remaining_size_of_buffers.erase(front_async_copy);
      memory_access_queue_to_share_bandwidth.pop();
    }
  }
  if (remaining_bytes > 0) {
    // The queue that shares the bandwidth is drained, we can now use the full
    // bandwidth.
    elapsed_time += remaining_bytes / default_memory_bytes_per_second;
  }
  return elapsed_time;
};

void RuntimeSimulator::ProcessAsyncCopyInTimeWindow(
    float time_windows, std::queue<const HloInstruction*>& read_queue,
    std::queue<const HloInstruction*>& write_queue,
    absl::flat_hash_map<const HloInstruction*, float>&
        remaining_size_of_buffers,
    float default_memory_bytes_per_second) {
  float elapsed_time = time_windows;
  while (!read_queue.empty() || !write_queue.empty()) {
    if (elapsed_time <= 0) {
      // Run out of time, return
      return;
    }
    if (!read_queue.empty() && !write_queue.empty()) {
      // Both queues are not empty, share the bandwidth between them.
      const HloInstruction* front_read_default_async_copy = read_queue.front();
      const HloInstruction* front_write_default_async_copy =
          write_queue.front();
      float smaller_buffer_size = std::min(
          remaining_size_of_buffers.at(front_read_default_async_copy),
          remaining_size_of_buffers.at(front_write_default_async_copy));
      float required_time =
          smaller_buffer_size / (0.5 * default_memory_bytes_per_second);
      if (required_time > elapsed_time) {
        // The required time is larger than the remaining
        // computation time, use the remaining computation time as
        // the required time to transfer a part of the buffer.
        required_time = elapsed_time;
        smaller_buffer_size =
            required_time * (0.5 * default_memory_bytes_per_second);
      }
      elapsed_time -= required_time;
      remaining_size_of_buffers.at(front_read_default_async_copy) -=
          smaller_buffer_size;
      remaining_size_of_buffers.at(front_write_default_async_copy) -=
          smaller_buffer_size;
      if (remaining_size_of_buffers.at(front_read_default_async_copy) <= 0) {
        remaining_size_of_buffers.erase(front_read_default_async_copy);
        read_queue.pop();
      }
      if (remaining_size_of_buffers.at(front_write_default_async_copy) <= 0) {
        remaining_size_of_buffers.erase(front_write_default_async_copy);
        write_queue.pop();
      }
    } else {
      // One of the queue is not empty, execute the async copy from
      // that queue with full bandwidth.
      std::queue<const HloInstruction*>& queue =
          read_queue.empty() ? write_queue : read_queue;
      const HloInstruction* front_async_copy = queue.front();
      float required_time = remaining_size_of_buffers.at(front_async_copy) /
                            default_memory_bytes_per_second;
      if (required_time > elapsed_time) {
        required_time = elapsed_time;
      }
      elapsed_time -= required_time;
      remaining_size_of_buffers.at(front_async_copy) -=
          required_time * default_memory_bytes_per_second;
      if (remaining_size_of_buffers.at(front_async_copy) <= 0) {
        remaining_size_of_buffers.erase(queue.front());
        queue.pop();
      }
    }
  }
}

float RuntimeSimulator::SimulateElapsedTime(
    const HloModule* hlo_module, const HloLiveRange& hlo_live_range,
    const AllocationSequence& allocations, int64_t alternate_memory_space,
    float default_memory_bytes_per_second) {
  InitializeAlternateMemoryMap(allocations);

  // Cannot provide a valid result if the bandwidth is invalid.
  if (default_memory_bytes_per_second <= 0.0) {
    return 0.0;
  }

  float total_elapsed = 0.0;
  for (const HloComputation* computation :
       hlo_module->MakeNonfusionComputations()) {
    CHECK(
        hlo_module->has_schedule() &&
        hlo_module->schedule().sequences().contains(computation->unique_id()));
    // Use two queues to track read-from-default and write-to-default-memory
    // async copies.
    std::queue<const HloInstruction*> issued_read_default_instructions;
    std::queue<const HloInstruction*> issued_write_default_instructions;

    // Used to track the bytes remaining in asynchronous copies. (Completed
    // copies are removed.)
    absl::flat_hash_map<const HloInstruction*, float> remaining_size_of_buffers;

    const HloInstructionSequence& instruction_sequence =
        hlo_module->schedule().sequence(computation);
    for (const HloInstruction* instruction :
         instruction_sequence.instructions()) {
      float inst_elapsed = 0.0;
      if (instruction->opcode() == HloOpcode::kWhile) {
        // Since the instructions in the while body are calculated
        // separately, we can skip the while instruction.
        continue;
      }
      if (instruction->opcode() == HloOpcode::kCopyStart) {
        // Try to categorize the async copy instruction into
        // read-from-default and write-to-default queues.
        MemoryTransferDirection direction =
            GetAsyncCopyDirection(instruction, alternate_memory_space);
        if (direction == MemoryTransferDirection::kDefaultToAlternate) {
          issued_read_default_instructions.push(instruction);
        } else if (direction == MemoryTransferDirection::kAlternateToDefault) {
          issued_write_default_instructions.push(instruction);
        } else {
          // The async copy instruction is not related to default memory.
          continue;
        }
        remaining_size_of_buffers.insert(
            {instruction, cost_analysis_->base_costs().GetShapeSize(
                              instruction->operand(0)->shape())});
      } else if (instruction->opcode() == HloOpcode::kCopyDone) {
        const HloInstruction* copy_start_instruction = instruction->operand(0);
        MemoryTransferDirection direction = GetAsyncCopyDirection(
            copy_start_instruction, alternate_memory_space);
        // Check how many bytes are required to be transferred for the
        // async copy instruction. This include the bytes of the corresponding
        // copy-start instruction and the instructions issued before it.

        std::queue<const HloInstruction*>* same_direction_queue = nullptr;
        std::queue<const HloInstruction*>* opposite_direction_queue = nullptr;
        if (direction == MemoryTransferDirection::kDefaultToAlternate) {
          same_direction_queue = &issued_read_default_instructions;
          opposite_direction_queue = &issued_write_default_instructions;
        } else if (direction == MemoryTransferDirection::kAlternateToDefault) {
          same_direction_queue = &issued_write_default_instructions;
          opposite_direction_queue = &issued_read_default_instructions;
        } else {
          // The async copy instruction is not related to default memory.
          continue;
        }
        if (same_direction_queue && opposite_direction_queue) {
          float total_bytes_to_transfer = 0;
          while (remaining_size_of_buffers.contains(copy_start_instruction)) {
            total_bytes_to_transfer +=
                remaining_size_of_buffers.at(same_direction_queue->front());
            remaining_size_of_buffers.erase(same_direction_queue->front());
            same_direction_queue->pop();
          }
          // Simulate the process of accessing total_bytes_to_transfer bytes
          // while sharing the bandwidth with the other queue.
          float copy_done_elapsed = SimulateAsyncCopyTransfer(
              total_bytes_to_transfer, *opposite_direction_queue,
              remaining_size_of_buffers, default_memory_bytes_per_second);
          inst_elapsed = copy_done_elapsed;
        }
      } else {
        // This branch is for the compute instructions.
        // TODO(b/351913186): Plan to add another branch to handle async
        // copy instructions caused by slicing.
        auto output_it = outputs_in_alternate_memory_map_.find(instruction);
        const std::vector<ShapeIndex>& outputs_in_alternate_memory =
            (output_it != outputs_in_alternate_memory_map_.end())
                ? output_it->second
                : std::vector<ShapeIndex>();
        auto operand_it = operands_in_alternate_memory_map_.find(instruction);
        const std::vector<std::pair<int64_t, ShapeIndex>>&
            operands_in_alternate_memory =
                (operand_it != operands_in_alternate_memory_map_.end())
                    ? operand_it->second
                    : std::vector<std::pair<int64_t, ShapeIndex>>();

        // Although TPU chip applies pipelining, for simplicity, we assume the
        // elapsed time of the compute instruction is
        // max(default_memory_access_time, alternate_memory_access_time,
        // computation_time). Specifically, if alternate memory access time or
        // computation time is larger than the default memory access time, it
        // means there are a time window that the default memory bandwidth is
        // not used by the compute instruction.. Thus, we can use this time
        // window (DefaultMemoryBandwidthIdleTime) to execute the async copy
        // instructions in the queues.

        // Instead of the above time window, the memory request queues are also
        // processed during the default memory access time. Specifically, we
        // assume the default memory access time contains two sequential phases:
        // load operands and store outputs. Thus, we apply the following
        // simulation:

        // 1) During the DefaultMemoryBandwidthIdleTime period, we try to drain
        // the memory request from both read/write queues as much as possible.
        // If both queues are not empty, they have to share the bandwidth; 2)
        // During the load operands period, the operands are loaded from default
        // memory. At the same time, if there are memory request in the write
        // queue, the request will share the bandwidth with loading operand
        // process. 3) During the store outputs period, the outputs are stored
        // to default memory. At the same time, if there are memory request in
        // the read-default queue, the request will share the bandwidth with
        // storing output process.

        // Load operand period:
        float elapsed_time_for_loading_operands = SimulateAsyncCopyTransfer(
            cost_analysis_->GetBytesAccessedFromDefaultMemory(
                *instruction, operands_in_alternate_memory,
                /*outputs_in_alternate_mem=*/{},
                /*include_operand_access=*/true,
                /*include_output_access=*/false),
            issued_write_default_instructions, remaining_size_of_buffers,
            default_memory_bytes_per_second);
        inst_elapsed += elapsed_time_for_loading_operands;

        // Memory bandwidth idle period:
        float no_default_access_elapsed_time =
            cost_analysis_->GetDefaultMemoryBandwidthIdleTime(
                *instruction, operands_in_alternate_memory,
                outputs_in_alternate_memory);

        inst_elapsed += no_default_access_elapsed_time;

        ProcessAsyncCopyInTimeWindow(
            no_default_access_elapsed_time, issued_read_default_instructions,
            issued_write_default_instructions, remaining_size_of_buffers,
            default_memory_bytes_per_second);

        // Store output period:
        float elapsed_time_for_storing_outputs = SimulateAsyncCopyTransfer(
            cost_analysis_->GetBytesAccessedFromDefaultMemory(
                *instruction, /*operands_in_alternate_mem=*/{},
                outputs_in_alternate_memory,
                /*include_operand_access=*/false,
                /*include_output_access=*/true),
            issued_read_default_instructions, remaining_size_of_buffers,
            default_memory_bytes_per_second);
        inst_elapsed += elapsed_time_for_storing_outputs;
      }
      if (inst_elapsed > 0) {
        float total_trip_count = cost_analysis_->CalculateNestTripCount(
            instruction, &cost_analysis_cache_);
        total_elapsed += inst_elapsed * total_trip_count;
      }
    }
  }
  return total_elapsed;
}

}  // namespace memory_space_assignment
}  // namespace xla
