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

#ifndef XLA_SERVICE_HEAP_SIMULATOR_LOOP_OPTIMIZED_ALLOCATION_H_
#define XLA_SERVICE_HEAP_SIMULATOR_LOOP_OPTIMIZED_ALLOCATION_H_

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"

namespace xla {

struct LoopOptimizerAllocation {
  int64_t id;
  // Both start_time and end_time are inclusive.
  int64_t start_time;
  int64_t end_time;
  int64_t size;
  std::string ToString() const {
    return absl::StrCat(start_time, ", ", end_time, ", ", size);
  }

  bool operator<(const LoopOptimizerAllocation& other) const {
    return id < other.id;
  }
};

}  // namespace xla

#endif  // XLA_SERVICE_HEAP_SIMULATOR_LOOP_OPTIMIZED_ALLOCATION_H_
