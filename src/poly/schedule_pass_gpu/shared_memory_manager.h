/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SHARED_MEMORY_MANAGER_H_
#define SHARED_MEMORY_MANAGER_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

using TensorClusters = std::pair<isl::id, std::vector<std::shared_ptr<TensorFootprintCluster>>>;

/*
 * Manager shared memory in GPU.
 */
class SharedMemoryManager : public SchedulePass {
 public:
  explicit SharedMemoryManager(ScopInfo &scop_info) : scop_info_(scop_info) {
    pass_name_ = __FUNCTION__;
    // use 48KB in current GPU
    share_memory_size_ = 49152;
    if (!scop_info.user_config_.GetSharedTensors().empty()) {
      configed_tensors_ = Split(scop_info.user_config_.GetSharedTensors(), " ");
    }
    unroll_copies_ = false;
  };
  ~SharedMemoryManager() {}

  virtual isl::schedule Run(isl::schedule sch);

  isl::schedule_node HoistSharedMemoryOnDepth(const isl::schedule_node &root, size_t &remain_memory, size_t depth);

  isl::union_set GatherMappingsTo(MappingCfg *cfg);

  isl::schedule_node MapCopiesToThreads(isl::schedule_node &root, bool unroll);

  isl::schedule_node ManageToShareBelow(isl::schedule &root, isl::schedule_node &node, size_t &remaining_memory);

  void CreateClusterList(const isl::schedule_node &node, const isl::union_map &outer_sch);

  void GatherBufferFootprintDefInfo(const isl::schedule_node &node, BufferDefInfo &tensor_info);

  isl::schedule_node HoistClusters(const isl::schedule_node &root, const isl::schedule_node &node,
                                   size_t &remaining_memory);

  isl::schedule_node HoistToBlockThreadMemory(isl::schedule_node &tree, GpuMemType type, const isl::id &tensor_id,
                                              TensorFootprintCluster &cluster, bool force_last_extension_odd);

  bool ReuseTensorCluster(const TensorFootprintCluster &cluster, const isl::multi_union_pw_aff &outer_pw_aff);

  bool CoalescingAccessWay(const isl::schedule_node &root, const isl::schedule_node &node,
                           const TensorFootprintCluster &cluster);

  void UpdateDepth(const isl::schedule_node &root);

  bool UnderThreadMarker(size_t depth);

  std::string InAtomicTensors(isl::schedule_node &node);
  bool InAtomicTensors(std::string name);
  bool InReduceTensors(std::string name);

  std::string AtomicMarker(std::string type);

  std::set<std::string> AnalysisReduceTensors();

 private:
  ScopInfo &scop_info_;
  isl::schedule schedule_;
  size_t share_memory_size_;
  int depth_{1};
  bool use_config_{false};
  std::vector<std::string> configed_tensors_;
  bool unroll_copies_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif