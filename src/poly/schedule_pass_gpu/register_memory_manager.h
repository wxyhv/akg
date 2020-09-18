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

#ifndef REGISTER_MEMORY_MANAGER_H_
#define REGISTER_MEMORY_MANAGER_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * Manager shared memory in GPU.
 */
class RegisterMemoryManager : public SchedulePass {
 public:
  explicit RegisterMemoryManager(ScopInfo &scop_info) : scop_info_(scop_info) { pass_name_ = __FUNCTION__; };
  ~RegisterMemoryManager() {}

  virtual isl::schedule Run(isl::schedule sch);

  void HoistRegisterMemoryOnDepth(isl::schedule_node &node);

  isl::union_set GatherMappingsTo(MappingCfg *cfg);

  void CreateTensorCluster(const isl::schedule_node &node, const isl::union_map &outer_sch);

  void GatherBufferFootprintDefInfo(const isl::schedule_node &node, BufferDefInfo &tensor_info);

  bool ReuseTensorCluster(const TensorFootprintCluster &cluster, const isl::multi_union_pw_aff &outer_pw_aff);

  bool IsPromote(const TensorFootprintCluster &fp_cluster, const isl::multi_union_pw_aff &partial_sched_mupa,
                 const isl::multi_union_pw_aff &thread_schedule);

  bool UnrolledLoop(const TensorFootprintCluster &fp_cluster);

  size_t UpdateDepth(const isl::schedule_node &root);

 private:
  ScopInfo &scop_info_;
  isl::schedule schedule_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif