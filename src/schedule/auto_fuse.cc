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
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/schedule_pass.h>
#include <tvm.h>

namespace akg {
namespace schedule {

class FuseOpAxis {
 public:
  explicit FuseOpAxis(const Schedule &sch) { sch_ = sch; }

  void Traverse(const Operation &op) {
    if (!op.defined() || op->IsInstance<PlaceholderOpNode>() || visited.count(op)) {
      return;
    }
    RunFuse(op);
    for (auto t : op->InputTensors()) {
      Traverse(t->op);
    }
    visited.insert(op);
  }

 private:
  void RunFuse(const Operation &op) {
    // skip op thas been inlined
    if (sch_[op]->attach_type == air::kInline) {
      return;
    }
    auto tensor = op.output(0);
    // fuse reduce axis of op
    auto compute_op = sch_[tensor]->op.as<air::BaseComputeOpNode>();
    CHECK_NOTNULL(compute_op);
    if (compute_op->reduce_axis.size() > 1) {
      IterVar fused_reduce_axis;
      sch_[tensor].fuse(compute_op->reduce_axis, &fused_reduce_axis);
      // reduce by the fused_reduce_axis
      auto data_rf = sch_.rfactor(tensor, fused_reduce_axis);
      if (data_rf.size() == 1) {
        sch_[data_rf[0]].compute_inline();
      }
    }
    // fuse axis of op
    compute_op = sch_[tensor]->op.as<air::BaseComputeOpNode>();
    CHECK_NOTNULL(compute_op);
    if (compute_op->axis.size() > 1) {
      IterVar fused_axis;
      sch_[tensor].fuse(compute_op->axis, &fused_axis);
    }
  }
  Schedule sch_;
  std::unordered_set<Operation, ExprHash, ExprEqual> visited;
};

bool NeedToFuse(const Schedule &sch) {
  for (const auto &s : sch->stages) {
    // If there is reduce, return true
    auto op = s->op;
    CHECK(op.defined());
    auto compute_op = op.as<air::BaseComputeOpNode>();
    if (compute_op && !compute_op->reduce_axis.empty()) {
      return true;
    }
  }
  return false;
}

void AutoFuse(Schedule sch) {
  if (!NeedToFuse(sch)) {
    return;
  }
  auto fuse_op_axis = FuseOpAxis(sch);
  for (auto op : sch->outputs) {
    fuse_op_axis.Traverse(op);
  }
}

}  // namespace schedule
}  // namespace akg
