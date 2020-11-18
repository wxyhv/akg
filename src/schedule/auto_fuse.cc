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

#define DEBUG_AUTO_FUSE 0

class FuseOpAxis {
 public:
  explicit FuseOpAxis(const Schedule &sch, const std::unordered_set<IterVar, ExprHash, ExprEqual> &reduce_group) {
    sch_ = sch;
    reduce_group_ = reduce_group;
  }

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
  Schedule sch_;
  std::unordered_set<Operation, ExprHash, ExprEqual> visited;
  std::unordered_set<IterVar, ExprHash, ExprEqual> reduce_group_;

  void RunFuse(const Operation &op) {
    // skip op that has been inlined
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
      auto axis_groups = SplitAxisToGroups(compute_op->axis);
      for (auto axis_group : axis_groups) {
        IterVar fused_axis;
        sch_[tensor].fuse(axis_group, &fused_axis);
      }
    }
  }

  std::vector<Array<IterVar>> SplitAxisToGroups(const Array<IterVar> &axis) {
    std::vector<bool> reduced(axis.size(), false);
    for (size_t i = 0; i < axis.size(); ++i) {
      if (reduce_group_.count(axis[i])) {
        reduced[i] = true;
      }
    }
    std::vector<size_t> split_index;
    split_index.push_back(0);
    for (size_t i = 1; i < axis.size(); ++i) {
      if (reduced[i] != reduced[i - 1]) {
        split_index.push_back(i);
      }
    }
    split_index.push_back(axis.size());
    std::vector<Array<IterVar>> res;
    for (size_t i = 0; i < split_index.size() - 1; ++i) {
      Array<IterVar> cur_axis_group;
      for (auto j = split_index[i]; j < split_index[i + 1]; ++j) {
        cur_axis_group.push_back(axis[j]);
      }
      res.push_back(cur_axis_group);
    }
    return res;
  }
};

bool IsMatmul(const Operation &op) {
  if (op->tag == "dense" || op->tag == "batch_matmul" || op->tag == "matmul") {
    return true;
  }

  auto compute_op = op.as<ComputeOpNode>();
  auto reduce = compute_op->body[0].as<Reduce>();
  CHECK_NOTNULL(reduce);
  // combiner should be `lhs + rhs`
  auto combiner = reduce->combiner;
  if (combiner->lhs.size() != 1 || combiner->rhs.size() != 1 || combiner->result.size() != 1 ||
      !combiner->result[0].as<Add>()) {
    return false;
  }
  // size of reduce_axis should be 1
  auto reduce_axis = reduce->axis;
  if (reduce_axis.size() != 1) {
    return false;
  }
  // source should be such as: left[..., i, k] * right[..., j, k]
  auto source = reduce->source;
  if (source.size() != 1 || !source[0].as<Mul>()) {
    return false;
  }
  auto mul = source[0].as<Mul>();
  auto left = mul->a.as<Call>();
  auto right = mul->b.as<Call>();
  if (!left || !right || left->args.size() != right->args.size()) {
    return false;
  }
  auto args_size = left->args.size();
  if (args_size < 2) {
    return false;
  }
  for (size_t i = 0; i < args_size - 2; ++i) {
    if (!left->args[i].same_as(right->args[i])) {
      return false;
    }
  }
  auto reduce_var = reduce_axis[0]->var.get();
  if ((left->args[args_size - 1].as<Variable>() != reduce_var &&
       left->args[args_size - 2].as<Variable>() != reduce_var) ||
      (right->args[args_size - 1].as<Variable>() != reduce_var &&
       right->args[args_size - 2].as<Variable>() != reduce_var)) {
    return false;
  }
  return true;
}

bool NeedToFuse(const Schedule &sch) {
  bool has_reduce = false;
  bool has_matmul = false;
  for (const auto &s : sch->stages) {
    // If there is reduce, return true
    auto op = s->op;
    CHECK(op.defined());
    auto compute_op = op.as<air::ComputeOpNode>();
    if (compute_op && !compute_op->reduce_axis.empty()) {
      has_reduce = true;
      if (IsMatmul(op)) {
        has_matmul = true;
      }
    }
  }
  return has_reduce && !has_matmul;
}

class ComputeInfo : public IRVisitor {
 public:
  explicit ComputeInfo(const Schedule &sch) { sch_ = sch; }

  void Run() {
    for (size_t i = 0; i < sch_->stages.size(); ++i) {
      auto op = sch_->stages[i]->op;
      if (auto compute_op = op.as<air::ComputeOpNode>()) {
        GetAxisInfo(compute_op->axis, compute_op->reduce_axis);
        VisitComputeOp(op);
      }
      if (DEBUG_AUTO_FUSE) {
        LOG(INFO) << " stage_id: " << i << " op: " << op->func_name() << std::endl;
      }
    }
    UpdateReduceGroup();
    if (DEBUG_AUTO_FUSE) {
      std::stringstream info;
      info << "==== reduce_group_ start" << std::endl;
      for (auto ax : reduce_group_) {
        info << ax << "(" << ax.get() << ")" << std::endl;
      }
      info << "==== reduce_group_ end" << std::endl;
      LOG(INFO) << info.str();
    }
  }

  std::unordered_set<IterVar, ExprHash, ExprEqual> reduce_group_;

 private:
  Schedule sch_;
  std::unordered_set<const Variable *> reduce_axis_var_;
  std::unordered_set<const Variable *> axis_var_;
  std::unordered_map<const Variable *, IterVar> all_axis_var_axis_;
  std::vector<FunctionRef> func_keys_;
  std::unordered_map<FunctionRef, std::vector<std::unordered_set<IterVar, ExprHash, ExprEqual>>, ExprHash, ExprEqual>
    func_index_axis_;

  void VisitComputeOp(const Operation &op) {
    CHECK(!func_index_axis_.count(op));
    func_keys_.push_back(op);
    auto compute_op = op.as<air::ComputeOpNode>();
    auto func_dim = compute_op->axis.size();
    func_index_axis_[op] = std::vector<std::unordered_set<IterVar, ExprHash, ExprEqual>>(func_dim);
    for (size_t i = 0; i < func_dim; ++i) {
      func_index_axis_[op][i].insert(compute_op->axis[i]);
    }
    for (auto expr : compute_op->body) {
      Visit(expr);
    }
  }

  void Visit_(const Call *op) override {
    auto func = op->func;
    if (!(func.defined() && func.as<OperationNode>())) {
      return IRVisitor::Visit_(op);
    }
    auto func_dim = op->args.size();
    if (!func_index_axis_.count(func)) {
      func_keys_.push_back(func);
      func_index_axis_[func] = std::vector<std::unordered_set<IterVar, ExprHash, ExprEqual>>(func_dim);
    }
    for (size_t i = 0; i < func_dim; ++i) {
      auto arg = op->args[i];
      if (auto var = arg.as<Variable>()) {
        if (reduce_axis_var_.count(var) || axis_var_.count(var)) {
          CHECK(all_axis_var_axis_.count(var));
          auto ax = all_axis_var_axis_.at(var);
          func_index_axis_[func][i].insert(ax);
          if (reduce_axis_var_.count(var)) {
            reduce_group_.insert(ax);
          }
        }
      }
    }
  }

  void GetAxisInfo(const Array<IterVar> &axis, const Array<IterVar> &reduce_axis) {
    axis_var_.clear();
    reduce_axis_var_.clear();
    all_axis_var_axis_.clear();
    for (auto ax : axis) {
      auto var = ax->var.get();
      axis_var_.insert(var);
      CHECK_EQ(all_axis_var_axis_.count(var), 0);
      all_axis_var_axis_[var] = ax;
    }
    for (auto ax : reduce_axis) {
      auto var = ax->var.get();
      reduce_axis_var_.insert(var);
      CHECK_EQ(all_axis_var_axis_.count(var), 0);
      all_axis_var_axis_[var] = ax;
    }
  }

  void UpdateReduceGroup() {
    size_t group_size;
    std::unordered_map<FunctionRef, std::vector<bool>, ExprHash, ExprEqual> visited;
    for (auto func : func_keys_) {
      visited[func] = std::vector<bool>(func_index_axis_.at(func).size(), false);
    }
    do {
      group_size = reduce_group_.size();
      for (auto func : func_keys_) {
        CHECK(func_index_axis_.count(func));
        auto index_axis = func_index_axis_.at(func);
        for (size_t i = 0; i < index_axis.size(); ++i) {
          if (visited[func][i]) {
            continue;
          }
          auto axis = index_axis[i];
          bool has_in_reduce_group = false;
          for (auto ax : axis) {
            if (reduce_group_.count(ax)) {
              has_in_reduce_group = true;
              break;
            }
          }
          if (!has_in_reduce_group) {
            continue;
          }
          for (auto ax : axis) {
            reduce_group_.insert(ax);
          }
          visited[func][i] = true;
        }
      }
    } while (reduce_group_.size() > group_size);
  }
};

void AutoFuse(Schedule sch) {
  if (!NeedToFuse(sch)) {
    return;
  }
  auto compute_info = ComputeInfo(sch);
  compute_info.Run();
  auto reduce_group = compute_info.reduce_group_;
  auto fuse_op_axis = FuseOpAxis(sch, reduce_group);
  for (auto op : sch->outputs) {
    fuse_op_axis.Traverse(op);
  }
}
}  // namespace schedule
}  // namespace akg
