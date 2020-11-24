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

struct FuncIndex {
  air::ir::FunctionRef f;
  size_t arg_index;

  inline bool operator==(const FuncIndex &other) const { return f == other.f && arg_index == other.arg_index; }
  inline std::string GetStr() const {
    std::ostringstream os;
    os << f->func_name() << "_arg_" << arg_index;
    return os.str();
  }
};

namespace std {
template <>
struct hash<FuncIndex> {
  std::size_t operator()(const FuncIndex &k) const {
    size_t lhs = ::air::NodeHash()(k.f);
    size_t rhs = k.arg_index;
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
  }
};
}  // namespace std

namespace akg {
namespace schedule {

#define DEBUG_AUTO_FUSE 0

class FuseOpAxis {
 public:
  explicit FuseOpAxis(
    const Schedule &sch,
    const std::unordered_map<IterVar, std::unordered_set<size_t>, ExprHash, ExprEqual> &axis_reduce_group_ids) {
    sch_ = sch;
    axis_reduce_group_ids_ = axis_reduce_group_ids;
  }

  void Run() {
    for (auto op : sch_->outputs) {
      TraverseCheck(op);
    }
    if (!enable_fuse) {
      return;
    }
    for (auto op : sch_->outputs) {
      TraverseFuse(op);
    }
  }

  void TraverseCheck(const Operation &op) {
    if (!op.defined() || op->IsInstance<PlaceholderOpNode>() || check_visited.count(op)) {
      return;
    }
    RunCheck(op);
    for (auto t : op->InputTensors()) {
      TraverseCheck(t->op);
    }
    check_visited.insert(op);
  }

  void TraverseFuse(const Operation &op) {
    if (!op.defined() || op->IsInstance<PlaceholderOpNode>() || fuse_visited.count(op)) {
      return;
    }
    RunFuse(op);
    for (auto t : op->InputTensors()) {
      TraverseFuse(t->op);
    }
    fuse_visited.insert(op);
  }

 private:
  Schedule sch_;
  bool enable_fuse{true};
  std::unordered_set<Operation, ExprHash, ExprEqual> check_visited;
  std::unordered_set<Operation, ExprHash, ExprEqual> fuse_visited;
  std::unordered_map<IterVar, std::unordered_set<size_t>, ExprHash, ExprEqual> axis_reduce_group_ids_;

  void RunCheck(const Operation &op) {
    // skip op that has been inlined
    if (sch_[op]->attach_type == air::kInline) {
      return;
    }
    auto tensor = op.output(0);
    // reduce axis of op should be same
    auto compute_op = sch_[tensor]->op.as<air::ComputeOpNode>();
    CHECK_NOTNULL(compute_op);
    if (compute_op->reduce_axis.size() > 1) {
      if (SplitAxisToGroups(compute_op->reduce_axis).size() > 1) {
        LOG(ERROR) << "The scenes where the reduce_axis cannot be fused into one axis currently not supported."
                   << std::endl;
        enable_fuse = false;
      }
    }
  }
  void RunFuse(const Operation &op) {
    // skip op that has been inlined
    if (sch_[op]->attach_type == air::kInline) {
      return;
    }
    auto tensor = op.output(0);
    // fuse reduce axis of op
    auto compute_op = sch_[tensor]->op.as<air::ComputeOpNode>();
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
    compute_op = sch_[tensor]->op.as<air::ComputeOpNode>();
    CHECK_NOTNULL(compute_op);
    if (compute_op->axis.size() > 1) {
      auto axis_groups = SplitAxisToGroups(compute_op->axis);
      for (const auto &axis_group : axis_groups) {
        IterVar fused_axis;
        sch_[tensor].fuse(axis_group, &fused_axis);
      }
    }
  }

  std::vector<Array<IterVar>> SplitAxisToGroups(const Array<IterVar> &axis) {
    std::vector<std::unordered_set<size_t>> reduce_group_ids(axis.size());
    for (size_t i = 0; i < axis.size(); ++i) {
      if (axis_reduce_group_ids_.count(axis[i])) {
        reduce_group_ids[i] = axis_reduce_group_ids_.at(axis[i]);
      }
    }
    std::vector<size_t> split_index;
    split_index.push_back(0);
    for (size_t i = 1; i < axis.size(); ++i) {
      if (reduce_group_ids[i] != reduce_group_ids[i - 1]) {
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
      stage_id_ = i;
      if (auto compute_op = op.as<air::ComputeOpNode>()) {
        GetAxisInfo(compute_op->axis, compute_op->reduce_axis);
        VisitComputeOp(op);
      }
      if (DEBUG_AUTO_FUSE) {
        LOG(INFO) << " stage_id: " << stage_id_ << " op: " << op->func_name() << std::endl;
      }
    }
    GetAxisReduceGroup();
    if (DEBUG_AUTO_FUSE) {
      std::stringstream info;
      info << "==== axis_reduce_group_ids_ start" << std::endl;
      for (size_t i = 0; i < sch_->stages.size(); ++i) {
        auto op = sch_->stages[i]->op;
        if (auto compute_op = op.as<air::ComputeOpNode>()) {
          std::vector<IterVar> all_axis(compute_op->axis.begin(), compute_op->axis.end());
          for (auto ax : compute_op->reduce_axis) {
            all_axis.push_back(ax);
          }
          for (const auto &ax : all_axis) {
            if (axis_reduce_group_ids_.count(ax)) {
              info << compute_op->func_name() << ", " << ax << ": ";
              auto reduce_groups_ids = axis_reduce_group_ids_.at(ax);
              info << "[";
              for (auto id : reduce_groups_ids) {
                info << sch_->stages[id]->op->func_name() << ",";
              }
              info << "]" << std::endl;
            }
          }
        }
      }
      info << "==== axis_reduce_group_ids_ end" << std::endl;
      LOG(INFO) << info.str();
    }
  }

  std::unordered_map<IterVar, std::unordered_set<size_t>, ExprHash, ExprEqual> axis_reduce_group_ids_;

 private:
  Schedule sch_;
  size_t stage_id_{0};
  std::unordered_set<const Variable *> reduce_axis_var_;
  std::unordered_set<const Variable *> axis_var_;
  std::unordered_map<const Variable *, IterVar> all_axis_var_axis_;
  std::vector<FuncIndex> func_index_keys_;
  std::unordered_map<FuncIndex, std::unordered_set<IterVar, ExprHash, ExprEqual>> func_index_axis_;
  std::unordered_map<FuncIndex, std::unordered_set<size_t>> func_index_reduce_group_ids_;
  std::unordered_map<IterVar, std::unordered_set<FuncIndex>, ExprHash, ExprEqual> axis_func_indexs_;

  void VisitComputeOp(const Operation &op) {
    auto compute_op = op.as<air::ComputeOpNode>();
    auto func_dim = compute_op->axis.size();
    for (size_t i = 0; i < func_dim; ++i) {
      FuncIndex func_index = FuncIndex{op, i};
      if (!func_index_axis_.count(func_index)) {
        func_index_keys_.push_back(func_index);
      }
      func_index_axis_[func_index].insert(compute_op->axis[i]);
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
    for (size_t i = 0; i < func_dim; ++i) {
      auto func_index = FuncIndex{func, i};
      auto arg = op->args[i];
      if (auto var = arg.as<Variable>()) {
        if (reduce_axis_var_.count(var) || axis_var_.count(var)) {
          CHECK(all_axis_var_axis_.count(var));
          auto ax = all_axis_var_axis_.at(var);
          if (!func_index_axis_.count(func_index)) {
            func_index_keys_.push_back(func_index);
          }
          func_index_axis_[func_index].insert(ax);
          if (reduce_axis_var_.count(var)) {
            func_index_reduce_group_ids_[func_index].insert(stage_id_);
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

  void GetAxisReduceGroup() {
    // record map that axis to func_indexs
    for (const auto &kv : func_index_axis_) {
      auto func_index = kv.first;
      auto axis = kv.second;
      for (auto ax : axis) {
        axis_func_indexs_[ax].insert(func_index);
      }
    }
    // update reduce group by func_index
    std::unordered_map<FuncIndex, std::unordered_set<FuncIndex>> func_index_map_func_indexs;
    for (auto kv : func_index_axis_) {
      auto func_index = kv.first;
      auto axis = kv.second;
      std::unordered_set<FuncIndex> func_indexs;
      for (const auto &ax : axis) {
        auto cur_func_indexs = axis_func_indexs_.at(ax);
        func_indexs.insert(cur_func_indexs.begin(), cur_func_indexs.end());
      }
      // remove self from the map
      func_indexs.erase(func_index);
      if (!func_indexs.empty()) {
        func_index_map_func_indexs[func_index] = func_indexs;
      }
    }
    std::unordered_set<FuncIndex> last_updated;
    for (const auto &kv : func_index_reduce_group_ids_) {
      last_updated.insert(kv.first);
    }
    std::vector<FuncIndex> func_index_keys_has_map_;
    for (const auto &func_index : func_index_keys_) {
      if (func_index_map_func_indexs.count(func_index)) {
        func_index_keys_has_map_.push_back(func_index);
      }
    }
    do {
      std::unordered_set<FuncIndex> updated;
      for (const auto &func_index : func_index_keys_has_map_) {
        std::unordered_set<FuncIndex> map_func_indexs = func_index_map_func_indexs.at(func_index);
        std::unordered_set<size_t> reduce_group_ids;
        if (func_index_reduce_group_ids_.count(func_index)) {
          reduce_group_ids = func_index_reduce_group_ids_.at(func_index);
        }
        auto pre_size = reduce_group_ids.size();
        for (const auto &cur_map_func_index : map_func_indexs) {
          if (last_updated.count(cur_map_func_index)) {
            auto cur_map_reduce_group_ids = func_index_reduce_group_ids_.at(cur_map_func_index);
            reduce_group_ids.insert(cur_map_reduce_group_ids.begin(), cur_map_reduce_group_ids.end());
          }
        }
        if (reduce_group_ids.size() > pre_size) {
          func_index_reduce_group_ids_[func_index] = reduce_group_ids;
          updated.insert(func_index);
        }
      }
      last_updated = updated;
    } while (!last_updated.empty());
    // get reduce_group ids for axis
    for (const auto &kv : axis_func_indexs_) {
      auto ax = kv.first;
      auto func_indexs = kv.second;
      std::unordered_set<size_t> reduce_group_ids;
      for (const auto &func_index : func_indexs) {
        if (func_index_reduce_group_ids_.count(func_index)) {
          auto cur_reduce_group_ids = func_index_reduce_group_ids_.at(func_index);
          reduce_group_ids.insert(cur_reduce_group_ids.begin(), cur_reduce_group_ids.end());
        }
      }
      if (!reduce_group_ids.empty()) {
        axis_reduce_group_ids_[ax] = reduce_group_ids;
      }
    }
  }
};

void AutoFuse(Schedule sch) {
  if (!NeedToFuse(sch)) {
    return;
  }
  auto compute_info = ComputeInfo(sch);
  compute_info.Run();
  auto axis_reduce_group_ids = compute_info.axis_reduce_group_ids_;
  auto fuse_op_axis = FuseOpAxis(sch, axis_reduce_group_ids);
  fuse_op_axis.Run();
}
}  // namespace schedule
}  // namespace akg
