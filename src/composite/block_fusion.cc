/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <tvm/api_registry.h>
#include "ir_pass.h"

namespace akg {
namespace ir {
const std::string BLOCKIDX = "blockIdx.";
const std::string THREADIDX = "threadIdx.";
const std::string BLOCKIDXX = "blockIdx.x";
const std::string THREADIDXX = "threadIdx.x";
const std::string THREADEXTENT = "thread_extent";
constexpr int BLOCKIDX_LEN = 9;
constexpr int THREADX_LEN = 10;
constexpr int BLOCKIDXX_LEN = 10;
constexpr int THREADXX_LEN = 11;
struct FuncInfo {
  Stmt stmt;
  Var block;
  std::string origin_block_name;
  Expr block_ext = make_const(Int(32), 1);
  Var thread;
  std::string origin_thread_name;
  Expr thread_ext = make_const(Int(32), 1);
};

bool IsVarDefault(const Var &var) { return var->name_hint == "v"; }

int RegularizeOffset(int offset) {
  int base = 16;
  return (offset + base - 1) / base * base;
}

class SharedMemoryManager : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    if (op->attr_key == "storage_scope" && op->value.as<StringImm>()->value == "shared") {
      const Variable* v = op->node.as<Variable>();
      shared_memory_set_.insert(v);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    const Variable* buffer = op->buffer_var.get();
    if (shared_memory_set_.count(buffer) != 0) {
      // Add attribute to shared memory offset.
      Expr offset_expr = IntImm::make(Int(32), total_sm_size_);
      total_sm_size_ += op->type.bytes() * op->constant_allocation_size();
      total_sm_size_ = RegularizeOffset(total_sm_size_);
      return AttrStmt::make(op->buffer_var, "shared_memory_offset", offset_expr, stmt);
    }

    return stmt;
  }

  int GetTotalSMSize() { return total_sm_size_; }

 private:
  std::set<const Variable *> shared_memory_set_;
  int total_sm_size_{0};
};

class DimCompressor : public IRMutator {
 public:
  Stmt Run(const Stmt &s) {
    is_collect_ = true;
    Stmt st = Mutate(s);
    is_collect_ = false;
    return Mutate(st);
  }
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    if (op->attr_key == THREADEXTENT) {
      const IterVarNode *iv = op->node.as<IterVarNode>();
      CHECK(iv);
      Expr extent;
      std::string name = iv->var->name_hint;
      bool is_left = false;
      if (name.compare(0, BLOCKIDX_LEN, BLOCKIDX) == 0) {
        if (is_collect_) {
          block_idx_.emplace_back(std::make_pair(iv->var, op->value));
        } else {
          is_left = LeftIdx(iv->var);
          extent = CompressIdx(block_idx_);
        }
      } else {
        CHECK_EQ(name.compare(0, THREADX_LEN, THREADIDX), 0);
        if (is_collect_) {
          thread_idx_.emplace_back(std::make_pair(iv->var, op->value));
        } else {
          is_left = LeftIdx(iv->var);
          extent = CompressIdx(thread_idx_);
        }
      }
      if (!is_collect_ && is_left) {
        Stmt body = IRMutator::Mutate(op->body);
        if (!extent.defined()) {
          return body;
        } else {
          return AttrStmt::make(op->node, op->attr_key, extent, body);
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable *op, const Expr &e) {
    if (!is_collect_) {
      auto it = replace_.find(op);
      return it == replace_.end() ? e : it->second;
    } else {
      return e;
    }
  }

  Var LeftBlock() {
    if (block_idx_.empty()) {
      return Var();
    }
    return block_idx_.back().first;
  }

  Var LeftThread() {
    if (thread_idx_.empty()) {
      return Var();
    }
    return thread_idx_.back().first;
  }

 private:
  bool LeftIdx(const Var &var) {
    bool is_left = false;
    if (!block_idx_.empty()) {
      Var bidx = block_idx_.back().first;
      is_left = is_left || bidx.get() == var.get();
    }

    if (!thread_idx_.empty()) {
      Var tidx = thread_idx_.back().first;
      is_left = is_left || tidx.get() == var.get();
    }
    return is_left;
  }

  Expr CompressIdx(const std::vector<std::pair<Var, Expr>> &idx) {
    CHECK(!idx.empty()) << "idx size must be greater than 0!";
    // expected idx order: z, x, y
    Var x = idx.back().first;
    Expr dx = idx.back().second;
    size_t idx_len = idx.size();
    if (idx_len == 1) {
      return idx[0].second;
    } else if (idx_len == 2) {
      replace_.emplace(idx[0].first.get(), x / dx);
      replace_.emplace(idx[1].first.get(), truncmod(x, dx));
      return Simplify(idx[0].second * dx);
    } else {
      CHECK_EQ(idx_len, 3);
      Expr dxy = Simplify(idx[1].second * idx[2].second);
      replace_.emplace(idx[0].first.get(), x / dxy);
      replace_.emplace(idx[1].first.get(), truncmod(x, dxy) / dx);
      replace_.emplace(idx[2].first.get(), truncmod(x, dx));
      return Simplify(dxy * idx[0].second);
    }
  }

  std::vector<std::pair<Var, Expr>> block_idx_;
  std::vector<std::pair<Var, Expr>> thread_idx_;
  std::unordered_map<const Variable *, Expr> replace_;
  bool is_collect_;
};

class DimInfoVisitor : public IRVisitor {
 public:
  DimInfoVisitor(FuncInfo &info, const Var &block_var, const Var &thread_var) : info_(info) {
    if (!IsVarDefault(block_var)) {
      block_name_ = block_var->name_hint;
    }
    if (!IsVarDefault(thread_var)) {
      thread_name_ = thread_var->name_hint;
    }
  }

  void Visit_(const AttrStmt *op) {
    if (op->attr_key == THREADEXTENT) {
      const IterVarNode *iv = op->node.as<IterVarNode>();
      CHECK(iv);
      std::string name = iv->var->name_hint;
      if (name.compare(block_name_) == 0) {
        info_.block_ext = op->value;
      } else if (name.compare(thread_name_) == 0) {
        info_.thread_ext = op->value;
      }
    }
    IRVisitor::Visit_(op);
  }
  FuncInfo &info_;

 private:
  std::string block_name_{BLOCKIDXX};
  std::string thread_name_{THREADIDXX};
};
class RemoveDimAttr : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    if (op->attr_key == THREADEXTENT) {
      return IRMutator::Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }
};

class BlockIndexRewrite final : public IRMutator {
 public:
  explicit BlockIndexRewrite(int offset) : offset_(offset) {}
  ~BlockIndexRewrite() override = default;
  Expr Mutate_(const Variable *op, const Expr &e) {
    if (op->name_hint == BLOCKIDXX && offset_ != 0) {
      return Sub::make(e, Expr(offset_));
    }
    return e;
  }
  int offset_;
};

class LowerBlockFusion {
 public:
  explicit LowerBlockFusion(const std::vector<Stmt> &funcs) {
    funcs_.resize(funcs.size());
    for (size_t i = 0; i < funcs.size(); ++i) {
      funcs_[i].stmt = funcs[i];
    }
  }

  void ArrangeSharedMemory(std::vector<FuncInfo> &funcs_) {
    for (auto &func : funcs_) {
      SharedMemoryManager sm_mng;
      func.stmt = sm_mng.Mutate(func.stmt);
      // Collect the maximum shared memory among of irs.
      total_shared_memory_ = std::max(total_shared_memory_, sm_mng.GetTotalSMSize());
    }
  }

  void CompressDim(std::vector<FuncInfo> &funcs_) {
    for (auto &func : funcs_) {
      DimCompressor dim_comp;
      func.stmt = dim_comp.Run(func.stmt);

      Var left_block = dim_comp.LeftBlock();
      Var left_thread = dim_comp.LeftThread();

      // Collect extent info to funcs_;
      DimInfoVisitor dv(func, left_block, left_thread);
      dv.Visit(func.stmt);

      // Replace all variable to left one.
      std::unordered_map<const Variable *, Expr> vmap;
      if (!IsVarDefault(left_block)) {
        auto block_var = Variable::make(left_block->type, BLOCKIDXX);
        vmap[left_block.get()] = block_var;
        func.block = block_var;
      }
      if (!IsVarDefault(left_thread)) {
        auto thread_var = Variable::make(left_thread->type, THREADIDXX);
        vmap[left_thread.get()] = thread_var;
        func.thread = thread_var;
      }
      func.stmt = Substitute(func.stmt, vmap);
    }
  }

  void UnifyDimInfo(std::vector<FuncInfo> &funcs_) {
    for (const auto &f : funcs_) {
      if (!IsVarDefault(f.block)) {
        block_var_ = f.block;
      }
      if (!IsVarDefault(f.thread)) {
        thread_var_ = f.thread;
      }
    }

    for (size_t i = 0; i < funcs_.size(); ++i) {
      FuncInfo &info = funcs_[i];
      std::unordered_map<const Variable *, Expr> vmap;
      vmap[info.block.get()] = block_var_;
      vmap[info.thread.get()] = thread_var_;
      info.stmt = Substitute(info.stmt, vmap);
    }
  }

  void RemoveDimInfo(std::vector<FuncInfo> &funcs_) {
    for (auto &func : funcs_) {
      func.stmt = RemoveDimAttr().Mutate(func.stmt);
    }
  }

  void UpdateOffSet(std::vector<size_t> &block_info, std::vector<size_t> &max_block_info, std::vector<FuncInfo> &funcs_,
                    size_t &max_thread_num, size_t &max_block) {
    for (auto &func : funcs_) {
      block_info.emplace_back(func.block_ext.as<IntImm>()->value);
      max_thread_num = std::max(max_thread_num, static_cast<size_t>(func.thread_ext.as<IntImm>()->value));
    }

    for (auto it : block_info) {
      max_block += it;
      max_block_info.emplace_back(max_block);
    }

    size_t cur_block_num = 0;
    for (size_t i = 0; i < max_block_info.size(); ++i) {
      int offset = static_cast<int>(cur_block_num);
      funcs_[i].stmt = BlockIndexRewrite(offset).Mutate(funcs_[i].stmt);
      cur_block_num = max_block_info[i];
    }
  }

  Stmt MergeIr(const size_t &max_thread_num, std::vector<FuncInfo> &funcs_, Var fusion_bx, Var fusion_tx,
               const std::vector<size_t> &max_block_info) {
    int fthread_num = funcs_.back().thread_ext.as<IntImm>()->value;
    bool thread_overflow = static_cast<int>(max_thread_num) > fthread_num;
    Stmt res_stmt =
      thread_overflow ? IfThenElse::make(fusion_tx < fthread_num, funcs_.back().stmt) : funcs_.back().stmt;
    for (size_t i = funcs_.size() - 1; i > 0; --i) {
      auto &func = funcs_[i - 1];
      fthread_num = func.thread_ext.as<IntImm>()->value;
      thread_overflow = static_cast<int>(max_thread_num) > fthread_num;
      Stmt stmt = thread_overflow ? IfThenElse::make(fusion_tx < fthread_num, func.stmt) : func.stmt;
      res_stmt = IfThenElse::make(fusion_bx < static_cast<int>(max_block_info[i - 1]), stmt, res_stmt);
    }
    return res_stmt;
  }

  Stmt AddNewDimAttrs(Expr fusion_tx_ext, Expr fusion_bx_ext, Var fusion_tx, Var fusion_bx, Stmt res_stmt) {
    IterVar thread_iv = IterVarNode::make(Range(make_const(Int(32), 0), fusion_tx_ext), fusion_tx,
                                          air::IterVarType::kThreadIndex, THREADIDXX);
    IterVar block_iv = IterVarNode::make(Range(make_const(Int(32), 0), fusion_bx_ext), fusion_bx,
                                         air::IterVarType::kThreadIndex, BLOCKIDXX);
    if (total_shared_memory_ > 0) {
      res_stmt = AttrStmt::make(make_zero(Int(32)), "total_shared_memory", IntImm::make(Int(32), total_shared_memory_),
                                res_stmt);
    }
    res_stmt = AttrStmt::make(thread_iv, THREADEXTENT, fusion_tx_ext, res_stmt);
    return AttrStmt::make(block_iv, THREADEXTENT, fusion_bx_ext, res_stmt);
  }

  Stmt Fusion() {
    // 0. manager shared memory information.
    ArrangeSharedMemory(funcs_);

    // 1. to make all parts with same dim, compress dim to one direction.
    CompressDim(funcs_);

    // 2.unify dim var and get extent
    UnifyDimInfo(funcs_);

    // 3.remove dim info
    RemoveDimInfo(funcs_);

    // 4.update offset of blockIdx.x
    std::vector<size_t> block_info;
    std::vector<size_t> max_block_info;
    size_t max_thread_num = 0;
    size_t max_block = 0;
    UpdateOffSet(block_info, max_block_info, funcs_, max_thread_num, max_block);

    auto &fusion_bx = block_var_;
    if (IsVarDefault(fusion_bx)) {
      fusion_bx = Variable::make(Int(32), BLOCKIDXX);
    }
    auto &fusion_tx = thread_var_;
    if (IsVarDefault(fusion_tx)) {
      fusion_tx = Variable::make(Int(32), THREADIDXX);
    }
    Expr fusion_bx_ext = make_const(Int(32), max_block);       // update it by fusion block extent
    Expr fusion_tx_ext = make_const(Int(32), max_thread_num);  // update it by fusion thread extent

    // 5.merge ir with IfThenElse
    // a.update thread_overflow by comparing thread extent with final extent
    // b.update thread condition
    // c.update block condition
    Stmt res_stmt = MergeIr(max_thread_num, funcs_, fusion_bx, fusion_tx, max_block_info);

    // 6.add new dim attr
    return AddNewDimAttrs(fusion_tx_ext, fusion_bx_ext, fusion_tx, fusion_bx, res_stmt);
  }

 private:
  std::vector<FuncInfo> funcs_;
  Var block_var_;
  Var thread_var_;
  int total_shared_memory_{0};
};

Stmt BlockFusion(std::vector<Stmt> &stmts) { return LowerBlockFusion(stmts).Fusion(); }

Stmt BlockFusionTest(Array<Stmt> funcs) {
  std::vector<Stmt> stmts;
  for (Stmt s : funcs) {
    stmts.push_back(s);
  }
  return BlockFusion(stmts);
}
TVM_REGISTER_API("ir_pass.BlockFusion").set_body_typed(BlockFusionTest);
}  // namespace ir
}  // namespace akg
