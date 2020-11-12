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
#ifndef POLY_GPU_ISL_EMITTER_H_
#define POLY_GPU_ISL_EMITTER_H_

#include "isl_emitter.h"

namespace akg {
namespace ir {
namespace poly {
/*!
 * IslEmitter for GPU
 */
constexpr auto AKG_ALL_REDUCE = "akg_reduce::ALL_REDUCE";
constexpr auto AKG_X_REDUCE = "akg_reduce::REDUCE2D_X";
constexpr auto AKG_Y_REDUCE = "akg_reduce::REDUCE2D_Y";

// example:
// red_init_SumOp_S_1_0
constexpr auto REDUCE_FLAG_SIZE = 6;
constexpr auto REDUCE_FLAG_TYPE_POS = 2;
constexpr auto REDUCE_FLAG_STMT_PREFIX_POS = 3;
constexpr auto REDUCE_FLAG_STMT_NUM_POS = 4;
constexpr auto REDUCE_FLAG_REDUCE_INDEX = 5;

// example:
// atomic_SumOp
constexpr auto REDUCE_ATOMIC_FLAG_SIZE = 2;
constexpr auto REDUCE_ATOMIC_FLAG = "atomic";
constexpr auto REDUCE_ATOMIC_FLAG_POS = 0;
constexpr auto REDUCE_ATOMIC_FLAG_TYPE_POS = 1;

constexpr auto DEFAULT_TENSOR_INDEX = "[0]";

constexpr auto USELESS_INDEX = "0";
constexpr auto USELESS_SHAPE_SIZE = "1";
constexpr auto SCALAR_TENSOR_PREFIX = "acc_";
constexpr auto SHARED_MEMORY_PREFIX = "__shared__";
constexpr auto SHARED_TENSOR_PREFIX = "red_buf";

constexpr auto REDUCE_LIB_TYPE_ORIGIN = "origin";
constexpr auto REDUCE_LIB_TYPE_PARIS = "paris";
constexpr auto AKG_REDUCE_LIB_SPACE = "akg_reduce";
constexpr auto AKG_REDUCE_LIB_NAME = "AkgReduce";
constexpr auto PARIS_REDUCE_LIB_SPACE = "paris_reduce";
constexpr auto PARIS_REDUCE_LIB_NAME = "ParisReduce";
constexpr auto AKG_REDUCE_RETURN_NAME = "AkgAtomicReturn";
constexpr auto PARIS_REDUCE_RETURN_NAME = "ParisReturn";
constexpr auto TENSOR_MAP_INFO_FLAG = "tensorMapInfo";
constexpr auto TENSOR_INDEX_MODIFY_FLAG = "tensorIndexModify";
constexpr auto REDUCE_LIB_TYPE_FLAG = "reduceLibType";
constexpr auto GM_WRITE_FLAG = "GMWriteFlag";

constexpr auto MEM_TYPE_SHARED = "shared";
constexpr auto MEM_TYPE_LOCAL = "local";
const std::map<std::string, std::string> init_value_adapter{{"0f", "0.0f"}, {"0h", "0"}};

struct ReduceEmitInfo {
  // output tensor info
  std::string output_tensor_name_;
  std::vector<std::string> output_tensor_indexs_;
  std::string output_tensor_info_;

  // ouput promoted tensor info used for atomic emit
  std::string output_promoted_tensor_name_for_atomic_;
  std::vector<std::string> output_promoted_tensor_indexs_for_atomic_;
  std::string output_promoted_tensor_info_for_atomic_;

  // used for atomic tensor
  std::set<std::string> atomic_tensors_;

  // tensor info used for reduce emit
  // This tensor may be output promoted tensor and temporary promoted tensor
  std::string promoted_tensor_name_for_reduce_;
  std::map<std::string, std::vector<std::string>> promoted_tensor_indexs_for_reduce_;
  std::map<std::string, std::vector<std::string>> promoted_tensor_shape_for_reduce_;
  std::string promoted_tensor_info_for_reduce_;

  // used for AkgReduce interface emit
  std::string shared_compute_info_;
  std::string scalar_tensor_info_;

  std::string reduce_op_;
  std::string reduce_stmt_index_;
  bool is_atomic{false};
  std::string output_tensor_data_type_;
  std::string reduce_data_type_;

  // add for init stmt emit
  std::string for_index_{""};
};

class GpuIslEmitter : public IslEmitter {
 public:
  GpuIslEmitter(ScopInfo &info, const NodeInfoRepo &n, const isl::id_list &i) : IslEmitter(info, n, i) {}
  ~GpuIslEmitter() override = default;

  Stmt Emit(const isl::ast_node &node) final;
  Expr Interpret(const isl::ast_expr &e);

 private:
  // override emitters for GPU
  Stmt EmitBlock(const isl::ast_node_block &node) final;
  Stmt EmitStmt(const isl::ast_node_user &node) final;
  Stmt EmitFor(const isl::ast_node_for &node) final;
  Stmt EmitMark(const isl::ast_node_mark &node_id) override;
  Stmt EmitIf(const isl::ast_node_if &node) final;

  // DMA emitters for GPU
  Expr EmitLoad(const isl::ast_expr &lhs, Type type);
  Expr EmitLoadAtomic(const isl::ast_expr &lhs, Type type);
  Stmt EmitRead(const isl::ast_node_user &node);
  Stmt EmitWrite(const isl::ast_node_user &node);
  Stmt EmitWriteAtomic(const isl::ast_node_user &node);

  Stmt EmitAccessNodeFromPromoteAcsCall(isl::id var, const Node *node, Array<Expr> &args);
  Stmt EmitAccessNodeFromPromoteAcsProvide(isl::id var, const Node *node, Array<Expr> &args);

  Stmt EmitSync();
  Stmt EmitReduceInit(const isl::ast_node_user &node);
  Stmt EmitReduceUpdate(const isl::ast_node_user &node);
  Stmt EmitReduceArea(const isl::ast_node_user &node);
  Stmt EmitAttr();  // thread_extent, virtual_thread

  Expr FindRealizeScope(const isl::id &var);
  std::string FindRealizeScopeToString(const isl::id &var);
  Stmt InsertRealize(Stmt stmt, const isl::id &var);

  VarExpr IterNameAdaptor(std::string name);
  int GetThreadExtent(const std::string &name);

  // func to modify the stride
  Expr ModifyTheInitExpr(const Expr &e);
  Expr ModifyTheCondExpr(const Expr &e, int inc);
  Expr ModifyTheIterExpr(const VarExpr &iter, int inc, const Expr &init);

  // used for realize emit
  Stmt EmitRealizeForGlobalTensor(Stmt stmt);

  // policy for GMWrite, when the target tensor is temporary tensor, the stmt is not emitted
  bool NoNeedToEmitForTempTensor(const isl::id &id);

  // used for reduce
  std::string PrepareAkgReduceInfo();
  std::string PrepareAkgAtomicReturnInfo();
  void MakeOutputTensorInfo();
  void MakeOutputPromotedTensorInfoForAtomic();
  void MakePromotedTensorInfoForReduce();
  std::string MakePromotedTensorInitStmt(std::string init_value);
  Stmt EmitAkgAtomicReturnInfo(Stmt s, std::string info);
  std::string GetTheIndexOfPromotedTensor(std::string s);

  std::set<Tensor> realized_;

  std::unordered_map<const Variable *, Expr> stride_modify_iter_map_;
  std::map<std::string, VarExpr> gpuiter_;
  std::map<std::string, VarExpr> iter_name_map_{{B0, VarExpr(BLOCK_IDX_X)},  {B1, VarExpr(BLOCK_IDX_Y)},
                                                {B2, VarExpr(BLOCK_IDX_Z)},  {T0, VarExpr(THREAD_IDX_X)},
                                                {T1, VarExpr(THREAD_IDX_Y)}, {T2, VarExpr(THREAD_IDX_Z)}};

  // used for reduce emit
  bool in_reduce_area_{false};
  struct ReduceEmitInfo reduce_info_;
  bool is_sync_before_{false};
};

class AddAttrCheck : public air::ir::IRVisitor {
 public:
  AddAttrCheck() = default;
  ~AddAttrCheck() = default;
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == air::ir::attr::thread_extent) {
      const auto iv = op->node.as<IterVarNode>();
      std::string name = iv->thread_tag;
      if (name == THREAD_IDX_X || name == THREAD_IDX_Y || name == THREAD_IDX_Z) {
        need_add_ = false;
        return;
      }
    }
    IRVisitor::Visit_(op);
  }

  bool Run(const Stmt &op) {
    IRVisitor::Visit(op);
    return need_add_;
  }

 private:
  bool need_add_{true};
};

class AkgReduceAddTensorIndex : public air::ir::IRMutator {
 public:
  explicit AkgReduceAddTensorIndex(std::map<std::string, std::vector<std::string>> i,
                                   std::map<std::string, std::vector<std::string>> j)
      : indexs(i), shapes(j) {}
  ~AkgReduceAddTensorIndex() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "tensorIndexModify") {
      std::string tensor_name = op->value.as<StringImm>()->value;

      CHECK(op->body.as<Evaluate>());
      auto evaluate = op->body.as<Evaluate>();

      CHECK(evaluate->value.as<Call>());
      const Call *call = evaluate->value.as<Call>();

      if (call->args.size() != 2) {
        return IRMutator::Mutate_(op, s);
      }

      const StringImm *si = call->args[1].as<StringImm>();
      std::string arg2 = si->value;

      std::string::size_type n = arg2.find(tensor_name);
      if (n == std::string::npos) {
        return IRMutator::Mutate_(op, s);
      }

      int tensor_len = tensor_name.size();
      int size = indexs[tensor_name].size();
      if (size == 0) {
        arg2 = arg2.insert(n + tensor_len, DEFAULT_TENSOR_INDEX);
      } else if (size == 1) {
        arg2 = arg2.insert(n + tensor_len, "[");
        arg2 = arg2.insert(n + tensor_len + 1, indexs[tensor_name].at(0));
        arg2 = arg2.insert(n + tensor_len + 1 + indexs[tensor_name].at(0).size(), "]");
      } else {
        std::string index = "[";
        for (int i = 0; i < size - 1; ++i) {
          index += indexs[tensor_name].at(i);
          for (int j = i + 1; j < size; ++j) {
            index += "*";
            index += "(";
            index += shapes[tensor_name].at(j);
            index += ")";
          }
          index += "+";
        }
        index += indexs[tensor_name].at(size - 1);
        index += "]";
        arg2 = arg2.insert(n + tensor_len, index);
      }

      Array<Expr> new_args;
      new_args.push_back(call->args[0]);
      new_args.push_back(StringImm::make(arg2));

      return Evaluate::make(
        Call::make(call->type, call->name, new_args, call->call_type, call->func, call->value_index));
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::map<std::string, std::vector<std::string>> indexs;
  std::map<std::string, std::vector<std::string>> shapes;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_GPU_ISL_EMITTER_H_
