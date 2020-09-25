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
  Stmt EmitIf(const isl::ast_node_if &node) final;

  // DMA emitters for GPU
  Expr EmitLoad(const isl::ast_expr &lhs, Type type);
  Stmt EmitRead(const isl::ast_node_user &node);
  Stmt EmitWrite(const isl::ast_node_user &node);

  Stmt EmitAccessNodeFromPromoteAcsCall(isl::id var, const Node *node, Array<Expr> &args);
  Stmt EmitAccessNodeFromPromoteAcsProvide(isl::id var, const Node *node, Array<Expr> &args);

  Stmt EmitSync();
  Stmt EmitAttr();  // thread_extent, virtual_thread

  Expr FindRealizeScope(const isl::id &var);
  std::string FindRealizeScopeToString(const isl::id &var);
  Stmt InsertRealize(Stmt stmt, const isl::id &var);

  VarExpr IterNameAdaptor(std::string name);
  int GetThreadExtent(std::string name);

  // func to modify the stride
  Expr ModifyTheInitExpr(const Expr &e);
  Expr ModifyTheCondExpr(const Expr &e, int inc);
  Expr ModifyTheIterExpr(const VarExpr &iter, int inc, const Expr &init);

  // used for realize emit
  Stmt EmitRealizeForGlobalTensor(Stmt stmt);

  // policy for GMWrite, when the target tensor is temporary tensor, the stmt is not emitted
  bool NoNeedToEmitForTempTensor(const isl::id &id);

  std::set<Tensor> realized_;

  std::unordered_map<const Variable*, Expr> stride_modify_iter_map_;
  std::map<std::string, VarExpr> gpuiter_;
  std::map<std::string, VarExpr> iter_name_map_{
    {B0, VarExpr(BLOCK_IDX_X)},  {B1, VarExpr(BLOCK_IDX_Y)},
    {B2, VarExpr(BLOCK_IDX_Z)},  {T0, VarExpr(THREAD_IDX_X)},
    {T1, VarExpr(THREAD_IDX_Y)}, {T2, VarExpr(THREAD_IDX_Z)}};
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

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_GPU_ISL_EMITTER_H_
