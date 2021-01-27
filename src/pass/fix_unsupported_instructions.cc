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
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <pass/ir_util.h>
#include <pass/storage_access.h>
#include <stack>

#include "npu_utils.h"
namespace akg {
namespace ir {
/*
 replace vadds by vadd for int32 only consider following example:
 ----------------------------------------------------
 // attr [0] pragma_emit_insn = "vec_single_adds"
 T_add_input_0_local_UB[0] = (input_0_local_UB[0] + 1)

 ========>

 // attr [fix_vadds_int_0_local_UB] storage_scope = "local.UB"
 allocate fix_vadds_int_0_local_UB[int32 * 1]
 // attr [0] pragma_emit_insn = "broadcast"
 fix_vadds_int_0_local_UB[0] = 1
 // attr [0] pragma_emit_insn = "vec_binary_add"
 T_add_input_0_local_UB[0] = (input_0_local_UB[0] + fix_vadds_int_0_local_UB[0])
 -----------------------------------------------------------------------------
*/
class FixVaddsInt : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (NeedFix(op)) {
      auto store = op->body.as<Store>();
      auto add = store->value.as<Add>();

      auto new_buf = VarExpr("fix_vadds_int_" + std::to_string(id++) + LOCAL_BUF, Int(32));

      auto store_stmt = Store::make(new_buf, add->b, 0, const_true(1));
      auto store_attr = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("broadcast"), store_stmt);

      auto new_add = Add::make(add->a, Load::make(Int(32), new_buf, 0, const_true(1)));
      auto vadd_stmt = Store::make(store->buffer_var, new_add, store->index, store->predicate);
      auto vadd_attr = AttrStmt::make(op->node, "pragma_emit_insn", Expr("vec_binary_add"), vadd_stmt);

      auto body = Block::make({store_attr, vadd_attr});
      auto alloc_stmt = Allocate::make(new_buf, Int(32), {make_const(Int(32), 1)}, const_true(), body);
      auto alloc_attr = AttrStmt::make(new_buf, "storage_scope", Expr("local.UB"), alloc_stmt);
      return alloc_attr;
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool NeedFix(const AttrStmt *op) {
    if ((op->value.as<StringImm>() == nullptr)
        || (op->value.as<StringImm>()->value != "vec_single_adds")
        || (op->attr_key != "pragma_emit_insn")) {
      return false;
    }

    auto store = op->body.as<Store>();
    if (store == nullptr) {
      return false;
    }

    auto add = store->value.as<Add>();
    if (add->b.type().is_int() && add->b.type().is_scalar()) {
      return true;
    }

    return false;
  }
  int id = 0;
};

Stmt FixUnsupportedInstruction(Stmt stmt) {
  stmt = FixVaddsInt().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg

