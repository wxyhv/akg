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
#include <algorithm>
#include "emit_insn/insn_info.h"
#include "emit_insn/insn_pattern.h"
#include "emit_insn/insn_args_calculator.h"
namespace akg {
namespace ir {

class TailSpliter : public IRMutator {
 public:
  TailSpliter() = default;

  ~TailSpliter() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn") {
      auto intrin_name = op->value.as<StringImm>()->value;
      if (include_intrin_list_.find(intrin_name) == include_intrin_list_.end()) {
        return s;
      }
      StmtInfoList dst_info_list;
      StmtInfoList src_info_list;
      StmtInfo if_info;
      StmtInfo for_info;

      GetCompactComputationInfo(op->body, dst_info_list, src_info_list, if_info, for_info, false);
      CHECK(!dst_info_list.empty());
      auto dst_info = dst_info_list[0];
      if (src_info_list.empty()) {
        src_info_list = {dst_info.Copy()};
      }

      auto info_list = GetInfoList(dst_info, src_info_list);
      FillEmptyVar(info_list);
      int dst_block_size = GetUbBlkSize(dst_info->dtype_);
      int src_block_size = GetUbBlkSize(src_info_list[0]->dtype_);
      int block_size = dst_block_size < src_block_size ? dst_block_size : src_block_size;
      int cast_block_size = dst_block_size > src_block_size ? dst_block_size : src_block_size;
      int vec_max_len = block_size * FULL_BLOCK_NUM;
      auto args_calculator = InsnArgsCalculator(dst_info_list, src_info_list, for_info, "");
      auto vec_axis_it = args_calculator.GetVecAxisIt();
      bool cast = dst_block_size != src_block_size;
      if (args_calculator.IsValid(vec_axis_it)) {
        auto vec_axis = *vec_axis_it;
        auto vec_axis_shape = vec_axis.extent;
        if (vec_axis_shape >= vec_max_len) {
          if (vec_axis_shape % vec_max_len != 0) {
            return TailBlock(s, vec_axis, vec_max_len);
          }
        } else {
          if (vec_axis_shape < vec_max_len * tail_rate_ && vec_axis_shape > cast_block_size &&
              vec_axis_shape % cast_block_size != 0 && args_calculator.axis_list_.size() > 1) {
            return TailBlock(s, vec_axis, cast_block_size);
          }
        }
      }
      if (!cast && (!args_calculator.IsValid(vec_axis_it) || vec_axis_it->extent <= cast_block_size * tail_rate_)) {
        auto get_block_axis = [&](std::list<InsnAxis> &axis_list) {
          InsnAxis block_axis;
          block_axis.is_valid = false;
          std::vector<InsnAxis> temp_axis_set;
          auto block_stride_lambda = [&](int stride) { return stride % block_size == 0 && stride / block_size <= 4; };
          for (auto axis : axis_list) {
            if (std::all_of(axis.stride_list.begin(), axis.stride_list.end(), block_stride_lambda) &&
                axis.dst_stride != 0 && axis.extent != 0 && axis.extent > FULL_BLOCK_NUM &&
                axis.extent % FULL_BLOCK_NUM != 0) {
              temp_axis_set.push_back(axis);
            }
          }
          if (!temp_axis_set.empty()) {
            return temp_axis_set[0];
          } else {
            return block_axis;
          }
        };
        auto block_axis = get_block_axis(args_calculator.axis_list_);
        if (block_axis.IsValid()) {
          return TailBlock(s, block_axis, FULL_BLOCK_NUM);
        }
      }
      return s;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt TailBlock(const Stmt &s, const InsnAxis &tail_axis, int body_size) {
    return Block::make(TailMake(s, tail_axis, body_size, false), TailMake(s, tail_axis, body_size, true));
  }
  Stmt TailMake(const Stmt &s, const InsnAxis &tail_axis, int body_size, bool is_tail) {
    if (auto attr_stmt = s.as<AttrStmt>()) {
      return AttrStmt::make(attr_stmt->node, attr_stmt->attr_key, attr_stmt->value,
                            TailMake(attr_stmt->body, tail_axis, body_size, is_tail));
    }
    if (auto for_stmt = s.as<For>()) {
      if (Equal(for_stmt->loop_var, tail_axis.var) && GetIntConst(for_stmt->extent) == tail_axis.extent) {
        if (is_tail) {
          return For::make(for_stmt->loop_var, for_stmt->min, Expr(tail_axis.extent % body_size), for_stmt->for_type,
                           for_stmt->device_api, TailMake(for_stmt->body, tail_axis, body_size, is_tail));
        }
        CHECK_NE(body_size, 0);
        Expr remain_extent = Expr(tail_axis.extent / body_size * body_size);
        return For::make(for_stmt->loop_var, for_stmt->min, remain_extent, for_stmt->for_type, for_stmt->device_api,
                         TailMake(for_stmt->body, tail_axis, body_size, is_tail));
      }
      return For::make(for_stmt->loop_var, for_stmt->min, for_stmt->extent, for_stmt->for_type, for_stmt->device_api,
                       TailMake(for_stmt->body, tail_axis, body_size, is_tail));
    }
    if (s.as<Store>() && is_tail) {
      return substitute(tail_axis.var, Add::make(Expr(tail_axis.extent / body_size * body_size), tail_axis.var), s);
    }
    return s;
  }

 private:
  const float tail_rate_{0.6};
  const std::set<std::string> include_intrin_list_ = {
    // binary vec
    "vec_binary_add",
    "vec_binary_sub",
    "vec_binary_mul",
    "vec_binary_min",
    "vec_binary_max",
    "vec_binary_div",
    "vec_binary_and",
    "vec_binary_or",
    "vec_binary_vmadd",
    "vec_binary_vmaddrelu",
    "vec_binary_vmla",

    // single vec
    "vec_single_fabs",
    "vec_single_log",
    "vec_single_exp",
    "vec_single_rec",
    "vec_single_not",
    "vec_single_sqrt",
    "vec_single_rsqrt",
    "vec_single_relu",
    "vec_single_not",
    // Mov
    "broadcast",
    "mask_broadcast",
    // vector_cast
    "vec_single_cast",
    "vec_single_floor",
    "vec_single_round",
    "vec_single_ceil",
    "vec_single_trunc",
    // scalar case
    "vector_dup",
    "vmuls",
    "vadds",
    "vaxpy",
  };
};
Stmt SplitTail(Stmt stmt) {
  auto tail_spliter = TailSpliter();
  auto first_round = tail_spliter.Mutate(stmt);
  auto second_round = tail_spliter.Mutate(stmt);
  return second_round;
}

}  // namespace ir
}  // namespace akg