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

#include <tvm/base.h>
#include <tvm/ir_pass.h>

#include "common/array_api.h"
#include "pass/expr_alg_simplify.h"
#include "insn_pattern.h"
#include "insn_args_calculator.h"

namespace akg {

InsnAxis::InsnAxis(const For *for_stmt, const Array<StmtStoreInfo> &info_list) {
  this->var = for_stmt->loop_var;
  this->extent = GetInt32Const(for_stmt->extent);
  this->min = GetInt32Const(for_stmt->min);
  int index = 0;
  for (auto it : info_list) {
    auto stride = GetInt32Const(GetStrideByAxis(it->var_, it->strides_, this->var));
    this->stride_list.push_back(stride);
    if (index == 0) {
      this->dst_stride = stride;
    } else {
      this->src_stride_list.push_back(stride);
    }
    index++;
  }
}

Expr InsnAxis::GetStrideByAxis(const Array<Var> &vars, const Array<Expr> &strides, Var obj_var) {
  int index = 0;
  for (auto var_it : vars) {
    if (Equal(var_it, obj_var)) {
      return strides[index];
    }
    index++;
  }
  return Expr(0);
};

bool InsnAxis::IsValid() { return this->is_valid; }

void InsnAxis::Print(const std::string &name) {
  if (!name.empty()) {
    LOG(DEBUG) << "********** " << name << " ************";
  }
  auto r_stride = this->src_stride_list.size() > 1 ? src_stride_list[1] : 99999;
  LOG(DEBUG) << "var:" << this->var << " extent:" << this->extent << " min:" << this->min
             << " dst_stride:" << this->dst_stride << " src_stride_l:" << this->src_stride_list.front()
             << "src_stride_r:" << r_stride;
}

Array<StmtStoreInfo> GetInfoList(const StmtStoreInfo &dst_info, const Array<StmtStoreInfo> &src_info_list) {
  Array<StmtStoreInfo> res;
  res.push_back(dst_info.Copy());
  for (auto it : src_info_list) {
    res.push_back(it.Copy());
  }
  return res;
};

std::list<InsnAxis> GetAxisList(const StmtInfo &for_info, const Array<StmtStoreInfo> &info_list) {
  std::list<InsnAxis> axis_list;
  for (auto it : for_info.ops_) {
    auto for_stmt = it.as<For>();
    CHECK(for_stmt);
    auto axis = InsnAxis(for_stmt, info_list);
    axis_list.push_back(axis);
  }
  return axis_list;
}

void Print(std::list<InsnAxis> &axis_list) {
  LOG(DEBUG) << "+++++++++++++++++++ AXIS_LIST +++++++++++++++++++";
  int index = 0;
  for (auto it : axis_list) {
    LOG(DEBUG) << "================== INDEX " << index << " =================";
    it.Print();
    index++;
  }
  LOG(DEBUG) << "------------------ END ---------------------";
}

InsnArgsCalculator::InsnArgsCalculator(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list,
                                       const StmtInfo &for_info, const std::string &intrin_name)
    : dst_info_list_(dst_info_list), src_info_list_(src_info_list), for_info_(for_info), intrin_name_(intrin_name) {
  InitArg();
  CalAxis();
}

void InsnArgsCalculator::CalAxis() {
  CHECK(!dst_info_list_.empty());
  dst_info_ = dst_info_list_[0];
  if (src_info_list_.empty()) {
    src_info_list_ = {dst_info_.Copy()};
  }
  auto src_info = src_info_list_[0];
  dst_info_.Print();
  for (auto src_info_it : src_info_list_) {
    src_info_it.Print();
    if (src_info_it->name_ == dst_info_->name_) {
      meta_.same_dst_src = true;
    }
  }
  meta_.dst_block_size = GetUbBlkSize(dst_info_->dtype_);
  meta_.src_block_size = GetUbBlkSize(src_info->dtype_);
  meta_.cast = meta_.dst_block_size != meta_.src_block_size;
  meta_.block_size = meta_.dst_block_size <= meta_.src_block_size ? meta_.dst_block_size : meta_.src_block_size;
  meta_.src_dtype = src_info->dtype_;
  meta_.dst_dtype = dst_info_->dtype_;
  meta_.dtype = meta_.dst_dtype.bits() >= meta_.src_dtype.bits() ? meta_.dst_dtype : meta_.src_dtype;
  auto elem_offset_mod = ir::ExprSimplifier().Simplify(Mod::make(dst_info_->elem_offset_, meta_.block_size));
  if (elem_offset_mod.as<IntImm>()) {
    meta_.block_offset = elem_offset_mod.as<IntImm>()->value;
  }
  axis_list_ = GetAxisList(for_info_, GetInfoList(dst_info_, src_info_list_));
}  // namespace akg

void InsnArgsCalculator::InitArg() {
  arg_.src_m1_list = {0, 0};
  arg_.src_m0_list = {1, 1};
}

std::function<bool(const InsnAxis &)> InsnArgsCalculator::GetStrideLambda() {
  return [&](const InsnAxis &axis) {
    auto is_stride = [&](int stride) { return stride % meta_.block_size == 0; };
    auto zero_stride = [&](int stride) { return stride == 0; };
    return std::all_of(axis.stride_list.begin(), axis.stride_list.end(), is_stride) &&
           !std::all_of(axis.stride_list.begin(), axis.stride_list.end(), zero_stride);
  };
}

std::function<bool(const InsnAxis &)> InsnArgsCalculator::GetM0LimitLambda() {
  return [&](const InsnAxis &axis) {
    auto is_limit = [&](int stride) { return stride / meta_.block_size < MAX_STRIDE_M0_SINGLE; };
    return std::all_of(axis.stride_list.begin(), axis.stride_list.end(), is_limit);
  };
}
std::function<bool(const InsnAxis &)> InsnArgsCalculator::GetBlockStrideLimitLambda() {
  return [&](const InsnAxis &axis) {
    auto is_limit = [&](int stride) { return stride / meta_.block_size <= max_block_stride_; };
    return std::all_of(axis.stride_list.begin(), axis.stride_list.end(), is_limit);
  };
}
std::function<bool(const InsnAxis &)> InsnArgsCalculator::GetM1LimitLambda() {
  return [&](const InsnAxis &axis) {
    auto is_limit = [&](int stride) { return stride / meta_.src_block_size < MAX_STRIDE_M1; };
    return axis.dst_stride / meta_.dst_block_size < MAX_STRIDE_M1 &&
           std::all_of(axis.src_stride_list.begin(), axis.src_stride_list.end(), is_limit);
  };
}

std::function<bool(const InsnAxis &)> And(const std::list<std::function<bool(const InsnAxis &)>> &lambda_list) {
  return [&lambda_list](const InsnAxis &axis) {
    bool res = true;
    for (auto lambda_it : lambda_list) {
      res = res && lambda_it(axis);
    }
    return res;
  };
}

AxisIt InsnArgsCalculator::GetAxisByLambda(const std::function<bool(const InsnAxis &)> &lambda) {
  for (auto axis_it = axis_list_.begin(); axis_it != axis_list_.end(); axis_it++) {
    if (lambda(*axis_it)) {
      return axis_it;
    }
  }
  return axis_list_.end();
}

InsnAxis InsnArgsCalculator::ExtractAxis(AxisIt &it) {
  InsnAxis res = *it;
  axis_list_.erase(it);
  return res;
}

bool InsnArgsCalculator::IsValid(AxisIt &it) { return it != axis_list_.end(); }

void AxisSort(std::list<InsnAxis> &axis_arr, bool order = true) {
  auto up_compare = [&](InsnAxis &a, InsnAxis &b) { return a.extent < b.extent; };
  auto down_compare = [&](InsnAxis &a, InsnAxis &b) { return a.extent > b.extent; };

  if (order) {
    axis_arr.sort(up_compare);
  } else {
    axis_arr.sort(down_compare);
  }
}

AxisIt InsnArgsCalculator::GetVecAxisIt() {
  axis_list_.reverse();
  auto IsVecAxis = [&](const InsnAxis &axis) {
    return !(std::any_of(axis.stride_list.begin(), axis.stride_list.end(), [](int stride) { return stride > 1; }) ||
             std::all_of(axis.stride_list.begin(), axis.stride_list.end(), [](int stride) { return stride == 0; }));
  };
  return GetAxisByLambda(IsVecAxis);
}

SplitStat InsnArgsCalculator::SplitAxis(int extent, InsnAxis &axis) {
  if (axis.extent <= extent) {
    return NO_SPLIT;
  }
  if (axis.extent % extent != 0) {
    return TAIL;
  }
  InsnAxis new_axis;
  new_axis.extent = axis.extent / extent;
  for (auto stride : axis.stride_list) {
    new_axis.stride_list.push_back(stride * extent);
  }
  auto temp_stride_list = new_axis.stride_list;
  CHECK(!temp_stride_list.empty());
  new_axis.dst_stride = temp_stride_list.front();
  temp_stride_list.erase(temp_stride_list.begin());
  new_axis.src_stride_list = temp_stride_list;
  new_axis.var = Var(axis.var->name_hint);
  axis_list_.push_back(new_axis);
  axis.extent = extent;
  return SUCCESS;
}

AxisIt InsnArgsCalculator::GetBlockAxis() {
  AxisSort(axis_list_);
  auto stride_lambda = GetStrideLambda();
  auto m0_limit_lambda = GetM0LimitLambda();
  auto block_stride_limit_lambda = GetBlockStrideLimitLambda();
  auto axis_it =
    GetAxisByLambda(And({stride_lambda, m0_limit_lambda, block_stride_limit_lambda, [&](const InsnAxis &axis) {
                           return axis.extent >= FULL_BLOCK_NUM && axis.extent % FULL_BLOCK_NUM == 0;
                         }}));
  if (IsValid(axis_it)) {
    return axis_it;
  }
  axis_it = GetAxisByLambda(And({stride_lambda, m0_limit_lambda, block_stride_limit_lambda,
                                 [&](const InsnAxis &axis) { return axis.extent >= FULL_BLOCK_NUM; }}));
  if (IsValid(axis_it) && axis_list_.size() == 1) {
    return axis_it;
  }
  axis_list_.reverse();
  axis_it = GetAxisByLambda(And({stride_lambda, m0_limit_lambda, block_stride_limit_lambda,
                                 [&](const InsnAxis &axis) { return axis.extent < FULL_BLOCK_NUM; }}));
  if (IsValid(axis_it)) {
    return axis_it;
  }
  return GetAxisByLambda(And({stride_lambda, m0_limit_lambda, [&](const InsnAxis &axis) {
                                return axis.extent <= FULL_BLOCK_NUM || axis.extent % FULL_BLOCK_NUM == 0;
                              }}));
}

AxisIt InsnArgsCalculator::GetRepeatAxisIt() {
  AxisSort(axis_list_);
  auto stride_lambda = GetStrideLambda();
  auto m1_limit_lambda = GetM1LimitLambda();
  auto axis_it = GetAxisByLambda(
    And({stride_lambda, m1_limit_lambda, [&](const InsnAxis &axis) { return axis.extent >= MAX_REPEAT - 1; }}));
  if (IsValid(axis_it)) {
    return axis_it;
  }
  axis_list_.reverse();
  return GetAxisByLambda(And({stride_lambda, m1_limit_lambda}));
}

void InsnArgsCalculator::SetArgMask(int len) {
  SetArgBlockNum(1);
  SetArgBlockLen(len);
}

void InsnArgsCalculator::SetArgBlockNum(int data_num) { arg_.block_num = data_num; }
void InsnArgsCalculator::SetArgBlockLen(int data_len) { arg_.block_len = data_len; }

void InsnArgsCalculator::SetArgM0(int dst_m0, int lsrc_m0, int rsrc_m0 = 0) {
  arg_.dst_m0 = dst_m0;
  arg_.src_m0_list = {lsrc_m0, rsrc_m0};
}

void InsnArgsCalculator::SetArgM1(int dst_m1, int lsrc_m1, int rsrc_m1 = 0) {
  arg_.dst_m1 = dst_m1;
  arg_.src_m1_list = {lsrc_m1, rsrc_m1};
}

void InsnArgsCalculator::SetArgRepeat(int repeat) { arg_.repeat = repeat; }

void InsnArgsCalculator::BlockAxisReduction() {
  Print(axis_list_);
  auto block_axis_it = GetBlockAxis();
  if (IsValid(block_axis_it)) {
    auto origin_block_axis = *block_axis_it;
    InsnAxis block_axis = ExtractAxis(block_axis_it);
    if (block_axis.extent % FULL_BLOCK_NUM != 0 && block_axis.extent > FULL_BLOCK_NUM) {
      arg_.tail_len = block_axis.extent % FULL_BLOCK_NUM;
      block_axis.extent = FloorTo(block_axis.extent, FULL_BLOCK_NUM);
      arg_.dst_tail_offset = block_axis.dst_stride * block_axis.extent;
      for (auto stride : block_axis.src_stride_list) {
        arg_.src_tail_offset_list.push_back(stride * block_axis.extent);
      }
      SplitAxis(FULL_BLOCK_NUM, block_axis);
      auto repeat_axis_it = GetRepeatAxisIt();
      if (!IsValid(repeat_axis_it) && axis_list_.size() > 0) {
        for (auto it = axis_list_.begin(); it != axis_list_.end(); it++) {
          if (it->var->name_hint == block_axis.var->name_hint) {
            axis_list_.erase(it);
            break;
          }
        }
        axis_list_.push_back(origin_block_axis);
        return;
      }
    } else {
      SplitAxis(FULL_BLOCK_NUM, block_axis);
    }

    block_axis.Print("BLOCK_AXIS");
    SetArgM0(block_axis.dst_stride / meta_.block_size, block_axis.src_stride_list.front() / meta_.block_size,
             block_axis.src_stride_list.back() / meta_.block_size);
    SetArgBlockNum(block_axis.extent);
  }
}

void InsnArgsCalculator::RepeatAxisReduction() {
  Print(axis_list_);
  auto repeat_axis = GetRepeatAxis();
  if (repeat_axis.IsValid()) {
    repeat_axis.Print("REPEAT_AXIS");
    SetArgM1(repeat_axis.dst_stride / meta_.dst_block_size, repeat_axis.src_stride_list.front() / meta_.src_block_size,
             repeat_axis.src_stride_list.back() / meta_.src_block_size);
    SetArgRepeat(repeat_axis.extent);
  }
}

InsnAxis InsnArgsCalculator::GetInvalidAxis() {
  InsnAxis res;
  res.is_valid = false;
  return res;
}

InsnAxis InsnArgsCalculator::GetRepeatAxis() {
  auto repeat_axis_it = GetRepeatAxisIt();
  if (IsValid(repeat_axis_it)) {
    InsnAxis repeat_axis = ExtractAxis(repeat_axis_it);
    SplitAxis(MAX_REPEAT - 1, repeat_axis);
    return repeat_axis;
  }
  return GetInvalidAxis();
}

void InsnArgsCalculator::CastCaseReduction() {
  if (axis_list_.empty()) {
    return;
  }
  Print(axis_list_);
  int cast_block_size = meta_.dst_block_size < meta_.src_block_size ? meta_.dst_block_size : meta_.src_block_size;
  auto vec_axis_it = GetVecAxisIt();
  if (IsValid(vec_axis_it)) {
    InsnAxis vec_axis = ExtractAxis(vec_axis_it);
    int max_vec_len = cast_block_size * FULL_BLOCK_NUM;
    if (vec_axis.extent > cast_block_size && vec_axis.extent < max_vec_len) {
      SetArgMask(DivFloor(vec_axis.extent, cast_block_size) * cast_block_size);
      SetArgM0(1, 1, 1);
    } else if (vec_axis.extent >= max_vec_len) {
      SplitAxis(max_vec_len, vec_axis);
      SetArgMask(DivFloor(vec_axis.extent, cast_block_size) * cast_block_size);
      SetArgM0(1, 1, 1);
    } else {
      SetArgBlockLen(cast_block_size);
    }
  }
  RepeatAxisReduction();
}

int DivFloor(int a, int b) {
  if (a % b == 0) {
    return a / b;
  } else {
    return a / b + 1;
  }
}

void InsnArgsCalculator::InsnReduction() {
  if (axis_list_.empty()) {
    return;
  }
  Print(axis_list_);
  auto vec_axis_it = GetVecAxisIt();
  meta_.scalar = !IsValid(vec_axis_it);
  if (!meta_.scalar) {
    InsnAxis vec_axis = ExtractAxis(vec_axis_it);
    int max_vec_len = meta_.block_size * FULL_BLOCK_NUM;
    if (vec_axis.extent > meta_.block_size && vec_axis.extent < max_vec_len &&
        (vec_axis.extent % meta_.block_size != 0 || vec_axis.extent > max_vec_len * meta_.vec_rate)) {
      vec_axis.Print("VEC_BLOCK_AXIS");
      SetArgMask(DivFloor(vec_axis.extent, meta_.block_size) * meta_.block_size);
      SetArgM0(1, 1, 1);
    } else {
      SplitAxis(meta_.block_size, vec_axis);
      vec_axis.Print("VEC_AXIS");
      SetArgBlockLen(meta_.block_size);
      BlockAxisReduction();
    }
    RepeatAxisReduction();
  } else {
    BlockAxisReduction();
    RepeatAxisReduction();
  }
  Print(axis_list_);
}

Expr InsnArgsCalculator::GetOffset(int stride_index) {
  Expr res = Expr(0);
  for (auto axis_it : axis_list_) {
    auto stride = axis_it.stride_list[stride_index];
    auto mul_expr = Mul::make(stride, axis_it.var);
    res = Add::make(mul_expr, res);
  }
  return Simplify(res);
}

StmtInfo InsnArgsCalculator::ExportForInfo() {
  if (for_info_.ops_.empty()) {
    return for_info_;
  }
  int last_index = for_info_.ops_.size() - 1;
  auto last_for = for_info_.ops_[last_index].as<For>();
  auto store_stmt = last_for->body;
  Stmt for_stmt = store_stmt;
  StmtInfo result;
  for (auto axis_it : axis_list_) {
    for_stmt = For::make(axis_it.var, axis_it.min, axis_it.extent, last_for->for_type, last_for->device_api, for_stmt);
    result.ops_.push_back(for_stmt);
    result.vars_.push_back(axis_it.var);
  }
  return result;
}

PatternResult InsnArgsCalculator::ExportResult() {
  PatternResult res;
  auto arg_info = ArgInfo(make_node<ArgInfoNode>());
  auto body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
  body_args.GetNode()->body_num_ = arg_.body_num;
  body_args.GetNode()->body_offset_ = meta_.block_size * FULL_BLOCK_NUM;
  body_args.GetNode()->repeat_ = Expr(arg_.repeat);
  body_args.GetNode()->dst_stride_m0_ = Expr(arg_.dst_m0);
  body_args.GetNode()->dst_stride_m1_ = Expr(arg_.dst_m1);
  body_args.GetNode()->src_stride_m0_list_ = arg_.src_m0_list;
  body_args.GetNode()->src_stride_m1_list_ = arg_.src_m1_list;
  body_args.GetNode()->vec_mask_ = GetVecMask(arg_.block_len, arg_.block_num, meta_.dtype, meta_.block_offset);
  body_args.GetNode()->block_offset_ = make_const(Int(32), meta_.block_offset);
  arg_info.GetNode()->body_arg_info_ = body_args;
  if (arg_.tail_len > 0) {
    auto tail_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    tail_args.GetNode()->dst_head_ = Expr(arg_.dst_tail_offset);
    tail_args.GetNode()->dst_stride_m1_ = Expr(arg_.dst_m1);
    tail_args.GetNode()->src_stride_m1_list_ = arg_.src_m1_list;
    tail_args.GetNode()->repeat_ = Expr(1);
    tail_args.GetNode()->src_head_list_ = arg_.src_tail_offset_list;
    tail_args.GetNode()->body_offset_ = meta_.block_size * FULL_BLOCK_NUM;
    tail_args.GetNode()->dst_stride_m0_ = Expr(arg_.dst_m0);
    tail_args.GetNode()->src_stride_m0_list_ = arg_.src_m0_list;
    tail_args.GetNode()->vec_mask_ = GetVecMask(arg_.block_len, arg_.tail_len, meta_.dtype, meta_.block_offset);
    tail_args.GetNode()->block_offset_ = make_const(Int(32), meta_.block_offset);
    arg_info.GetNode()->tail_arg_info_ = tail_args;
  }
  StmtInfoList info_list = GetInfoList(dst_info_, src_info_list_);
  CleanZeroStrides(info_list);
  for (size_t i = 0; i < info_list.size(); i++) {
    info_list[i].GetNode()->insn_offset_ = GetOffset(i);
  }
  info_list[1].Print();
  res.for_info = ExportForInfo();
  res.arg_info = arg_info;
  res.dst_info_list = {info_list[0]};
  if (info_list.size() > 2) {
    res.src_info_list = {info_list[1], info_list[2]};
  } else {
    res.src_info_list = {info_list[1]};
  }
  body_args.Print();
  if (arg_info->tail_arg_info_.defined()) {
    arg_info->tail_arg_info_.Print();
  }
  return res;
}

SingleVecInsnArgsCalculator::SingleVecInsnArgsCalculator(const StmtInfoList &dst_info_list,
                                                         const StmtInfoList &src_info_list, const StmtInfo &for_info,
                                                         const std::string &intrin_name)
    : InsnArgsCalculator(dst_info_list, src_info_list, for_info, intrin_name) {}

PatternResult SingleVecInsnArgsCalculator::GetInsnArgs() {
  if (meta_.cast) {
    CastCaseReduction();
  } else {
    InsnReduction();
  }
  return ExportResult();
}

BinaryVecInsnArgsCalculator::BinaryVecInsnArgsCalculator(const StmtInfoList &dst_info_list,
                                                         const StmtInfoList &src_info_list, const StmtInfo &for_info,
                                                         const std::string &mode, const std::string &intrin_name,
                                                         bool expand_mask)
    : InsnArgsCalculator(dst_info_list, src_info_list, for_info, intrin_name), mode_{mode}, expand_mask_{expand_mask} {
  if (mode_ == "reduction" && src_info_list_.size() == 2 && src_info_list_[0]->name_ == dst_info_list[0]->name_) {
    auto temp = src_info_list_[0].Copy();
    src_info_list_.Set(0, src_info_list_[1].Copy());
    src_info_list_.Set(1, temp);
    CalAxis();
  }
}

PatternResult BinaryVecInsnArgsCalculator::GetInsnArgs() {
  LOG(DEBUG) << "Binary vec Insn reduction";
  InsnReduction();
  return ExportResult();
}

std::function<bool(const InsnAxis &)> BinaryVecInsnArgsCalculator::GetM0LimitLambda() {
  return [&](const InsnAxis &axis) {
    auto is_limit = [&](int stride) { return stride / meta_.block_size < MAX_STRIDE_M0; };
    return std::all_of(axis.stride_list.begin(), axis.stride_list.end(), is_limit) && axis.dst_stride != 0;
  };
}
std::function<bool(const InsnAxis &)> BinaryVecInsnArgsCalculator::GetM1LimitLambda() {
  return [&](const InsnAxis &axis) {
    auto is_limit = [&](int stride) { return stride / meta_.src_block_size < MAX_STRIDE_M1; };
    return axis.dst_stride / meta_.dst_block_size < MAX_STRIDE_M1 &&
           std::all_of(axis.src_stride_list.begin(), axis.src_stride_list.end(), is_limit);
  };
}

void BinaryVecInsnArgsCalculator::InsnReduction() {
  if (axis_list_.empty()) {
    return;
  }
  Print(axis_list_);

  auto vec_axis_it = GetVecAxisIt();
  meta_.scalar = !IsValid(vec_axis_it);
  if (!meta_.scalar) {
    vec_axis_ = *vec_axis_it;
    InsnAxis vec_axis = ExtractAxis(vec_axis_it);
    auto bad_axis_lambda = [&](const InsnAxis &axis) {
      int min_stride = vec_axis_it->extent;
      auto dst_name = dst_info_list_[0]->name_;
      if (meta_.same_dst_src && axis.dst_stride < min_stride && axis.dst_stride != 0) {
        return true;
      }
      return false;
    };
    auto bad_axis_it = GetAxisByLambda(bad_axis_lambda);
    InsnAxis bad_axis;
    bad_axis.is_valid = false;
    if (IsValid(bad_axis_it)) {
      bad_axis = ExtractAxis(bad_axis_it);
    }
    int max_vec_len = meta_.block_size * FULL_BLOCK_NUM;
    if (vec_axis.extent > meta_.block_size && vec_axis.extent < max_vec_len &&
        (vec_axis.extent % meta_.block_size != 0 || vec_axis.extent > max_vec_len * meta_.vec_rate)) {
      vec_axis.Print("VEC_BLOCK_AXIS");
      if (expand_mask_) {
        SetArgMask(DivFloor(vec_axis.extent, meta_.block_size) * meta_.block_size);
      } else {
        SetArgMask(vec_axis.extent);
      }
      SetArgM0(1, 1, 1);
    } else {
      SplitAxis(meta_.block_size, vec_axis);
      vec_axis.Print("VEC_AXIS");
      if (expand_mask_ && mode_ != "reduction") {
        SetArgBlockLen(meta_.block_size);
      } else {
        SetArgBlockLen(vec_axis.extent);
      }
      BlockAxisReduction();
    }
    RepeatAxisReduction();
    if (bad_axis.IsValid()) {
      axis_list_.push_back(bad_axis);
    }
  } else {
    BlockAxisReduction();
    RepeatAxisReduction();
  }
  Print(axis_list_);
}

PatternResult LastAxisReduceInsnArgsCalculator::GetInsnArgs() {
  CalcParams();
  Array<Var> elim_var;
  elim_var = GetPattern();
  arg_info.GetNode()->pattern_ = PATTERN_1D;
  return GenResult(elim_var);
}

Array<Var> LastAxisReduceInsnArgsCalculator::GetPattern() {
  int body_len = params.last_dim_shape / params.vec_max_len * params.vec_max_len;
  int tail_len = params.last_dim_shape % params.vec_max_len;
  int cmd_body_len = 0;
  bool is_vadd = intrin_name == "vadd";
  int repeat_stride = FULL_BLOCK_NUM;
  if (is_vadd) {
    repeat_stride = 1;
  }
  const int fp16_block_size = 16;

  if (body_len > 0) {
    body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    body_args.GetNode()->body_num_ = 1;
    body_args.GetNode()->body_offset_ = params.vec_max_len;
    body_args.GetNode()->repeat_ = Expr(body_len / params.vec_max_len);
    // Here use dst_stride_m1 as dst_stride
    body_args.GetNode()->dst_stride_m1_ = Expr(1);
    body_args.GetNode()->src_stride_m0_list_ = {Expr(1)};
    body_args.GetNode()->src_stride_m1_list_ = {Expr(FULL_BLOCK_NUM)};
    body_args.GetNode()->vec_mask_ = GetVecMask(params.vec_max_len, 1, dst_info->dtype_);
    cmd_body_len += GetInt32Const(body_args->repeat_) * repeat_stride;
  }
  if (tail_len > 0) {
    tail_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    tail_args.GetNode()->body_offset_ = params.vec_max_len;
    tail_args.GetNode()->dst_head_ = Expr(cmd_body_len);
    tail_args.GetNode()->src_head_list_ = {Expr(body_len)};
    tail_args.GetNode()->repeat_ = Expr(1);
    tail_args.GetNode()->dst_stride_m1_ = Expr(1);
    tail_args.GetNode()->src_stride_m0_list_ = {Expr(1)};
    tail_args.GetNode()->src_stride_m1_list_ = {Expr(0)};
    tail_args.GetNode()->vec_mask_ = GetVecMask(tail_len, 1, dst_info->dtype_);
    if (is_vadd) {
      cmd_body_len += 1;
    } else {
      cmd_body_len += tail_len / fp16_block_size;
      if (tail_len % fp16_block_size != 0) {
        cmd_body_len += 1;
      }
    }
  }
  // cmd_body_len > 1 means vcadd size greater than 128, need to use vcadd again to compute final result
  // if cmd_body_len > 128, then need to recursively emit vcadd
  while (cmd_body_len > 1) {
    int cmd_tail_len = cmd_body_len % params.vec_max_len;
    cmd_body_len = cmd_body_len / params.vec_max_len;
    if (cmd_body_len > 0) {
      VectorArgInfo mix_vec_args = VectorArgInfo(make_node<VectorArgInfoNode>());
      mix_vec_args.GetNode()->repeat_ = Expr(cmd_body_len);
      mix_vec_args.GetNode()->dst_head_ = Expr(0);
      mix_vec_args.GetNode()->src_head_list_ = {Expr(0)};
      mix_vec_args.GetNode()->dst_stride_m1_ = Expr(1);
      mix_vec_args.GetNode()->src_stride_m0_list_ = {Expr(1)};
      mix_vec_args.GetNode()->src_stride_m1_list_ = {Expr(FULL_BLOCK_NUM)};
      mix_vec_args.GetNode()->vec_mask_ = GetVecMask(params.vec_max_len, 1, dst_info->dtype_);
      mix_vec_arg_list.push_back(mix_vec_args);
      if (!is_vadd) {
        cmd_body_len *= FULL_BLOCK_NUM;
      }
    }
    if (cmd_tail_len > 0) {
      VectorArgInfo mix_vec_args = VectorArgInfo(make_node<VectorArgInfoNode>());
      mix_vec_args.GetNode()->repeat_ = Expr(1);
      mix_vec_args.GetNode()->dst_head_ = Expr(cmd_body_len);
      if (is_vadd) {
        mix_vec_args.GetNode()->src_head_list_ = {Expr(cmd_body_len * params.vec_max_len)};
      } else {
        mix_vec_args.GetNode()->src_head_list_ = {Expr(cmd_body_len / FULL_BLOCK_NUM * params.vec_max_len)};
      }
      mix_vec_args.GetNode()->dst_stride_m1_ = Expr(1);
      mix_vec_args.GetNode()->src_stride_m0_list_ = {Expr(1)};
      mix_vec_args.GetNode()->src_stride_m1_list_ = {Expr(FULL_BLOCK_NUM)};
      mix_vec_args.GetNode()->vec_mask_ = GetVecMask(cmd_tail_len, 1, dst_info->dtype_);
      if (is_vadd) {
        cmd_body_len += 1;
      } else {
        cmd_body_len += cmd_tail_len / fp16_block_size;
        if (cmd_tail_len % fp16_block_size != 0) {
          cmd_body_len += 1;
        }
      }
      mix_vec_arg_list.push_back(mix_vec_args);
    }
  }

  params.insn_offset_scale_factor = Expr(params.block_size);
  int max_num = body_len / params.vec_max_len;
  if (intrin_name == "vmax" || intrin_name == "vmin") {
    max_num *= FULL_BLOCK_NUM;
  }
  if (max_num >= params.block_size) {
    params.insn_offset_scale_factor = max_num + params.block_size - 1;
    if (tail_len > 0) {
      params.insn_offset_scale_factor += 1;
    }
    params.insn_offset_scale_factor = truncdiv(params.insn_offset_scale_factor, params.block_size) * params.block_size;
  }

  if (!params.src_var.empty()) {
    return GetRange(params.src_var, -1, 1);
  }

  return {};
}

PatternResult LastAxisReduceInsnArgsCalculator::GenResult(const Array<Var> &elim_var) {
  dst_info.GetNode()->insn_offset_ = GetInsnOffset(dst_info, elim_var) * params.insn_offset_scale_factor;
  src_info.GetNode()->insn_offset_ = GetInsnOffset(src_info, elim_var);

  if (body_args.defined()) {
    body_args.GetNode()->insn_offset_scale_factor_ = params.insn_offset_scale_factor;
  }
  if (tail_args.defined()) {
    tail_args.GetNode()->insn_offset_scale_factor_ = params.insn_offset_scale_factor;
  }
  for (auto &arg : mix_vec_arg_list) {
    arg.GetNode()->insn_offset_scale_factor_ = params.insn_offset_scale_factor;
  }

  arg_info.GetNode()->body_arg_info_ = body_args;
  arg_info.GetNode()->tail_arg_info_ = tail_args;
  arg_info.GetNode()->reduction_tail_args_ = mix_vec_arg_list;

  CleanForInfoVars(for_info, elim_var);
  arg_info.GetNode()->arg_type_ = ARG_VECTOR_REDUCTION_LAST_AXIS;

  PatternResult result;
  result.dst_info_list = {dst_info};
  result.src_info_list = {src_info};
  result.for_info = for_info;
  result.arg_info = arg_info;

  return result;
}

void LastAxisReduceInsnArgsCalculator::CalcParams() {
  // check shape len
  if (dst_info->shape_.empty() || src_info->shape_.empty()) {
    LOG(FATAL) << "CCE Vector Insn Error: dst_buffer and src_buffer can not be scalar, should keep len(shape) > 0.";
  }
  // check data type
  if (dst_info->dtype_ != src_info->dtype_) {
    LOG(FATAL) << "CCE Vector Insn Error: dst_buffer and src_buffer can not be different data type.";
  }

  params.src_var = src_info->var_;
  params.block_size = GetUbBlkSize(dst_info->dtype_);
  params.last_dim_shape = GetInt32Const(GetItem(src_info->shape_, -1));
  params.vec_max_len = GetVecMaxLen(dst_info->dtype_);
  CHECK_NE(params.block_size, 0);
  CHECK_NE(params.vec_max_len, 0);
}
/// Generete info list for bisection intrin
/// \param dst_info_list
/// \param src_info_list
/// \param for_info
/// \param if_info
/// \param last_axis
/// \param postfix
/// \return
BisectionInfoWrapper SeparateComInfoToBisectionInfoList(const StmtInfoList &dst_info_list,
                                                        const StmtInfoList &src_info_list, const StmtInfo &for_info,
                                                        StmtInfo &if_info, bool last_axis, int postfix = 0) {
  CHECK_EQ(dst_info_list.size(), 1);
  CHECK_EQ(src_info_list.size(), 2);
  BisectionInfoWrapper wrapper;
  // Separate com_info and for_info
  int compare_idx = 1;
  int var_idx = -1;
  var_idx = GetBisectionReductionIdx(dst_info_list, src_info_list, compare_idx);
  StmtStoreInfo dst_info = dst_info_list[0];
  CHECK_GE(compare_idx, 0);
  StmtStoreInfo src_info1 = src_info_list[compare_idx];

  Var reduce_var = GetItem(src_info1->var_, var_idx);
  int stride_len = GetInt32Const(GetItem(src_info1->strides_, var_idx));
  size_t for_idx = 0;
  bool suc = GetIndexOfElement(for_info.vars_, VarExpr(reduce_var), for_idx);
  CHECK(suc);
  auto exist_for = GetItem(for_info.ops_, for_idx).as<For>();
  CHECK(exist_for);
  int extent = GetInt32Const(exist_for->extent);

  int simd_len = 1;
  const std::string un_def_var = "un_def_var";
  Var simd_var = Var("un_def_var");
  CHECK_GT(src_info1->strides_.size(), 0);
  CHECK_EQ(src_info1->var_.size(), src_info1->strides_.size());
  for (size_t i = 0; i <= src_info1->strides_.size() - 1; i++) {
    if (GetInt32Const(src_info1->strides_[i]) == 1) {
      simd_var = src_info1->var_[i];
      size_t simd_for_idx = 0;
      bool suc = GetIndexOfElement(for_info.vars_, VarExpr(simd_var), simd_for_idx);
      CHECK(suc);
      auto simd_for = GetItem(for_info.ops_, simd_for_idx).as<For>();
      CHECK(simd_for);
      simd_len = GetInt32Const(simd_for->extent);
    }
  }

  int block_unit = GetUbBlkSize(src_info1->dtype_);
  int last_dim_len = ((simd_len - 1) / block_unit + 1) * block_unit;

  Var bisec_var;
  Buffer bisec_buffer;
  std::string bisec_pre_header = "bisec";
  std::string bisec_name = bisec_pre_header + "_local_UB";
  if (postfix > 0) {
    bisec_name = bisec_name + "_" + std::to_string(postfix);
  }

  int vec_max_len = GetVecMaxLen(dst_info->dtype_);
  CHECK_NE(vec_max_len, 0);
  std::vector<int> pow2_list = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
  int origin_len = extent;
  for (int i : pow2_list) {
    if (extent <= i) {
      extent = i / 2;
      break;
    }
  }
  int prolog_len = origin_len - extent;

  src_info1.Print();
  auto src_vars = src_info1->var_;
  auto src_strides = src_info1->strides_;
  auto src_dims = src_info1->shape_;
  auto new_vars = src_info1->var_;
  auto new_strides = src_info1->strides_;
  auto new_dims = src_info1->shape_;
  LOG(DEBUG) << "\nvar_idx:" << var_idx << "\n";
  var_idx = var_idx + src_info1->var_.size();
  if (var_idx != static_cast<int>(src_info1->var_.size()) - 1) {
    new_dims.Set(var_idx, extent);
    new_dims.Set(new_dims.size() - 1, last_dim_len);
    CHECK_GT(new_dims.size(), 1);
    new_strides.Set(new_strides.size() - 1, 1);
    for (int i = static_cast<int>(new_dims.size()) - 2; i >= 0; i--) {
      new_strides.Set(i, new_strides[i + 1] * new_dims[i + 1]);
    }
    new_dims.Set(new_dims.size() - 1, simd_len);
  } else {
    new_dims = {extent};
  }

  // copy data from origin buffer to new temp buffer
  Array<Expr> shape = new_dims;
  wrapper.original_shape_ = new_dims;
  bisec_var = Var(bisec_name, Handle());
  bisec_buffer = BufferNode::make(bisec_var, dst_info->dtype_, shape, Array<Expr>(), Expr(), bisec_name, SCOPE_UBUF, 0,
                                  0, BufferType::kDefault);
  // Need to copy input to bisect buffer
  StmtStoreInfo copy_dst_info{src_info1.Copy()};
  StmtStoreInfo copy_src_info{src_info1.Copy()};
  StmtInfoList src_list = {copy_src_info};

  auto for_tmp_info = for_info.Copy();
  auto new_for = GetItem(for_tmp_info.ops_, for_idx).as<For>();
  CHECK(new_for);
  SetItem(for_tmp_info.ops_, static_cast<int>(for_idx),
          For::make(new_for->loop_var, new_for->min, extent, new_for->for_type, new_for->device_api, new_for->body));
  ReplaceVarWithNewForInfo(copy_dst_info, for_info, for_tmp_info);
  ReplaceVarWithNewForInfo(copy_src_info, for_info, for_tmp_info);
  SetItem(copy_src_info.GetNode()->shape_, var_idx, Expr(extent));
  SetItem(copy_dst_info.GetNode()->shape_, var_idx, Expr(extent));
  SetItem(copy_dst_info.GetNode()->strides_, var_idx, Expr(last_dim_len));
  if (simd_var->name_hint != un_def_var) {
    copy_dst_info.GetNode()->index_ = 0;
    for (size_t i = 0; i <= new_vars.size() - 1; i++) {
      copy_dst_info.GetNode()->index_ += new_vars[i] * new_strides[i];
    }
  } else {
    copy_dst_info.GetNode()->index_ = last_dim_len * reduce_var;
  }
  copy_dst_info.GetNode()->elem_offset_ = 0;
  copy_dst_info.GetNode()->name_ = bisec_name;
  copy_dst_info.GetNode()->buffer_ = bisec_buffer;
  copy_dst_info.GetNode()->data_ = bisec_var;
  copy_dst_info.GetNode()->strides_ = new_strides;

  CompactComputationInfoList(copy_dst_info, src_list, if_info, for_tmp_info);
  wrapper.bisec_info_list_.emplace_back(StmtInfoList{copy_dst_info, copy_src_info});
  wrapper.for_info_list_.push_back(for_tmp_info);

  // Generate the vadd wrapper
  while (extent >= 0) {
    StmtStoreInfo dst_tmp_info = dst_info.Copy();
    StmtStoreInfo src_tmp_info0{src_info1.Copy()};
    StmtStoreInfo src_tmp_info1{src_info1.Copy()};
    auto for_tmp_info = for_info.Copy();
    int vadd_length = (prolog_len != 0) ? prolog_len : extent;

    if (extent > 0) {
      dst_tmp_info = src_info1.Copy();
      dst_tmp_info.GetNode()->data_alignment_ = simd_len;
      dst_tmp_info.GetNode()->name_ = bisec_name;
      dst_tmp_info.GetNode()->buffer_ = bisec_buffer;
      dst_tmp_info.GetNode()->data_ = bisec_var;
      dst_tmp_info.GetNode()->shape_ = new_dims;
      SetItem(dst_tmp_info.GetNode()->shape_, var_idx, Expr(vadd_length));
      dst_tmp_info.GetNode()->strides_ = new_strides;
      dst_tmp_info.GetNode()->var_ = new_vars;
      dst_tmp_info.GetNode()->index_ = 0;
      for (size_t i = 0; i <= new_vars.size() - 1; i++) {
        dst_tmp_info.GetNode()->index_ += new_vars[i] * new_strides[i];
      }
      if (prolog_len == 0) {
        src_tmp_info1 = dst_tmp_info.Copy();
        src_tmp_info1.GetNode()->index_ = dst_tmp_info.GetNode()->index_ + extent * last_dim_len;
      } else {
        SetItem(src_tmp_info1.GetNode()->shape_, var_idx, Expr(vadd_length));
        src_tmp_info1.GetNode()->index_ += extent * stride_len;
      }
    }

    src_tmp_info0 = dst_tmp_info.Copy();
    auto new_for = GetItem(for_tmp_info.ops_, for_idx).as<For>();
    CHECK(new_for);
    int temp_for_len = (vadd_length != 0) ? vadd_length : 1;
    SetItem(
      for_tmp_info.ops_, static_cast<int>(for_idx),
      For::make(new_for->loop_var, new_for->min, temp_for_len, new_for->for_type, new_for->device_api, new_for->body));

    if (extent == 0) {
      src_tmp_info1.GetNode()->name_ = bisec_name;
      src_tmp_info1.GetNode()->buffer_ = bisec_buffer;
      src_tmp_info1.GetNode()->data_ = bisec_var;
      if (simd_var->name_hint != un_def_var) {
        src_tmp_info1.GetNode()->shape_ = RemoveItemAtIndex(new_dims, var_idx);
        src_tmp_info1.GetNode()->strides_ = RemoveItemAtIndex(new_strides, var_idx);
        src_tmp_info1.GetNode()->var_ = RemoveItemAtIndex(new_vars, var_idx);
        src_tmp_info1.GetNode()->index_ = 0;
        for (size_t i = 0; i <= src_tmp_info1->var_.size() - 1; i++) {
          src_tmp_info1.GetNode()->index_ += src_tmp_info1->var_[i] * src_tmp_info1->strides_[i];
        }
      } else {
        src_tmp_info1.GetNode()->shape_ = dst_tmp_info->shape_;
        src_tmp_info1.GetNode()->strides_ = dst_tmp_info->strides_;
        src_tmp_info1.GetNode()->var_ = dst_tmp_info->var_;
        src_tmp_info1.GetNode()->index_ = dst_tmp_info->index_;
      }
    }

    ReplaceVarWithNewForInfo(dst_tmp_info, for_info, for_tmp_info);
    ReplaceVarWithNewForInfo(src_tmp_info0, for_info, for_tmp_info);
    ReplaceVarWithNewForInfo(src_tmp_info1, for_info, for_tmp_info);
    StmtInfoList src_list = {src_tmp_info0, src_tmp_info1};
    CompactComputationInfoList(dst_tmp_info, src_list, if_info, for_tmp_info);
    wrapper.for_info_list_.emplace_back(for_tmp_info);
    if (extent == 0) {
      // normally is bisect_tmp = bisect_tmp + bisect_tmp/src_tmp
      wrapper.bisec_info_list_.emplace_back(StmtInfoList{dst_tmp_info, dst_tmp_info, src_tmp_info1});
    } else {
      // normally is dst_tmp = dst_tmp + bisect_tmp
      wrapper.bisec_info_list_.emplace_back(StmtInfoList{dst_tmp_info, src_tmp_info0, src_tmp_info1});
    }

    if (extent == 0) {
      break;
    } else {
      extent = extent / 2;
    }
    prolog_len = 0;
  }
  // Generate arg_info
  for (size_t i = 0; i < wrapper.bisec_info_list_.size(); ++i) {
    auto info_list = wrapper.bisec_info_list_[i];
    auto new_for_info = wrapper.for_info_list_[i];

    ArgInfo arg_info;
    auto dst_list = GetRange(info_list, 0, 1);
    auto src_list = GetRange(info_list, 1, info_list.size() - 1);
    if (info_list.size() == 2) {
      std::string dma_intrin = INTRIN_NAME_COPY_UB_TO_UB;
      wrapper.dma_arg_info_map_ = GetDmaCopyInsnArgs(dma_intrin, dst_list, src_list, new_for_info);
    } else {
      // Bisect can't expand mask because it has inplace operation
      if (i != wrapper.bisec_info_list_.size() - 1) {
        // Last round dont need to add
        FillLastDim(dst_list, src_list, new_for_info);
      }
      std::string mode = GetBinaryVecMode(dst_list, src_list, "vadd", false);

      BinaryVecInsnArgsCalculator args_calculator =
        BinaryVecInsnArgsCalculator(dst_list, src_list, new_for_info, mode, "", false);
      PatternResult params = args_calculator.GetInsnArgs();

      arg_info = params.arg_info;
      dst_list = params.dst_info_list;
      src_list = params.src_info_list;
      new_for_info = params.for_info;
      wrapper.bisec_info_list_[i] = {dst_list[0], src_list[0], src_list[1]};
    }
    wrapper.arg_info_list_.push_back(arg_info);
    wrapper.for_info_list_[i] = new_for_info;
  }
  return wrapper;
}

/// Get CCE Binary Vector Insn Computation Info
/// \param stmt         -  operand stmt
/// \param intrin_name   -  vector intrin name
/// \param dst_info_list  -  dst computation info list
/// \param src_info_list  -  src computation info list
/// \param if_info       -  if info list
/// \param for_info      -  for info list
/// \return intrin args
ArgInfo GetBinaryVecInsnArgs(const Stmt &stmt, std::string intrin_name, StmtInfoList &dst_info_list,
                             StmtInfoList &src_info_list, StmtInfo &if_info, StmtInfo &for_info, bool enable_bisect) {
  // check intrin_name
  std::set<std::string> intrin_name_list = {"vadd", "vmax",  "vmin",   "vmul",   "vdiv",  "vsel",      "vsub", "vand",
                                            "vor",  "vaxpy", "argmax", "argmin", "vmadd", "vmaddrelu", "vmla"};
  if (intrin_name_list.count(intrin_name) == 0) {
    LOG(FATAL) << "Error: CCE Binary Vector Insn doesn't support the given intrin_name.";
  }

  // get and check dst and src
  GetCompactComputationInfo(stmt, dst_info_list, src_info_list, if_info, for_info, true);
  // For vmadd/vmaddrelu/vmla we only need first two src
  if (dst_info_list.size() != 1 || src_info_list.size() < 2) {
    LOG(FATAL) << "CCE Binary Vector Insn only support ONE dst and TWO srcs.";
  }
  src_info_list = GetRange(src_info_list, 0, 2);
  ArgInfo arg_info = ArgInfo(make_node<ArgInfoNode>());

  // detect vector op mode
  std::string mode = GetBinaryVecMode(dst_info_list, src_info_list, intrin_name, enable_bisect);
  if (mode == "reduce_last_axis") {
    size_t src_var_list_size = src_info_list[1]->var_.size();
    if (src_info_list[0]->var_.size() > src_info_list[1]->var_.size()) {
      src_var_list_size = src_info_list[0]->var_.size();
    }

    CHECK(src_var_list_size > 0) << "Error: src can not be a scalar.";
    if (src_var_list_size - dst_info_list[0]->var_.size() == 1) {
      arg_info.GetNode()->arg_type_ = ARG_VECTOR_REDUCTION_LAST_AXIS;
    } else {
      LOG(FATAL) << "Error: cannot support multi-last-axis reduction.";
    }
  } else if (mode == "reduce_bisection") {
    arg_info.GetNode()->arg_type_ = ARG_VECTOR_REDUCTION_BISECTION;
  } else {
    if (mode != "reduction" && mode != "broadcast") {
      FillLastDim(dst_info_list, src_info_list, for_info);
    }

    // vmax/vmin can't expand mask because it may introduce dirty data
    bool can_expand_mask = intrin_name != "vmax" && intrin_name != "vmin";

    BinaryVecInsnArgsCalculator args_calculator =
      BinaryVecInsnArgsCalculator(dst_info_list, src_info_list, for_info, mode, intrin_name, can_expand_mask);
    PatternResult params = args_calculator.GetInsnArgs();

    arg_info = params.arg_info;
    dst_info_list = params.dst_info_list;
    src_info_list = params.src_info_list;
    for_info = params.for_info;
    if (mode == "broadcast") {
      bool has_last_axis = false;
      if ((arg_info->body_arg_info_.defined() && arg_info->body_arg_info_->last_axis_info_.src_index_ != -1) ||
          (arg_info->tail_arg_info_.defined() && arg_info->tail_arg_info_->last_axis_info_.src_index_ != -1)) {
        has_last_axis = true;
      }

      if (has_last_axis && (intrin_name == "vadd" || intrin_name == "vmul")) {
        Array<NodeRef> stores;
        Array<NodeRef> loads;
        GetStoreAndLoads(stmt, stores, loads);
        intrin_name = intrin_name + "s";
        if (arg_info->body_arg_info_.defined()) {
          arg_info.GetNode()->body_arg_info_.GetNode()->last_axis_info_.intrin_name_ = intrin_name;
          arg_info.GetNode()->body_arg_info_.GetNode()->last_axis_info_.src_op_ =
            Downcast<Expr>(loads[arg_info->body_arg_info_->last_axis_info_.src_index_]);
        }
      }
    }
  }
  return arg_info;
}
}  // namespace akg