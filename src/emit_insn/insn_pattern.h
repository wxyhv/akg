/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef EMIT_INSN_INSN_PATTERN_H_
#define EMIT_INSN_INSN_PATTERN_H_

#include <string>

#include "common/array_api.h"
#include "tvm.h"
#include "ir_pass.h"
#include "cce_params.h"
#include "insn_info.h"

namespace akg {
bool IsScalarMode(const StmtInfoList &info_list);

void SplitAxis(StmtInfoList &com_info_list, StmtInfo &for_info, const Var &axis_var, int new_size);

struct PatternResult {
  ArgInfo arg_info;
  StmtInfoList dst_info_list;
  StmtInfoList src_info_list;
  StmtInfo for_info;
};

std::string GetSingleVecComputationInfo(const Stmt &stmt, const std::string &intrin_name,
                                        Array<StmtStoreInfo> &dst_info_list, Array<StmtStoreInfo> &src_info_list,
                                        StmtInfo &if_info, StmtInfo &for_info, bool need_compact = true);
                             
std::string GetBinaryVecMode(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list,
                             const std::string &intrin_name, bool enable_bisect = true);

ArgInfo GetMultiVecInsnArgs(StmtInfoList &dst_info_list, StmtInfoList &src_info_list, StmtInfo &for_info);

void FillLastDim(StmtInfoList &dst_info_list, StmtInfoList &src_info_list, StmtInfo &for_info);

void FillEmptyVar(Array<StmtStoreInfo> &info_list);

void CleanZeroStrides(StmtStoreInfo &info);

void CleanZeroStrides(Array<StmtStoreInfo> &info_list);

Array<Expr> GetVecMask(int data_len, int data_num, Type data_type, int begin = 0);

Map<std::string, Expr> GetDmaLoad2DInsnArgs(const std::string &intrin_name, const StmtInfoList &dst_info_list,
                                            const StmtInfoList &src_info_list, StmtInfo &for_info);

void GetDmaComputationInfo(const Stmt &stmt, StmtInfoList &dst_info_list, StmtInfoList &src_info_list,
                           StmtInfo &if_info, StmtInfo &for_info, std::string &dma_mode, std::string &intrin_name);

Map<std::string, Expr> GetDmaCopyInsnArgs(std::string &intrin_name, const StmtInfoList &dst_info_list,
                                          const StmtInfoList &src_info_list, StmtInfo &for_info);

Map<std::string, Expr> GetDmaCopyInsnArgs(std::string &intrin_name, const StmtInfoList &dst_info_list,
                                          const StmtInfoList &src_info_list, StmtInfo &for_info,
                                          Map<std::string, Expr> &ub_copy_pre, Map<std::string, Expr> &ub_copy_post);

void ReplaceVarWithNewForInfo(StmtStoreInfo &info, const StmtInfo &old_for_info, const StmtInfo &new_for_info);
extern const char *const DummyLastVar;
}  // namespace akg
#endif  // EMIT_INSN_INSN_PATTERN_H_
