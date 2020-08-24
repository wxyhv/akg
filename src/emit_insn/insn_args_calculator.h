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

#ifndef EMIT_INSN_ARGS_CALCULATOR_H_
#define EMIT_INSN_ARGS_CALCULATOR_H_
namespace akg {
struct InsnArg {
  int dst_m0{1};
  int dst_m1{0};
  std::vector<Expr> src_m0_list;
  std::vector<Expr> src_m1_list;
  int repeat{1};
  int block_len{1};
  int block_num{1};
  int body_num{1};
  int tail_len{0};
  int dst_tail_offset{0};
  std::vector<Expr> src_tail_offset_list;
};

struct Meta {
  int block_size{0};
  int src_block_size{0};
  int dst_block_size{0};
  int block_offset{0};
  const float vec_rate{0.6};
  Type src_dtype;
  Type dst_dtype;
  Type dtype;
  bool cast{false};
  bool tail{false};
  bool scalar{false};
  bool liner{false};
  bool same_dst_src{false};
};

enum SplitStat { SUCCESS, NO_SPLIT, TAIL };

class InsnAxis {
 public:
  InsnAxis() = default;
  InsnAxis(const For *for_stmt, const Array<StmtStoreInfo> &info_list);
  virtual ~InsnAxis() = default;
  bool IsValid();
  void Print(const std::string &name = "");
  int min{0};
  int extent{0};
  Var var;
  int dst_stride{0};
  int src_stride{0};
  std::vector<int> src_stride_list;
  std::vector<int> stride_list;
  bool is_valid{true};

 private:
  Expr GetStrideByAxis(const Array<Var> &vars, const Array<Expr> &strides, Var obj_var);
};

using AxisIt = std::list<InsnAxis>::iterator;

std::list<InsnAxis> GetAxisList(const StmtInfo &for_info, const Array<StmtStoreInfo> &info_list);
Array<StmtStoreInfo> GetInfoList(const StmtStoreInfo &dst_info, const Array<StmtStoreInfo> &src_info_list);
int DivFloor(int a, int b);
void Print(std::list<InsnAxis> &axis_list);

class InsnArgsCalculator {
 public:
  InsnArgsCalculator(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list, const StmtInfo &for_info,
               const std::string &intrin_name);
  virtual ~InsnArgsCalculator() = default;

  PatternResult ExportResult();
  void CalAxis();
  void InitArg();

  virtual std::function<bool(const InsnAxis &)> GetStrideLambda();
  virtual std::function<bool(const InsnAxis &)> GetM0LimitLambda();
  virtual std::function<bool(const InsnAxis &)> GetM1LimitLambda();
  std::function<bool(const InsnAxis &)> GetBlockStrideLimitLambda();
  AxisIt GetAxisByLambda(const std::function<bool(const InsnAxis &)> &lambda);
  InsnAxis ExtractAxis(AxisIt &it);
  bool IsValid(AxisIt &it);
  AxisIt GetVecAxisIt();
  AxisIt GetBlockAxis();
  AxisIt GetRepeatAxisIt();
  InsnAxis GetRepeatAxis();

  void SetArgMask(int len);
  void SetArgBlockNum(int data_num);
  void SetArgBlockLen(int data_len);
  void SetArgM0(int dst_m0, int lsrc_m0, int rsrc_m0);
  void SetArgM1(int dst_m1, int lsrc_m1, int rsrc_m1);
  void SetArgRepeat(int repeat);
  void BlockAxisReduction();
  void RepeatAxisReduction();
  void CastCaseReduction();
  virtual void InsnReduction();

  StmtInfo ExportForInfo();
  Expr GetOffset(int stride_index);
  InsnAxis GetInvalidAxis();
  SplitStat SplitAxis(int extent, InsnAxis &axis);
  std::list<InsnAxis> axis_list_;

 protected:
  InsnArg arg_;
  Meta meta_;
  StmtInfoList dst_info_list_;
  StmtInfoList src_info_list_;
  StmtStoreInfo dst_info_;
  StmtInfo for_info_;
  const std::string intrin_name_;
  const int max_block_stride_{4};
};

class SingleVecInsnArgsCalculator : public InsnArgsCalculator {
 public:
  SingleVecInsnArgsCalculator(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list, const StmtInfo &for_info,
                        const std::string &intrin_name = "");
  virtual ~SingleVecInsnArgsCalculator() override = default;
  PatternResult GetInsnArgs();
};
class BinaryVecInsnArgsCalculator : public InsnArgsCalculator {
 public:
  BinaryVecInsnArgsCalculator(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list, const StmtInfo &for_info,
                        const std::string &mode, const std::string &intrin_name = "", bool expand_mask = true);
  virtual ~BinaryVecInsnArgsCalculator() override = default;
  PatternResult GetInsnArgs();
  std::function<bool(const InsnAxis &)> GetM0LimitLambda();
  std::function<bool(const InsnAxis &)> GetM1LimitLambda();
  void InsnReduction();

 private:
  std::string mode_;
  bool expand_mask_;
  InsnAxis vec_axis_;
};
class LastAxisReduceInsnArgsCalculator : InsnArgsCalculator{
 public:
  LastAxisReduceInsnArgsCalculator(const StmtStoreInfo &dst_info, const StmtStoreInfo &src_info, const StmtInfo &for_info,
                             const std::string &intrin_name)
      : InsnArgsCalculator({dst_info}, {src_info}, for_info, intrin_name),
        dst_info(dst_info),
        src_info(src_info),
        for_info(for_info),
        arg_info(ArgInfo(make_node<ArgInfoNode>())),
        body_args(VectorArgInfo()),
        tail_args(VectorArgInfo()),
        intrin_name(intrin_name) {}
  PatternResult GetInsnArgs();
  ~LastAxisReduceInsnArgsCalculator() = default;

 protected:
  Array<Var> GetPattern();
  PatternResult GenResult(const Array<Var> &elim_var);

 private:
  void CalcParams();

  struct Params {
    Array<Var> src_var;
    int block_size = 0;
    int vec_max_len = 0;
    int last_dim_shape = 0;
    Expr insn_offset_scale_factor;
  };
  StmtStoreInfo dst_info;
  StmtStoreInfo src_info;
  StmtInfo for_info;
  ArgInfo arg_info;
  VectorArgInfo body_args;
  VectorArgInfo tail_args;
  Array<VectorArgInfo> mix_vec_arg_list;
  std::string intrin_name;
  Params params;
};

BisectionInfoWrapper SeparateComInfoToBisectionInfoList(const StmtInfoList &dst_info_list,
                                                        const StmtInfoList &src_info_list, const StmtInfo &for_info,
                                                        StmtInfo &if_info, bool last_axis, int postfix);
                             
ArgInfo GetBinaryVecInsnArgs(const Stmt &stmt, std::string intrin_name, StmtInfoList &dst_info_list,
                             StmtInfoList &src_info_list, StmtInfo &if_info, StmtInfo &for_info,
                             bool enable_bisect = true);                         
}  // namespace akg
#endif