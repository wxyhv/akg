/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "poly/gpu_isl_emitter.h"
#include "pass/utils.h"
#include "gpu_emit/emit_pass.h"
#include <sstream>
#include <algorithm>

namespace akg {
namespace ir {
namespace poly {
std::string GetDataTypeStr(Type t) {
  std::stringstream ss;
  ss << t;
  if (normal_data_type_adapter.find(ss.str()) != normal_data_type_adapter.end()) {
    return normal_data_type_adapter.at(ss.str());
  }
  if (unique_data_type_adapter.find(ss.str()) != unique_data_type_adapter.end()) {
    return unique_data_type_adapter.at(ss.str());
  }
  CHECK(false) << "unsupported data type!";
  return "";
}

Expr GpuIslEmitter::EmitLoad(const isl::ast_expr &expr, const Type type) {
  if (PRINT_EMITTER) {
    LOG(INFO) << ">>>>>>>>>>>>INPUT AST_NODE[LOAD]<<<<<<<<<<<<<<\n" << expr;
  }
  if (auto op = expr.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      CHECK(op.get_arg(0).as<isl::ast_expr_id>());
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();
      Array<Expr> local_args;
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        local_args.push_back(Interpret(op.get_arg(i)));
      }

      Tensor t = info_.FindTensor(var);
      auto call = Call::make(type, t->op->name, local_args, Call::CallType::Halide, t->op, t->value_index);
      if (PRINT_EMITTER) {
        LOG(INFO) << ">>>>>>>>>>>>OUTPUT STMT<<<<<<<<<<<<\n" << call;
      }
      return call;
    }
  }
  return Expr();
}

Expr GpuIslEmitter::EmitLoadAtomic(const isl::ast_expr &expr, const Type type) {
  if (PRINT_EMITTER) {
    LOG(INFO) << ">>>>>>>>>>>>INPUT AST_NODE[LOADAtomic]<<<<<<<<<<<<<<\n" << expr;
  }
  if (auto op = expr.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      CHECK(op.get_arg(0).as<isl::ast_expr_id>());
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();
      Array<Expr> local_args;
      reduce_info_.output_promoted_tensor_indexs_for_atomic_.clear();
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        Expr arg = Interpret(op.get_arg(i));
        local_args.push_back(arg);
        std::stringstream ss;
        ss << arg;
        std::string arg_name = ss.str();
        if (arg_name != USELESS_INDEX) {
          reduce_info_.output_promoted_tensor_indexs_for_atomic_.push_back(arg_name);
        }
      }

      Tensor t = info_.FindTensor(var);
      auto call = Call::make(type, t->op->name, local_args, Call::CallType::Halide, t->op, t->value_index);
      if (PRINT_EMITTER) {
        LOG(INFO) << ">>>>>>>>>>>>OUTPUT STMT<<<<<<<<<<<<\n" << call;
      }
      return call;
    }
  }
  return Expr();
}

Stmt GpuIslEmitter::EmitRead(const isl::ast_node_user &node) {
  isl::id node_id = node.get_annotation();
  isl::pw_multi_aff iterator_map = node_info_map_.at(node_id).iterator_map;
  isl::pw_multi_aff hoisted = iterator_map.range_factor_range();
  isl::pw_multi_aff original = iterator_map.range_factor_domain().range_factor_range();

  isl::id original_tensor = original.get_tuple_id(isl_dim_out);

  auto build = node_info_map_.at(node_id).build;
  auto lhs = build.access_from(isl::multi_pw_aff(hoisted));
  auto rhs = build.access_from(isl::multi_pw_aff(original));

  Type type = info_.GetDtypeOf(rhs);
  if (auto op = lhs.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      Expr value = EmitLoad(rhs, type);
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();

      Array<Expr> local_args;
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        local_args.push_back(Interpret(op.get_arg(i)));
      }

      Tensor t = info_.FindTensor(var);
      CHECK(t.defined());
      if (info_.cube_info_.IsIm2col() && !info_.analysis_result_.GetUpdateTensor().empty()) {
        return Provide::make(info_.analysis_result_.GetUpdateTensor()[0]->op, 0, value, local_args);
      }
      return Provide::make(t->op, 0, value, local_args);
    }
  }
  return Stmt();
}

std::string SimplifyName(std::string input) {
  auto pos_local = input.find(LOCAL_SUFFIX);
  auto pos_shared = input.find(SHARE_SUFFIX);
  std::string res = input;
  if (pos_local != std::string::npos) {
    res = input.substr(0, pos_local);
  }
  if (pos_shared != std::string::npos) {
    res = res.substr(0, pos_shared);
  }
  return res;
}

Stmt GpuIslEmitter::EmitReadCore(const isl::ast_node_user &node) {
  isl::id node_id = node.get_annotation();
  isl::pw_multi_aff iterator_map = node_info_map_.at(node_id).iterator_map;
  isl::pw_multi_aff hoisted = iterator_map.range_factor_range();
  isl::pw_multi_aff original = iterator_map.range_factor_domain().range_factor_range();

  isl::id original_tensor = original.get_tuple_id(isl_dim_out);

  auto build = node_info_map_.at(node_id).build;
  auto lhs = build.access_from(isl::multi_pw_aff(hoisted));
  auto rhs = build.access_from(isl::multi_pw_aff(original));

  Type type = info_.GetDtypeOf(rhs);
  if (auto op = lhs.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      Expr value = EmitLoad(rhs, type);
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();

      Array<Expr> local_args;
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        local_args.push_back(Interpret(op.get_arg(i)));
      }

      Tensor t = info_.FindTensor(var);
      CHECK(t.defined());
      Stmt s = Provide::make(t->op, 0, value, local_args);

      auto op_new = s.as<Provide>();
      CHECK(op_new);
      const Call *call_value = op_new->value.as<Call>();
      CHECK(call_value != nullptr) << "Can only load fragment from a buffer";

      auto left_expr = MakeLeftCallFromProvide(op_new);
      auto left_call = left_expr.as<Call>();
      CHECK(left_call != nullptr) << "make right part call failed!";

      auto it = tensor_core_info_.strides_.find(call_value->name);
      CHECK(it != tensor_core_info_.strides_.end()) << "Cannot find stride for " << call_value->name;
      auto strides = it->second;
      CHECK_GE(strides.size(), 2);
      Expr stride = strides[strides.size() - 2];

      std::string call_name = op_new->func->func_name();
      Expr src = Call::make(call_value->type, "&", {op_new->value}, Call::Extern);

      Expr matrix_major;
      auto iter2 = tensor_core_info_.matrix_major_.find(SimplifyName(call_name));
      CHECK(iter2 != tensor_core_info_.matrix_major_.end()) << "Can not determine matrix major for " << call_name;
      if (iter2->second == COL_MAJOR) {
        matrix_major = StringImm::make(COL_MAJOR);
      } else if (iter2->second == ROW_MAJOR) {
        matrix_major = StringImm::make(ROW_MAJOR);
      } else {
        LOG(FATAL) << "invalid matrix major for " << call_name;
      }

      NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
      EmitTensorCoreHelper helper(tensor_core_info_);
      helper.SetDataForLoad(src, stride, matrix_major, left_call, op_new, buffer_node);
      return helper.MakeLoadTransform();
    }
  }
  return Stmt();
}

Expr GpuIslEmitter::MakeLeftCallFromProvide(const Provide *op) {
  std::string name = op->func->func_name();
  Type type = GetTypeOfTensor(name);
  Expr dst = Call::make(type, name, op->args, Call::Halide, op->func, 0);
  return dst;
}

Type GpuIslEmitter::GetTypeOfTensor(std::string name) {
  auto binds = info_.user_config_.GetBind();

  for (auto &i : binds) {
    if (!i.first.defined()) continue;
    if (!i.second.defined()) continue;

    if (name == i.first->op->name) {
      auto b = i.second;
      return b->dtype;
    }
  }

  CHECK(false) << "Can not find type of tensor " << name;
  return Type();
}

Stmt GpuIslEmitter::EmitWrite(const isl::ast_node_user &node) {
  auto node_id = node.get_annotation();
  CHECK_GT(node_info_map_.count(node_id), 0);
  auto iterator_map = node_info_map_.at(node_id).iterator_map;
  auto hoisted = iterator_map.range_factor_range();
  auto original = iterator_map.range_factor_domain().range_factor_range();

  auto build = node_info_map_.at(node_id).build;
  auto rhs = build.access_from(isl::multi_pw_aff(hoisted));
  auto lhs = build.access_from(isl::multi_pw_aff(original));
  Type type = info_.GetDtypeOf(lhs);

  if (auto op = lhs.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      Expr value = EmitLoad(rhs, type);
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();

      Array<Expr> local_args;
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        local_args.push_back(Interpret(op.get_arg(static_cast<int>(i))));
      }

      Tensor t = info_.FindTensor(var);
      CHECK(t.defined());

      return Provide::make(t->op, 0, value, local_args);
    }
  }
  return Stmt();
}

Stmt GpuIslEmitter::EmitWriteCore(const isl::ast_node_user &node) {
  auto node_id = node.get_annotation();
  CHECK_GT(node_info_map_.count(node_id), 0);
  auto iterator_map = node_info_map_.at(node_id).iterator_map;
  auto hoisted = iterator_map.range_factor_range();
  auto original = iterator_map.range_factor_domain().range_factor_range();

  auto build = node_info_map_.at(node_id).build;
  auto rhs = build.access_from(isl::multi_pw_aff(hoisted));
  auto lhs = build.access_from(isl::multi_pw_aff(original));
  Type type = info_.GetDtypeOf(lhs);

  if (auto op = lhs.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      Expr value = EmitLoad(rhs, type);
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();

      Array<Expr> local_args;
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        local_args.push_back(Interpret(op.get_arg(static_cast<int>(i))));
      }

      Tensor t = info_.FindTensor(var);
      CHECK(t.defined());

      Stmt s = Provide::make(t->op, 0, value, local_args);

      auto op = s.as<Provide>();
      CHECK(op);

      auto lh_expr = MakeLeftCallFromProvide(op);
      auto lh_call = lh_expr.as<Call>();
      CHECK(lh_call != nullptr) << "make right part call failed!";

      auto it = tensor_core_info_.strides_.find(lh_call->name);
      CHECK(it != tensor_core_info_.strides_.end()) << "Cannot find stride for " << lh_call->name;
      auto strides = it->second;
      CHECK_GE(strides.size(), 2);
      Expr stride = strides[strides.size() - 2];

      Expr dst = lh_expr;
      dst = Call::make(Handle(), "&", {dst}, Call::Extern);

      auto call = op->value.as<Call>();
      NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
      EmitTensorCoreHelper helper(tensor_core_info_);
      helper.SetDataForStore(dst, stride, call, buffer_node);
      return helper.MakeStoreTransform();
    }
  }
  return Stmt();
}

Stmt GpuIslEmitter::EmitWriteAtomic(const isl::ast_node_user &node) {
  auto node_id = node.get_annotation();
  CHECK_GT(node_info_map_.count(node_id), 0);
  auto iterator_map = node_info_map_.at(node_id).iterator_map;
  auto hoisted = iterator_map.range_factor_range();
  auto original = iterator_map.range_factor_domain().range_factor_range();

  auto build = node_info_map_.at(node_id).build;
  auto rhs = build.access_from(isl::multi_pw_aff(hoisted));
  auto lhs = build.access_from(isl::multi_pw_aff(original));

  auto opl = lhs.as<isl::ast_expr_op>();
  reduce_info_.output_tensor_name_ = opl.get_arg(0).as<isl::ast_expr_id>().get_id().name();
  auto opr = rhs.as<isl::ast_expr_op>();
  reduce_info_.output_promoted_tensor_name_for_atomic_ = opr.get_arg(0).as<isl::ast_expr_id>().get_id().name();
  reduce_info_.atomic_tensors_.insert(reduce_info_.output_promoted_tensor_name_for_atomic_);

  Type type = info_.GetDtypeOf(lhs);
  reduce_info_.output_tensor_data_type_ = GetDataTypeStr(type);
  CHECK(!reduce_info_.output_tensor_data_type_.empty()) << "output tensor type should not be empty!";

  if (auto op = lhs.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      Expr value = EmitLoadAtomic(rhs, type);
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();

      Array<Expr> local_args;
      reduce_info_.output_tensor_indexs_.clear();
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        Expr arg = Interpret(op.get_arg(static_cast<int>(i)));
        local_args.push_back(arg);

        std::stringstream ss;
        ss << arg;
        std::string arg_name = ss.str();
        if (arg_name != USELESS_INDEX) {
          reduce_info_.output_tensor_indexs_.push_back(arg_name);
        }
      }

      Tensor t = info_.FindTensor(var);
      CHECK(t.defined());

      return Provide::make(t->op, 0, value, local_args);
    }
  }
  return Stmt();
}

Stmt GpuIslEmitter::EmitSync() {
  return Evaluate::make(Call::make(Int(32), STORAGE_SYNC, {StringImm::make(SYNC_SCOP_SHARED)}, Call::Intrinsic));
}

Stmt GpuIslEmitter::EmitReduceInit(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();

  CHECK(!reduce_info_.scalar_tensor_info_.empty()) << "scalar tensor info should not be empty!";

  std::string temp_name = stmt_id.name();

  std::vector<std::string> strs = common::Split(temp_name, "_");
  CHECK_EQ(strs.size(), REDUCE_FLAG_SIZE) << "red init format is not right!.";

  std::string scalar_info = reduce_info_.reduce_data_type_ + " ";
  scalar_info += reduce_info_.scalar_tensor_info_;

  std::string stmt_name = strs[REDUCE_FLAG_STMT_PREFIX_POS] + "_" + strs[REDUCE_FLAG_STMT_NUM_POS];
  std::string init_value = "";
  for (auto it : info_.analysis_result_.GetReduceInitValueMap()) {
    if (it.first.name() == stmt_name) {
      std::stringstream ss;
      ss << it.second;
      init_value = ss.str();
      break;
    }
  }

  if (IsEndsWith(init_value, "h") || IsEndsWith(init_value, "f")) {
    init_value.pop_back();
  }

  std::string flag = "bool";
  if (init_value.find(flag) != std::string::npos) {
    init_value.replace(init_value.find(flag), flag.size(), "signed char");
  }

  CHECK(!init_value.empty()) << "init value should not be empty!";

  scalar_info += " = ";
  scalar_info += init_value;
  scalar_info += ";";

  int size = info_.user_config_.GetThreadConfig()->GetX().second * info_.user_config_.GetThreadConfig()->GetY().second;
  CHECK_NE(size, 0) << "Buffer size should not be zero!"
                    << "\n";
  std::string shared_info = SHARED_MEMORY_PREFIX;
  shared_info += " ";
  shared_info += reduce_info_.reduce_data_type_;
  shared_info += " ";
  CHECK(!reduce_info_.shared_compute_info_.empty()) << "shared compute info should not be empty!";
  shared_info += reduce_info_.shared_compute_info_;
  shared_info += "[";
  shared_info += std::to_string(size);
  shared_info += "];";
  std::string map_info = "";
  CHECK(!reduce_info_.promoted_tensor_name_for_reduce_.empty())
    << "promoted tensor for reduce name should not be empty!";
  map_info += reduce_info_.promoted_tensor_name_for_reduce_;
  map_info += "|";
  map_info += reduce_info_.scalar_tensor_info_;

  Stmt stmt = Evaluate::make(Call::make(
    Int(32), REDUCE, {StringImm::make(stmt_id.name()), StringImm::make(scalar_info), StringImm::make(shared_info)},
    Call::Intrinsic));
  stmt = AttrStmt::make(Expr("INFO"), TENSOR_MAP_INFO_FLAG, StringImm::make(map_info), stmt);
  return stmt;
}

Stmt GpuIslEmitter::EmitReduceUpdate(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();

  std::string temp_name = stmt_id.name();

  std::vector<std::string> strs = common::Split(temp_name, "_");
  CHECK_EQ(strs.size(), REDUCE_FLAG_SIZE) << "red update format is not right!.";

  reduce_info_.scalar_tensor_info_ = SCALAR_TENSOR_PREFIX;
  reduce_info_.scalar_tensor_info_ += strs[REDUCE_FLAG_REDUCE_INDEX];
  reduce_info_.reduce_stmt_index_ = strs[REDUCE_FLAG_REDUCE_INDEX];
  reduce_info_.reduce_op_.clear();
  if (AkgSupportedReduceOp.count(strs[REDUCE_FLAG_TYPE_POS])) {
    reduce_info_.reduce_op_ = AKG_REDUCE_LIB_SPACE;
    reduce_info_.reduce_op_ += "::";
    reduce_info_.reduce_op_ += strs[REDUCE_FLAG_TYPE_POS];
  }
  CHECK(!reduce_info_.reduce_op_.empty()) << "reduce op should not be empty!";
  std::string stmt_name = strs[REDUCE_FLAG_STMT_PREFIX_POS] + "_" + strs[REDUCE_FLAG_STMT_NUM_POS];
  std::string data_type = "";
  for (auto it : info_.analysis_result_.GetReduceWriteDtypeMap()) {
    if (it.first.name() == stmt_name) {
      data_type = GetDataTypeStr(it.second);
      break;
    }
  }
  CHECK(!data_type.empty()) << "data type should not be empty!";
  reduce_info_.reduce_data_type_ = data_type;

  std::string origin_tensor_name = "";
  for (auto it : info_.analysis_result_.GetReduceStatementWriteTensorMap()) {
    if (it.first.name() == stmt_name) {
      origin_tensor_name = it.second;
      break;
    }
  }
  CHECK(!origin_tensor_name.empty()) << "origin_tensor_name should not be empty!";

  for (const auto &buffer : info_.analysis_result_.active_buffer_footprints_) {
    auto cluster_id = buffer.second.cluster_id;
    auto buf_def = info_.analysis_result_.GetBufferDefInfo(cluster_id);
    if (buf_def.tensor_id.name() == origin_tensor_name) {
      reduce_info_.promoted_tensor_name_for_reduce_ = cluster_id.name();
      break;
    }
  }

  CHECK(!reduce_info_.promoted_tensor_name_for_reduce_.empty())
    << "promoted tensor name for reduce  should not be empty!";

  MakePromotedTensorInfoForReduce();
  std::string info = PrepareAkgReduceInfo();

  Stmt stmt = Evaluate::make(
    Call::make(Int(32), REDUCE, {StringImm::make(stmt_id.name()), StringImm::make(info)}, Call::Intrinsic));

  CHECK(!reduce_info_.promoted_tensor_name_for_reduce_.empty())
    << "promoted tensor for reduce name should not be empty!";
  Expr tensor_name = StringImm::make(reduce_info_.promoted_tensor_name_for_reduce_);
  stmt = AttrStmt::make(Expr("INFO"), TENSOR_INDEX_MODIFY_FLAG, tensor_name, stmt);
  stmt = AttrStmt::make(Expr("INFO"), REDUCE_LIB_TYPE_FLAG, info_.user_config_.GetReduceLibType(), stmt);
  return stmt;
}

std::string GpuIslEmitter::GetTheIndexOfPromotedTensor(std::string s) {
  CHECK(!reduce_info_.promoted_tensor_name_for_reduce_.empty())
    << "promoted tensor for reduce name should not be empty!";
  std::string name = reduce_info_.promoted_tensor_name_for_reduce_;
  int len = name.size();
  std::string::size_type n1 = s.find(name);
  if (n1 == std::string::npos) {
    return "";
  }
  int start = n1 + len + 1;

  std::string::size_type n2 = s.find("=");
  if (n2 == std::string::npos) {
    return "";
  }
  int end = n2 - 2;
  int sub_len = end - start;

  std::string index = s.substr(start, sub_len);
  auto itor = remove_if(index.begin(), index.end(), ::isspace);
  index.erase(itor, index.end());

  return index;
}

Stmt GpuIslEmitter::EmitReduceArea(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  stmt_id_ = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  node_id_ = node.get_annotation();
  const Node *stmt_node = info_.analysis_result_.GetStatementMap().at(stmt_id_);
  CHECK(stmt_node);
  // compute VarMap to replace old iterators
  auto build = node_info_map_.at(node_id_).build;
  auto tuple = info_.analysis_result_.GetOperatorDomainMap().at(stmt_id_).tuple;
  auto iterator_map = node_info_map_.at(node_id_).iterator_map;

  var_map_.clear();
  for (unsigned int i = 0; i < tuple.size(); ++i) {
    isl::id isl_old_iter = tuple.get_id(i);
    auto isl_expr = build.expr_from(iterator_map.get_pw_aff(i));
    Expr halide_new_iter = Interpret(isl_expr);
    var_map_.emplace(isl_old_iter, halide_new_iter);
  }

  Stmt stmt = EmitUserStmtContent(stmt_node);
  reduce_info_.reduce_stmt_[reduce_info_.promoted_tensor_name_for_reduce_] = stmt;
  std::stringstream ss;
  ss << stmt;
  std::string stmt_str = ss.str();
  std::string index_str = GetTheIndexOfPromotedTensor(stmt_str);
  if (index_str == "") {
    return stmt;
  }

  std::vector<std::string> args = common::Split(index_str, ",");
  reduce_info_.promoted_tensor_indexs_for_reduce_[reduce_info_.promoted_tensor_name_for_reduce_].clear();
  for (auto s : args) {
    if (s != USELESS_INDEX) {
      reduce_info_.promoted_tensor_indexs_for_reduce_[reduce_info_.promoted_tensor_name_for_reduce_].push_back(s);
    }
  }

  Tensor t = info_.FindTensor(reduce_info_.promoted_tensor_name_for_reduce_);
  CHECK(t.defined());
  std::vector<std::string> shapeInfo;
  for (auto &e : t->shape) {
    std::stringstream ss;
    ss << e;
    if (ss.str() != USELESS_SHAPE_SIZE) {
      shapeInfo.push_back(ss.str());
    }
  }
  reduce_info_.promoted_tensor_shape_for_reduce_[reduce_info_.promoted_tensor_name_for_reduce_] = shapeInfo;

  return stmt;
}

Stmt GpuIslEmitter::EmitUserStmtCore(const isl::ast_node_user &node) {
  if (tensor_core_info_.matrix_info_[MMA_SYNC]) {
    return EmitUserStmtCoreSync(node);
  }
  return Stmt();
}

Stmt GpuIslEmitter::EmitUserStmtCoreSync(const isl::ast_node_user &node) {
  static int serial_number = MMA_SYNC_STMT_SERIAL;
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  stmt_id_ = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  node_id_ = node.get_annotation();
  const Node *stmt_node = info_.analysis_result_.GetStatementMap().at(stmt_id_);
  CHECK(stmt_node);
  // compute VarMap to replace old iterators
  auto build = node_info_map_.at(node_id_).build;
  auto tuple = info_.analysis_result_.GetOperatorDomainMap().at(stmt_id_).tuple;
  auto iterator_map = node_info_map_.at(node_id_).iterator_map;

  var_map_.clear();
  for (unsigned int i = 0; i < tuple.size(); ++i) {
    isl::id isl_old_iter = tuple.get_id(i);
    auto isl_expr = build.expr_from(iterator_map.get_pw_aff(i));
    Expr halide_new_iter = Interpret(isl_expr);
    var_map_.emplace(isl_old_iter, halide_new_iter);
  }

  Stmt s = EmitUserStmtContent(stmt_node);

  auto SplitCast = [](const Type &tensor_type, const Expr &tensor) -> Expr {
    if (tensor.as<Cast>() == nullptr) {
      return tensor;
    } else if (tensor.as<Cast>()->type == tensor_type) {
      return tensor.as<Cast>()->value;
    } else {
      return Expr();
    }
  };

  if (serial_number == MMA_SYNC_STMT_SERIAL) {
    serial_number = MMA_FILL_STMT_SERIAL;
    auto op = s.as<Provide>();
    auto left_expr = MakeLeftCallFromProvide(op);
    Type type = GetTypeOfTensor(op->func->func_name());
    auto *add = op->value.as<Add>();
    CHECK(add) << "format error of bmm";
    auto mul = SplitCast(type, add->b).as<Mul>();
    CHECK(mul) << "format error of bmm";

    auto load_a_expr = SplitCast(type, mul->a);
    auto load_b_expr = SplitCast(type, mul->b);

    Expr a = load_a_expr;
    Expr b = load_b_expr;
    Expr c = left_expr;

    NodePtr<BufferNode> buffer_node_a = make_node<BufferNode>();
    NodePtr<BufferNode> buffer_node_b = make_node<BufferNode>();
    NodePtr<BufferNode> buffer_node_c = make_node<BufferNode>();

    EmitTensorCoreHelper helper(tensor_core_info_);
    helper.SetDataForSync(a, b, c, buffer_node_a, buffer_node_b, buffer_node_c);
    return helper.MakeSyncTransform();
  } else if (serial_number == MMA_FILL_STMT_SERIAL) {
    serial_number = MMA_SYNC_STMT_SERIAL;
    auto op = s.as<Provide>();
    auto left_expr = MakeLeftCallFromProvide(op);
    auto left_call = left_expr.as<Call>();
    CHECK(left_call != nullptr) << "make right part call failed";

    if (op->value.as<FloatImm>() != nullptr || op->value.as<IntImm>() != nullptr) {
      NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
      EmitTensorCoreHelper helper(tensor_core_info_);
      helper.SetDataForFill(op, left_call, buffer_node);
      return helper.MakeFillTransform();
    } else {
      CHECK(false) << "mma init stmt format error";
    }
  }

  return Stmt();
}

Stmt GpuIslEmitter::EmitStmt(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  auto node_id = node.get_annotation();

  if (info_.IsRead(stmt_id)) {
    Stmt s;
    is_sync_before_ = false;
    if (tensor_core_info_.core_area_) {
      s = EmitReadCore(node);
    } else {
      s = EmitRead(node);
      s = AttrStmt::make(Expr(""), GMREAD_FLAG, StringImm::make(GMREAD_FLAG), s);
    }
    if (PRINT_EMITTER) {
      LOG(INFO) << ">>>>>>>>>>>>INPUT AST_NODE[READ]<<<<<<<<<<<<<<\n" << node;
      LOG(INFO) << ">>>>>>>>>>>>OUTPUT STMT<<<<<<<<<<<<\n" << s;
    }
    return s;
  } else if (info_.IsWrite(stmt_id)) {
    auto s = Stmt();
    if (info_.IsGMWrite(stmt_id)) {
      if (tensor_core_info_.core_area_) {
        is_sync_before_ = false;
        s = EmitWriteCore(node);
        return s;
      }
      auto iterator_map = node_info_map_.at(node_id).iterator_map;
      auto original = iterator_map.range_factor_domain().range_factor_range();
      auto srcid = original.get_tuple_id(isl_dim_out);
      if (!NoNeedToEmitForTempTensor(srcid)) {
        if (reduce_info_.is_atomic) {
          s = EmitWriteAtomic(node);
          MakeOutputTensorInfo();
          MakeOutputPromotedTensorInfoForAtomic();
          std::string info = PrepareAkgAtomicReturnInfo();
          s = AttrStmt::make(Expr("INFO"), GM_WRITE_FLAG, Expr("True"), s);
          Stmt ato_info = EmitAkgAtomicReturnInfo(s, info);
          is_sync_before_ = false;
          s = Block::make(s, ato_info);
        } else {
          is_sync_before_ = false;
          s = EmitWrite(node);
        }
      }
    } else {
      is_sync_before_ = false;
      if (tensor_core_info_.core_area_) {
        s = EmitWriteCore(node);
      } else {
        s = EmitWrite(node);
      }
    }
    if (PRINT_EMITTER) {
      LOG(INFO) << ">>>>>>>>>>>>INPUT AST_NODE[WRITE]<<<<<<<<<<<<<<\n" << node;
      LOG(INFO) << ">>>>>>>>>>>>OUTPUT STMT<<<<<<<<<<<<\n" << s;
    }
    return s;
  } else if (info_.IsSync(stmt_id)) {
    if (is_sync_before_) {
      return Stmt();
    }
    Stmt s = EmitSync();
    is_sync_before_ = true;
    return s;
  } else if (info_.IsReduceInit(stmt_id)) {
    is_sync_before_ = false;
    in_reduce_area_ = false;
    return EmitReduceInit(node);
  } else if (in_reduce_area_) {
    is_sync_before_ = false;
    return EmitReduceArea(node);
  } else if (info_.IsReduceUpdate(stmt_id)) {
    is_sync_before_ = false;
    in_reduce_area_ = true;
    return EmitReduceUpdate(node);
  } else {
    is_sync_before_ = false;
    Stmt s;
    if (tensor_core_info_.core_area_) {
      s = EmitUserStmtCore(node);
    } else {
      s = EmitUserStmt(node);
    }

    return s;
  }
}

std::string GpuIslEmitter::PrepareAkgReduceInfo() {
  auto thread_cfg = info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg) << "thread config is null.";
  auto block_cfg = info_.user_config_.GetBlockConfig();
  CHECK(block_cfg) << "thread config is null.";
  int tx = thread_cfg->GetX().second;
  int ty = thread_cfg->GetY().second;
  int by = block_cfg->GetY().second;
  std::string direction = info_.analysis_result_.GetReduceDirection();
  CHECK(!direction.empty()) << "direction should not be empty!";
  std::string direction_size = "";
  if (direction == X_DIRECTION) {
    direction_size = std::to_string(tx);
  } else {
    direction_size = std::to_string(ty);
  }

  std::string reduce_lib_namespace = "";
  std::string reduce_lib_name = "";
  if (info_.user_config_.GetReduceLibType() == REDUCE_LIB_TYPE_ORIGIN) {
    reduce_lib_namespace = AKG_REDUCE_LIB_SPACE;
    reduce_lib_name = AKG_REDUCE_LIB_NAME;
  } else if (info_.user_config_.GetReduceLibType() == REDUCE_LIB_TYPE_PARIS) {
    reduce_lib_namespace = PARIS_REDUCE_LIB_SPACE;
    reduce_lib_name = PARIS_REDUCE_LIB_NAME;
  } else {
    CHECK(false) << "reduce lib type is invalid!"
                 << "\n";
  }
  std::string ret = reduce_lib_namespace;
  ret += "::";
  ret += reduce_lib_name;
  ret += "<";
  ret += reduce_info_.reduce_data_type_;
  ret += ", ";
  std::string op = reduce_info_.reduce_op_;
  ret += op;
  ret += ", ";

  // modify for one dimension mapping
  ret += std::to_string(tx);
  ret += ", ";
  ret += std::to_string(ty);
  std::string reduce_type = "";
  if (by == 1 && ty == 1) {
    reduce_type = AKG_ALL_REDUCE;
  } else if (direction == X_DIRECTION) {
    reduce_type = AKG_X_REDUCE;
  } else {
    reduce_type = AKG_Y_REDUCE;
  }
  ret += ", ";
  ret += reduce_type;
  ret += ">(";
  ret += op;
  ret += "(), ";

  reduce_info_.shared_compute_info_ = SHARED_TENSOR_PREFIX;
  reduce_info_.shared_compute_info_ += reduce_info_.reduce_stmt_index_;
  CHECK(!reduce_info_.promoted_tensor_info_for_reduce_.empty())
    << "output promoted tensor info for reduce should not be empty!";
  ret += reduce_info_.promoted_tensor_info_for_reduce_;
  ret += ", ";
  ret += reduce_info_.shared_compute_info_;
  ret += ", ";
  ret += reduce_info_.scalar_tensor_info_;
  std::string size = std::to_string(tx);
  ret += ", ";
  ret += size;
  ret += ");";

  return ret;
}

std::string GpuIslEmitter::PrepareAkgAtomicReturnInfo() {
  std::string reduce_lib_namespace = "";
  std::string reduce_return_name = "";
  if (info_.user_config_.GetReduceLibType() == REDUCE_LIB_TYPE_ORIGIN) {
    reduce_lib_namespace = AKG_REDUCE_LIB_SPACE;
    reduce_return_name = AKG_REDUCE_RETURN_NAME;
  } else if (info_.user_config_.GetReduceLibType() == REDUCE_LIB_TYPE_PARIS) {
    reduce_lib_namespace = PARIS_REDUCE_LIB_SPACE;
    reduce_return_name = PARIS_REDUCE_RETURN_NAME;
  } else {
    CHECK(false) << "reduce lib type is invalid!"
                 << "\n";
  }
  std::string ret = "";
  ret += reduce_lib_namespace;
  ret += "::";
  ret += reduce_return_name;
  ret += "<";
  ret += reduce_info_.output_tensor_data_type_;
  ret += ", ";
  std::string op = reduce_info_.reduce_op_;
  ret += op;
  ret += ">(";
  CHECK(!reduce_info_.output_promoted_tensor_info_for_atomic_.empty())
    << "output promoted tensor info for atomic should not be empty!";
  ret += reduce_info_.output_promoted_tensor_info_for_atomic_;
  ret += ", ";
  CHECK(!reduce_info_.output_tensor_info_.empty()) << "output tensor info should not be empty!";
  ret += reduce_info_.output_tensor_info_;
  ret += ", ";
  ret += op;
  ret += "());";
  return ret;
}

void GpuIslEmitter::MakeOutputTensorInfo() {
  std::string ret = "";
  ret += "&";
  CHECK(!reduce_info_.output_tensor_name_.empty()) << "output tensor name should not be empty!";
  ret += reduce_info_.output_tensor_name_;

  Tensor t = info_.FindTensor(reduce_info_.output_tensor_name_);
  CHECK(t.defined());
  std::vector<std::string> shapeInfo;
  for (auto &e : t->shape) {
    std::stringstream ss;
    ss << e;
    if (ss.str() != USELESS_SHAPE_SIZE) {
      shapeInfo.push_back(ss.str());
    }
  }

  int size = reduce_info_.output_tensor_indexs_.size();
  if (size == 0) {
    ret += DEFAULT_TENSOR_INDEX;
  } else if (size == 1) {
    ret += "[";
    ret += reduce_info_.output_tensor_indexs_.at(0);
    ret += "]";
  } else {
    ret += "[";
    for (int i = 0; i < size - 1; ++i) {
      ret += reduce_info_.output_tensor_indexs_.at(i);
      for (int j = i + 1; j < size; ++j) {
        ret += "*";
        ret += "(";
        ret += shapeInfo.at(j);
        ret += ")";
      }
      ret += "+";
    }
    ret += reduce_info_.output_tensor_indexs_.at(size - 1);
    ret += "]";
  }

  reduce_info_.output_tensor_info_ = ret;
}

void GpuIslEmitter::MakeOutputPromotedTensorInfoForAtomic() {
  std::string ret = "";
  CHECK(!reduce_info_.output_promoted_tensor_name_for_atomic_.empty())
    << "output promoted tensor name should not be empty!";
  ret += reduce_info_.output_promoted_tensor_name_for_atomic_;

  Tensor t = info_.FindTensor(reduce_info_.output_promoted_tensor_name_for_atomic_);
  CHECK(t.defined());
  std::vector<std::string> shapeInfo;
  for (auto &e : t->shape) {
    std::stringstream ss;
    ss << e;
    if (ss.str() != USELESS_SHAPE_SIZE) {
      shapeInfo.push_back(ss.str());
    }
  }

  int size = reduce_info_.output_promoted_tensor_indexs_for_atomic_.size();
  if (size == 0) {
    ret += DEFAULT_TENSOR_INDEX;
  } else if (size == 1) {
    ret += "[";
    ret += reduce_info_.output_promoted_tensor_indexs_for_atomic_.at(0);
    ret += "]";
  } else {
    ret += "[";
    for (int i = 0; i < size - 1; ++i) {
      ret += reduce_info_.output_promoted_tensor_indexs_for_atomic_.at(i);
      for (int j = i + 1; j < size; ++j) {
        ret += "*";
        ret += "(";
        ret += shapeInfo.at(j);
        ret += ")";
      }
      ret += "+";
    }
    ret += reduce_info_.output_promoted_tensor_indexs_for_atomic_.at(size - 1);
    ret += "]";
  }

  reduce_info_.output_promoted_tensor_info_for_atomic_ = ret;
}

void GpuIslEmitter::MakePromotedTensorInfoForReduce() {
  std::string ret = "";
  ret += "&";
  CHECK(!reduce_info_.promoted_tensor_name_for_reduce_.empty())
    << "promoted tensor for reduce name should not be empty!";
  ret += reduce_info_.promoted_tensor_name_for_reduce_;

  reduce_info_.promoted_tensor_info_for_reduce_ = ret;
}

Stmt GpuIslEmitter::EmitAkgAtomicReturnInfo(Stmt s, std::string info) {
  return Evaluate::make(
    Call::make(Int(32), REDUCE, {StringImm::make(REDUCE_ATOMIC_FLAG), StringImm::make(info)}, Call::Intrinsic));
}

bool GpuIslEmitter::NoNeedToEmitForTempTensor(const isl::id &id) {
  bool no_need = true;
  auto origin_binds = info_.user_config_.GetOriginBind();
  for (auto i : origin_binds) {
    if (!i.first.defined()) continue;
    std::string name = i.first->op->name;
    if (name == id.name()) {
      no_need = false;
      break;
    }
  }
  return no_need;
}

Stmt GpuIslEmitter::EmitBlock(const isl::ast_node_block &block_node) {
  std::vector<Stmt> stmts;

  int num = block_node.get_children().size();
  int last_num = 0;
  for (int i = num - 1; i >= 0; --i) {
    auto child = block_node.get_children().at(i);

    if (auto node = child.as<isl::ast_node_user>()) {
      CHECK(node.get_expr().isa<isl::ast_expr_op>());
      isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
      CHECK(usr_expr);
      auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
      if (info_.IsRealize(stmt_id)) {
        isl::id new_stmt_id = isl::id(stmt_id.ctx(), stmt_id.name().substr(REALIZE_PREFIX_LEN));
        int stmt_num = stmts.size();
        CHECK_NE(stmt_num, 0) << "when stmt_num is zero, no realize should be emitted!.";
        if (stmt_num == 1) {
          stmts[0] = InsertRealize(stmts[0], new_stmt_id);
        } else {
          if (stmt_num - last_num == 1) {
            stmts[0] = InsertRealize(stmts[0], new_stmt_id);
          } else {
            for (int index = stmt_num - 2 - last_num; index >= 0; --index) {
              auto p_index = static_cast<unsigned int>(index);
              stmts[p_index] = Block::make(stmts[p_index], stmts[p_index + 1]);
            }
            stmts[0] = InsertRealize(stmts[0], new_stmt_id);
          }
        }
        last_num = stmt_num - 1;
        continue;
      }
    }

    Stmt body = EmitAst(child);
    if (!body.defined()) continue;
    stmts.insert(stmts.begin(), body);
  }

  int len = stmts.size();

  if (len == 0) {
    return Stmt();
  }

  if (last_num == len - 1) {
    return stmts[0];
  } else {
    for (int index = len - 2 - last_num; index >= 0; --index) {
      auto p_index = static_cast<unsigned int>(index);
      stmts[p_index] = Block::make(stmts[p_index], stmts[p_index + 1]);
    }
    return stmts[0];
  }
}

Stmt GpuIslEmitter::EmitFor(const isl::ast_node_for &node) {
  isl::id isl_iter_id = node.get_iterator().as<isl::ast_expr_id>().get_id();
  VarExpr iter_expr(isl_iter_id.to_str());
  VarExpr iter_expr_new = AllocUniqueIterName(iter_expr);
  if (iter_expr.get() != iter_expr_new.get()) {
    iter_map_ssa_[iter_expr.get()] = iter_expr_new.get();
  }
  PushIter(iter_expr.get());

  Expr init_expr = Interpret(node.get_init());

  auto isl_cond = node.get_cond().as<isl::ast_expr_op>();
  CHECK(isl_cond.as<isl::ast_expr_op_lt>() || isl_cond.as<isl::ast_expr_op_le>());
  auto cond_lhs = isl_cond.get_arg(0).as<isl::ast_expr_id>();
  CHECK(cond_lhs);
  CHECK_EQ(cond_lhs.get_id(), isl_iter_id);
  Expr cond_expr = Interpret(isl_cond.get_arg(1));

  int64_t inc = static_cast<int64_t>(WrappedStrtol(node.get_inc().to_C_str()));
  CHECK_NE(inc, 0) << "stride should not be zero!.";

  bool need_to_modify_inc_ = false;
  if (inc != 1) {
    need_to_modify_inc_ = true;
    Expr original_init_expr = init_expr;
    init_expr = ModifyTheInitExpr(init_expr);
    cond_expr = ModifyTheCondExpr(cond_expr, static_cast<int>(inc));
    Expr modify_iter = ModifyTheIterExpr(iter_expr_new, static_cast<int>(inc), original_init_expr);
    stride_modify_iter_map_[iter_expr_new.get()] = modify_iter;
  }

  if (isl_cond.as<isl::ast_expr_op_le>()) {
    cond_expr = Simplify(cond_expr + 1);
  }

  cond_expr = Simplify(cond_expr - init_expr);

  // add for tensor core

  if (tensor_core_info_.core_area_) {
    tensor_core_info_.core_area_for_extent_[iter_expr_new] = cond_expr;
  }

  if (tensor_core_info_.fragment_axis_begin_) {
    if (tensor_core_info_.is_fragment_m_) {
      tensor_core_info_.fragment_m_ = cond_expr;
    } else if (tensor_core_info_.is_fragment_n_) {
      tensor_core_info_.fragment_n_ = cond_expr;
    }
  }

  Stmt body_stmt = EmitAst(node.get_body());

  if (!body_stmt.defined()) {
    PopIter(iter_expr.get());
    iter_map_ssa_.erase(iter_expr.get());
    if (tensor_core_info_.core_area_) {
      tensor_core_info_.core_area_for_extent_.erase(iter_expr_new);
    }
    return Stmt();
  }

  if (need_to_modify_inc_) {
    stride_modify_iter_map_.erase(iter_expr_new.get());
  }
  PopIter(iter_expr.get());
  iter_map_ssa_.erase(iter_expr.get());
  if (tensor_core_info_.core_area_) {
    tensor_core_info_.core_area_for_extent_.erase(iter_expr_new);
  }
  return For::make(iter_expr_new, init_expr, cond_expr, ForType::Serial, DeviceAPI::None, body_stmt);
}

Stmt GpuIslEmitter::EmitIf(const isl::ast_node_if &node) {
  Expr cond_expr = Interpret(node.get_cond());
  cur_if_list_.push_back(cond_expr.get());
  Stmt then_case = EmitAst(node.get_then_node());
  if (!then_case.defined()) {
    return Stmt();
  }
  Stmt else_case;
  if (node.has_else_node()) {
    else_case = EmitAst(node.get_else_node());
  }
  cur_if_list_.pop_back();
  if (reduce_info_.init_stmt_emit_) {
    reduce_info_.init_stmt_emit_ = false;
    if (info_.user_config_.GetEnableAtomicAdd()) {
      cond_expr = ConditionExprMod().Mutate(cond_expr);
    }
  }

  if (!cond_expr.defined()) {
    return then_case;
  }

  return IfThenElse::make(cond_expr, then_case, else_case);
}

Stmt GpuIslEmitter::EmitUserStmt(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  stmt_id_ = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  node_id_ = node.get_annotation();
  const Node *stmt_node = info_.analysis_result_.GetStatementMap().at(stmt_id_);
  CHECK(stmt_node);
  // compute VarMap to replace old iterators
  auto build = node_info_map_.at(node_id_).build;
  auto tuple = info_.analysis_result_.GetOperatorDomainMap().at(stmt_id_).tuple;
  auto iterator_map = node_info_map_.at(node_id_).iterator_map;

  auto ids = info_.analysis_result_.GetReduceInitIds();
  for (auto &i : ids) {
    if (i.get_name() == stmt_id_.get_name()) {
      reduce_info_.init_stmt_emit_ = true;
    }
  }

  var_map_.clear();
  for (unsigned int i = 0; i < tuple.size(); ++i) {
    isl::id isl_old_iter = tuple.get_id(i);
    auto isl_expr = build.expr_from(iterator_map.get_pw_aff(i));
    Expr halide_new_iter = Interpret(isl_expr);
    var_map_.emplace(isl_old_iter, halide_new_iter);
  }

  return EmitUserStmtContent(stmt_node);
}

Expr GpuIslEmitter::ModifyTheInitExpr(const Expr &e) { return 0; }

Expr GpuIslEmitter::ModifyTheCondExpr(const Expr &e, int inc) { return e / Expr(inc); }

Expr GpuIslEmitter::ModifyTheIterExpr(const VarExpr &iter, int inc, const Expr &init) {
  return Simplify(iter * inc + init);
}

int GpuIslEmitter::GetThreadExtent(const std::string &name) {
  if (name == BLOCK_IDX_X || name == BLOCK_IDX_Y || name == BLOCK_IDX_Z) {
    auto block_cfg = info_.user_config_.GetBlockConfig();
    CHECK(block_cfg) << "block config is null.";
    return name == BLOCK_IDX_X ? block_cfg->GetX().second
                               : (name == BLOCK_IDX_Y ? block_cfg->GetY().second : block_cfg->GetZ().second);
  }

  if (name == THREAD_IDX_X || name == THREAD_IDX_Y || name == THREAD_IDX_Z) {
    auto thread_cfg = info_.user_config_.GetThreadConfig();
    CHECK(thread_cfg) << "thread config is null.";
    if (info_.user_config_.GetEnableOneDimThread()) {
      return name == THREAD_IDX_X ? (thread_cfg->GetX().second * thread_cfg->GetY().second * thread_cfg->GetZ().second)
                                  : 1;
    }
    return name == THREAD_IDX_X ? thread_cfg->GetX().second
                                : (name == THREAD_IDX_Y ? thread_cfg->GetY().second : thread_cfg->GetZ().second);
  }
  LOG(WARNING) << "Unrecognized thread name " << name;
  return 1;
}

void GpuIslEmitter::PrepareDataForTensorCore() {
  auto binds = info_.user_config_.GetBind();

  auto thread_cfg = info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg) << "thread config is null";
  int tx = thread_cfg->GetX().second;
  int ty = thread_cfg->GetY().second;
  int tz = thread_cfg->GetZ().second;

  if (info_.user_config_.GetEnableOneDimThread()) {
    tx = tx * ty * tz;
    ty = 1;
    tz = 1;
  }

  for (auto i : binds) {
    if (!i.first.defined()) continue;
    if (!i.second.defined()) continue;
    auto t = i.first;
    auto b = i.second;

    std::string name = t->op->name;

    air::ir::TensorKey key{t->op, t->value_index};
    Region bounds;
    if (bounds.empty()) {
      for (auto j : t->shape) {
        bounds.push_back(Range::make_by_min_extent(Expr(0), j));
      }
    }

    tensor_core_info_.bounds_[key] = bounds;

    Array<Expr> strides;
    for (size_t i = 1; i < b->shape.size(); ++i) {
      Expr stride = IntImm::make(Int(32), 1);
      for (size_t j = b->shape.size() - 1; j >= i; --j) {
        stride = Mul::make(stride, b->shape[j]);
      }
      strides.push_back(stride);
    }
    strides.push_back(make_const(Int(32), 1));
    tensor_core_info_.strides_[name] = strides;
  }

  auto tile_size = info_.analysis_result_.GetTileSizes();
  CHECK_GE(tile_size.size(), 3) << "tile size should be greater to 3";
  int len = tile_size.size();
  tensor_core_info_.warp_tile_.m = tile_size[len - 3].l0_tiling_size;
  tensor_core_info_.warp_tile_.n = tile_size[len - 2].l0_tiling_size;
  tensor_core_info_.warp_tile_.k = tile_size[len - 1].l0_tiling_size;

  bool result = CheckTileValid(tensor_core_info_.warp_tile_);
  CHECK(result) << "tile set is not valid!";

  tensor_core_info_.thread_tile_.m = tensor_core_info_.warp_tile_.m / tx;
  tensor_core_info_.thread_tile_.n = tx / 2;
  tensor_core_info_.thread_tile_.k = tile_size[2].l0_tiling_size / tz;

  tensor_core_info_.matrix_abc_ = info_.analysis_result_.GetMatrixMatmulMap();
  tensor_core_info_.matrix_major_ = info_.analysis_result_.GetMatrixMatmulMajor();

  for (auto &i : tensor_core_info_.matrix_abc_) {
    tensor_core_info_.frag_reg_.insert(i.first + LOCAL_SUFFIX);
  }

  tensor_core_info_.warp_threads_y_ = 32 / tx;
  tensor_core_info_.warp_threads_x_ = tx;
}

bool GpuIslEmitter::CheckTileValid(Tile tile) {
  if (tile.m == 16 && tile.n == 16 && tile.k == 4) {
    tensor_core_info_.wmma_scope_ = "akg";
    return true;
  }
  if (tile.m == 16 && tile.n == 16 && tile.k == 16) {
    tensor_core_info_.wmma_scope_ = "nvcuda";
    return true;
  }
  if (tile.m == 8 && tile.n == 32 && tile.k == 16) {
    tensor_core_info_.wmma_scope_ = "nvcuda";
    return true;
  }
  if (tile.m == 32 && tile.n == 8 && tile.k == 16) {
    tensor_core_info_.wmma_scope_ = "nvcuda";
    return true;
  }
  return false;
}

Stmt GpuIslEmitter::Emit(const isl::ast_node &node) {
  Stmt stmt = EmitAst(node);

  // emit realize for temporary tensor
  stmt = EmitRealizeForGlobalTensor(stmt);

  // iter var node attr emit
  std::map<std::string, VarExpr>::iterator it;
  for (it = iter_name_map_.begin(); it != iter_name_map_.end(); it++) {
    IterVar axis = IterVarNode::make(Range(), it->second, air::kThreadIndex, it->second->name_hint);
    stmt = AttrStmt::make(axis, "thread_extent", Expr(GetThreadExtent(it->second->name_hint)), stmt);
  }

  // attr for one dimension mapping
  if (info_.user_config_.GetEnableOneDimThread()) {
    stmt =
      AttrStmt::make(Expr(""), ORIGIN_THREAD_DIM_X, Expr(info_.user_config_.GetThreadConfig()->GetX().second), stmt);
  }

  // add tensor core plan two attr
  if (info_.user_config_.GetEnableTensorCore()) {
    if (info_.user_config_.GetEnableTensorCoreUsePoly()) {
      stmt = AttrStmt::make(Expr(""), "pragma_tensor_core", StringImm::make(TENSOR_CORE_MODE_TWO), stmt);
      stmt = AttrStmt::make(Expr("INFO"), "wmma_scope", StringImm::make(tensor_core_info_.wmma_scope_), stmt);
    } else {
      stmt = AttrStmt::make(Expr(""), "pragma_tensor_core", StringImm::make(TENSOR_CORE_MODE_ONE), stmt);
    }
  }

  bool emit_attr = AddAttrCheck().Run(stmt);
  if (emit_attr) {
    IterVar axis = IterVarNode::make(Range(), VarExpr(THREAD_IDX_X), air::kThreadIndex, THREAD_IDX_X);
    int value = GetThreadExtent(THREAD_IDX_X);
    stmt = AttrStmt::make(axis, "thread_extent", Expr((value == 0) ? 1 : value), stmt);
  }

  Stmt s = AkgReduceAddTensorIndex(reduce_info_.promoted_tensor_indexs_for_reduce_,
                                   reduce_info_.promoted_tensor_shape_for_reduce_)
             .Mutate(stmt);

  if (tensor_core_info_.is_tensor_core_ && info_.user_config_.GetEnableTensorCoreUsePoly()) {
    s = AddMmaAttrFlag(tensor_core_info_).Mutate(s);
    s = EmitForTensorCore(s, tensor_core_info_);
  } else if (info_.user_config_.GetEnableTensorCore()) {
    tensor_core_info_.cast_tensors_ = info_.analysis_result_.GetCastTensors();
    s = EmitForTensorCoreDesignOne(s, tensor_core_info_);
  }

  return s;
}

Stmt GpuIslEmitter::EmitRealizeForGlobalTensor(Stmt stmt) {
  auto binds = info_.user_config_.GetBind();
  auto origin_binds = info_.user_config_.GetOriginBind();
  std::unordered_set<std::string> tensor_name;

  for (auto i : binds) {
    if (!i.first.defined()) continue;
    tensor_name.insert(i.first->op->name);
  }

  for (auto i : binds) {
    if (!i.first.defined()) continue;
    // input and output tensor, no need to emit realize
    if (origin_binds.find(i.first) != origin_binds.end()) {
      continue;
    }

    // promoted tensor, the realize info already emitted before
    std::string name = i.first->op->name;
    if (IsEndsWith(name, MEM_TYPE_SHARED) || IsEndsWith(name, MEM_TYPE_LOCAL)) {
      continue;
    }

    // if the tensor is temporary, but has already promoted, there is no need to emit realize
    if (tensor_name.find(name + "_" + MEM_TYPE_SHARED) != tensor_name.end() ||
        tensor_name.find(name + "_" + MEM_TYPE_LOCAL) != tensor_name.end()) {
      continue;
    }

    // if the tensor is temporary and it is not promoted, it needs to emit realize
    stmt = InsertRealize(stmt, isl::id(info_.GetCtx(), name));
  }
  return stmt;
}

Stmt GpuIslEmitter::EmitMark(const isl::ast_node_mark &node) {
  std::string mark = node.get_id().get_name();
  if (IsStartsWith(mark, REDUCE_ATOMIC_FLAG)) {
    std::vector<std::string> strs = common::Split(mark, "_");
    CHECK_EQ(strs.size(), REDUCE_ATOMIC_FLAG_SIZE) << "atomic mark format is not right!.";
    reduce_info_.reduce_op_.clear();
    if (AkgSupportedReduceOp.count(strs[REDUCE_ATOMIC_FLAG_TYPE_POS])) {
      reduce_info_.reduce_op_ = AKG_REDUCE_LIB_SPACE;
      reduce_info_.reduce_op_ += "::";
      reduce_info_.reduce_op_ += strs[REDUCE_ATOMIC_FLAG_TYPE_POS];
    }
    CHECK(!reduce_info_.reduce_op_.empty()) << "reduce op should not be empty!";

    if (strs[REDUCE_ATOMIC_FLAG_POS] == REDUCE_ATOMIC_FLAG) {
      reduce_info_.is_atomic = true;
    }
  }

  // add for prefetch pass
  if (mark == PROMOTE_GLOBAL_TO_SHARED_AB) {
    Stmt stmt = EmitAst(node.get_node());
    if (!stmt.defined()) {
      return Stmt();
    }
    return AttrStmt::make(Expr("INFO"), SHARED_MEM_PROMOTED_COMPLETE, StringImm::make(SHARED_MEM_PROMOTED_COMPLETE),
                          stmt);
  }

  if ((mark == MATRIX_A) || (mark == MATRIX_B) || (mark == MATRIX_C) || (mark == WARP_MARKER)) {
    if (!tensor_core_info_.data_is_set_) {
      PrepareDataForTensorCore();
      tensor_core_info_.data_is_set_ = true;
    }
    tensor_core_info_.fragment_axis_begin_ = false;
    if (mark == WARP_MARKER) {
      mark = MMA_SYNC;
    }
    if (mark == MATRIX_C) {
      mark = MMA_C;
    }

    if (!tensor_core_info_.data_is_set_) {
      PrepareDataForTensorCore();
      tensor_core_info_.data_is_set_ = true;
    }

    tensor_core_info_.is_tensor_core_ = true;
    tensor_core_info_.matrix_info_[mark] = true;
    tensor_core_info_.core_area_ = true;

    Stmt stmt = EmitAst(node.get_node());
    stmt = DeleteUselessFor().Mutate(stmt);
    tensor_core_info_.matrix_info_[mark] = false;
    tensor_core_info_.core_area_ = false;
    return AttrStmt::make(Expr("INFO"), mark, StringImm::make(mark), stmt);
  }

  if ((mark == FRAGMENT_A) || (mark == FRAGMENT_B)) {
    tensor_core_info_.fragment_axis_begin_ = true;
    if (mark == FRAGMENT_A) {
      tensor_core_info_.is_fragment_m_ = true;
    } else if (mark == FRAGMENT_B) {
      tensor_core_info_.is_fragment_n_ = true;
    }
    Stmt stmt = EmitAst(node.get_node());
    tensor_core_info_.fragment_axis_begin_ = false;
    tensor_core_info_.is_fragment_m_ = false;
    tensor_core_info_.is_fragment_n_ = false;
    if (!stmt.defined()) {
      return Stmt();
    }
    return AttrStmt::make(Expr("INFO"), mark, StringImm::make(mark), stmt);
  }

  if ((mark == "promote_vectorization") || (mark == "promote_local_to_global")) {
    Stmt stmt = EmitAst(node.get_node());
    if (!stmt.defined()) {
      return Stmt();
    }
    return AttrStmt::make(Expr("INFO"), mark, StringImm::make(mark), stmt);
  }
  return EmitAst(node.get_node());
}  // namespace poly

std::string GpuIslEmitter::FindRealizeScopeToString(const isl::id &var) {
  if (info_.analysis_result_.CountBufferDefInfo(var)) {
    auto tensor_info = info_.analysis_result_.GetBufferDefInfo(var);
    MemType mem_type = tensor_info.DstMemType();

    switch (mem_type) {
      case MemType::SHARED_:
        return MEM_TYPE_SHARED;
      case MemType::LOCAL_:
        return MEM_TYPE_LOCAL;
      default:
        LOG(FATAL) << "unexpected mem_type of var " << var;
        return "ERROR";
    }
  }
  return "";
}

Expr GpuIslEmitter::FindRealizeScope(const isl::id &var) { return Expr(FindRealizeScopeToString(var)); }

Stmt GpuIslEmitter::InsertRealize(Stmt stmt, const isl::id &var) {
  stmt = FindInnerRealize(var.get_name()).Mutate(stmt);

  // A tensor may be defined multiple times in BufferDefInfo due to nested realize.
  // Because we cannot determine which one we actually want, we have to be conservative here
  // and allocate space for the largest shape to avoid overflow.
  Tensor t = info_.FindTensorWithLargestShape(var);
  Region bounds;

  // no isolate
  if (bounds.empty()) {
    for (auto j : t->shape) {
      bounds.push_back(Range::make_by_min_extent(Expr(0), j));
    }
  }

  // If isolate, make a new buffer
  auto buf = info_.user_config_.GetBind().at(t);

  auto tt = placeholder(t->shape, t->dtype, t->op->name);

  stmt = TensorSubstitute(stmt, t->op, tt->op, tt->value_index);
  if (tensor_core_info_.is_tensor_core_) {
    stmt = TensorSubstituteTensorCore(t->op, tt->op, tt->value_index).Mutate(stmt);
  }
  t = tt;
  if (info_.analysis_result_.CountBufferDefInfo(var)) {
    auto decl = info_.analysis_result_.GetBufferDefInfo(var);
    decl.tensor = t;
  }
  info_.user_config_.SetBind(t, buf);
  stmt = TensorSubstitute2(stmt, t->op->func_name(), t->op, t->value_index);
  stmt = Realize::make(t->op, t->value_index, t->dtype, bounds, const_true(1), stmt);
  realized_.insert(t);
  stmt = AttrStmt::make(t->op, air::ir::attr::realize_scope, FindRealizeScope(var), stmt);

  return stmt;
}

Expr GpuIslEmitter::IterNameAdaptor(std::string name) {
  if (iter_name_map_.find(name) != iter_name_map_.end()) {
    return iter_name_map_[name];
  } else if (name.find(REPLACE) != std::string::npos) {
    name = name.substr(strlen(REPLACE));
    if (info_.user_config_.GetEnableTileL0()) {
      return SingleConfigToMultiBand(name);
    }
    return AdaptPolyNewVar(name);
  } else {
    return VarExpr(name);
  }
}

Expr GpuIslEmitter::SingleConfigToMultiBand(std::string name) {
  Expr e;
  VarExpr original_id;
  int rep_size = 1;
  auto l0_block_size = info_.user_config_.GetL0BlockSize();
  if (name.find(B0) != std::string::npos) {
    original_id = iter_name_map_[B0];
    rep_size = l0_block_size[0];
  } else if (name.find(B1) != std::string::npos) {
    original_id = iter_name_map_[B1];
    rep_size = l0_block_size[1];
  } else {
    original_id = iter_name_map_[B2];
    rep_size = l0_block_size[2];
  }

  if (rep_size < 0) {
    return e;
  }

  if (name.find(L0) != std::string::npos) {
    e = Mod::make(original_id, rep_size);
  } else if (name.find(L1) != std::string::npos) {
    e = Div::make(original_id, rep_size);
  } else {
    LOG(FATAL) << "Unexpected binding id: " << name;
  }
  return e;
}

// if new var is added in poly process, modify the logic here.
// another modify pos is IterNameAdaptor interface
Expr GpuIslEmitter::AdaptPolyNewVar(std::string name) {
  Expr e;
  std::string t0_string = T0;
  int suffix_len = t0_string.size() + 1;
  auto tensor_name = name.substr(0, name.size() - suffix_len);
  if (!info_.user_config_.GetReplaceConfig().count(tensor_name)) {
    return e;
  }
  auto mapping_cfg = (info_.user_config_.GetReplaceConfig()[tensor_name]);
  CHECK(mapping_cfg) << "mapping config is null.";
  int mx = mapping_cfg->GetX().second;
  int my = mapping_cfg->GetY().second;
  int mz = mapping_cfg->GetZ().second;
  if (name.find(WARP_COMPUTE) != std::string::npos) {
    if (name.find(T0) != std::string::npos) {
      e = Div::make(iter_name_map_[T0], WARP_SIZE);
      e = Mod::make(e, mx);
      return e;
    } else if (name.find(T1) != std::string::npos) {
      e = Div::make(iter_name_map_[T0], WARP_SIZE);
      e = Div::make(e, mx);
      return e;
    }
  } else {
    if (name.find(T0) != std::string::npos) {
      e = Mod::make(iter_name_map_[T0], mx);
      return e;
    } else if (name.find(T1) != std::string::npos) {
      e = Div::make(iter_name_map_[T0], mx);
      if (mz == 1) {
        return e;
      }
      e = Mod::make(e, my);
      return e;
    } else if (name.find(T2) != std::string::npos) {
      e = Div::make(iter_name_map_[T0], mx);
      e = Div::make(e, my);
      return e;
    }
  }
  return e;
}

Expr GpuIslEmitter::Interpret(const isl::ast_expr &e) {
  if (auto int_expr = e.as<isl::ast_expr_int>()) {
    return Expr(IslExprToSInt(int_expr));
  } else if (auto id_expr = e.as<isl::ast_expr_id>()) {
    // If this variable is defined by loop index, we need sharing it.
    const Variable *var = GetIterByName(id_expr.get_id().get_name());
    if (var) {
      if (iter_map_ssa_.find(var) != iter_map_ssa_.end()) {
        if (stride_modify_iter_map_.find(iter_map_ssa_.at(var)) != stride_modify_iter_map_.end()) {
          return stride_modify_iter_map_[iter_map_ssa_.at(var)];
        }
        return VarExpr(GetObjPtr(iter_map_ssa_.at(var)));
      }

      if (stride_modify_iter_map_.find(var) != stride_modify_iter_map_.end()) {
        return stride_modify_iter_map_[var];
      }
      return VarExpr(GetObjPtr(var));
    } else {
      return IterNameAdaptor(id_expr.get_id().to_str());
    }
  } else if (auto op_expr = e.as<isl::ast_expr_op>()) {
    return InterpretOp(op_expr);
  } else {
    LOG(FATAL) << "NYI " << e;
    return 0;
  }
}

Stmt GpuIslEmitter::EmitAccessNodeFromPromoteAcsCall(isl::id var, const Node *node, Array<Expr> &args) {
  const Call *call = static_cast<const Call *>(node);
  Tensor t = info_.FindTensor(var);
  return Evaluate::make(Call::make(call->type, var.get_name(), args, call->call_type, t->op, t->value_index));
}

Stmt GpuIslEmitter::EmitAccessNodeFromPromoteAcsProvide(isl::id var, const Node *node, Array<Expr> &args) {
  const auto provide = static_cast<const Provide *>(node);
  Tensor t = info_.FindTensor(var);
  Stmt s = Provide::make(t->op, 0, provide->value, args);
  return s;
}

VarExpr GpuIslEmitter::AllocUniqueIterName(const VarExpr v) {
  std::string ret = v->name_hint;
  if (for_iter_name_map_.find(ret) != for_iter_name_map_.end()) {
    ret += std::to_string(for_iter_name_map_[ret]);
    for_iter_name_map_[v->name_hint]++;
    return Variable::make(v.type(), ret);
  } else {
    for_iter_name_map_[v->name_hint] = 1;
    return v;
  }
}

void GetNameWithoutShared(isl::id &tensor_id, ScopInfo &info) {
  if (info.user_config_.GetEnableMatmul()) {
    size_t pos = tensor_id.get_name().find(SHARE_SUFFIX);
    std::string substr = tensor_id.get_name().substr(0, pos);
    if (pos != 0) tensor_id = isl::id(tensor_id.ctx(), substr);
  }
}

isl::multi_aff GpuIslEmitter::TensorAccessMultAff(isl::id &tensor_id, const Array<Expr> &tensor_index,
                                                  const isl::id &node_id) {
  GetNameWithoutShared(tensor_id, info_);
  return IslEmitter::TensorAccessMultAff(tensor_id, tensor_index, node_id);
}

Array<Expr> EmitTensorCoreHelper::GetTileSize(const std::string &name) {
  auto it = tensor_core_info_.matrix_abc_.find(name);
  auto it2 = tensor_core_info_.matrix_major_.find(name);
  CHECK(it != tensor_core_info_.matrix_abc_.end() && it2 != tensor_core_info_.matrix_major_.end())
    << "Cannot find matrix info for " << name;
  Expr size0 = make_const(Int(32), 16);
  Expr size1 = make_const(Int(32), 16);
  if (it->second == MMA_A && it2->second == COL_MAJOR) {
    size0 = make_const(Int(32), tensor_core_info_.warp_tile_.k);
    size1 = make_const(Int(32), tensor_core_info_.warp_tile_.m);
  }
  if (it->second == MMA_A && it2->second == ROW_MAJOR) {
    size0 = make_const(Int(32), tensor_core_info_.warp_tile_.m);
    size1 = make_const(Int(32), tensor_core_info_.warp_tile_.k);
  }
  if (it->second == MMA_B && it2->second == ROW_MAJOR) {
    size0 = make_const(Int(32), tensor_core_info_.warp_tile_.k);
    size1 = make_const(Int(32), tensor_core_info_.warp_tile_.n);
  }
  if (it->second == MMA_B && it2->second == COL_MAJOR) {
    size0 = make_const(Int(32), tensor_core_info_.warp_tile_.n);
    size1 = make_const(Int(32), tensor_core_info_.warp_tile_.k);
  }

  if (it->second == MATRIX_C) {
    size0 = make_const(Int(32), tensor_core_info_.warp_tile_.m);
    size1 = make_const(Int(32), tensor_core_info_.warp_tile_.n);
  }
  Array<Expr> tile_size = {size0, size1};
  return tile_size;
}

void EmitTensorCoreHelper::SetDataForLoad(Expr src, Expr stride, Expr major, const Call *call, const Provide *op,
                                          NodePtr<BufferNode> &node) {
  data_for_load_.src = src;
  data_for_load_.stride = stride;
  data_for_load_.major = major;
  data_for_load_.call = call;
  data_for_load_.op = op;
  data_for_load_.node = node;
}
void EmitTensorCoreHelper::SetDataForStore(Expr dst, Expr stride, const Call *call, NodePtr<BufferNode> &node) {
  data_for_store_.dst = dst;
  data_for_store_.stride = stride;
  data_for_store_.call = call;
  data_for_store_.node = node;
}
void EmitTensorCoreHelper::SetDataForFill(const Provide *op, const Call *call, NodePtr<BufferNode> &node) {
  data_for_fill_.call = call;
  data_for_fill_.op = op;
  data_for_fill_.node = node;
}
void EmitTensorCoreHelper::SetDataForSync(Expr a, Expr b, Expr c, NodePtr<BufferNode> &node_a,
                                          NodePtr<BufferNode> &node_b, NodePtr<BufferNode> &node_c) {
  data_for_sync_.a = a;
  data_for_sync_.b = b;
  data_for_sync_.c = c;
  data_for_sync_.node_a = node_a;
  data_for_sync_.node_b = node_b;
  data_for_sync_.node_c = node_c;
}

void EmitTensorCoreHelper::PrepareDataCore() {
  auto it = tensor_core_info_.bounds_.find(key_);
  CHECK(it != tensor_core_info_.bounds_.end());
  Array<Expr> min_bound;
  for (auto i : it->second) {
    min_bound.push_back(i->min);
  }

  CHECK_GE(it->second.size(), 2);
  Array<Expr> shape;
  for (size_t i = 0; i < it->second.size() - 2; ++i) {
    shape.push_back(it->second[i]->extent);
  }
  auto tile_size = GetTileSize(SimplifyName(call_->name));
  shape.push_back(tile_size[0]);
  shape.push_back(tile_size[1]);

  tensor_core_info_.min_bounds_[call_->name] = min_bound;

  Array<Expr> strides;
  for (size_t i = 1; i < shape.size(); ++i) {
    Expr stride = IntImm::make(Int(32), 1);
    for (size_t j = shape.size() - 1; j >= i; --j) {
      stride = Mul::make(stride, shape[j]);
    }
    strides.push_back(stride);
  }
  strides.push_back(make_const(Int(32), 1));

  Expr elem_offset = IntImm::make(Int(32), 0);
  CHECK_EQ(call_->args.size(), min_bound.size());
  for (size_t i = 0; i < min_bound.size(); i++) {
    auto arg = call_->args[i];
    arg = DeleteThreadIdx().Mutate(arg);
    arg = Simplify(arg);
    elem_offset = Add::make(elem_offset, Mul::make(strides[i], Sub::make(arg, min_bound[i])));
  }

  auto it2 = tensor_core_info_.matrix_abc_.find(SimplifyName(call_->name));
  CHECK(it2 != tensor_core_info_.matrix_abc_.end()) << "Cannot find matrix info for " << call_->name;
  buffer_node_->data = Variable::make(Handle(), call_->name);
  buffer_node_->name = call_->name;
  std::string name = it2->second;
  if (name == MATRIX_C) {
    name = MMA_C;
  }
  buffer_node_->scope = "wmma." + name;
  buffer_node_->dtype = data_type_;
  buffer_node_->strides = strides;
  buffer_node_->shape = shape;
  buffer_node_->data_alignment = 1;
  buffer_node_->elem_offset = Simplify(elem_offset);
  buffer_node_->offset_factor = 1;
  Buffer buffer(buffer_node_);

  NodePtr<TensorNode> tensor_node = make_node<TensorNode>();
  tensor_node->value_index = key_.value_index;
  tensor_node->op = Downcast<Operation>(key_.f);
  tensor_node->shape = shape;
  tensor_node->dtype = data_type_;
  Tensor tensor(tensor_node);

  Array<Expr> args;
  for (size_t i = 0; i < call_->args.size(); ++i) {
    auto arg = call_->args[i];
    arg = DeleteThreadIdx().Mutate(arg);
    arg = Simplify(arg);

    args.push_back(arg);
    args.push_back(shape[i]);
  }
  tuple_ = Call::make(Handle(), air::ir::intrinsic::tvm_tuple, args, Call::Intrinsic);
  node_ = {buffer, tensor};
}

Stmt EmitTensorCoreHelper::MakeLoadTransform() {
  key_ = air::ir::TensorKey{data_for_load_.op->func, data_for_load_.op->value_index};
  call_ = data_for_load_.call;
  buffer_node_ = data_for_load_.node;
  data_type_ = call_->type;

  PrepareDataCore();
  Buffer buffer = Downcast<Buffer>(node_[0]);
  Stmt stmt = Evaluate::make(Call::make(
    Handle(), air::ir::intrinsic::tvm_load_matrix_sync,
    {buffer->data, tensor_core_info_.warp_tile_.m, tensor_core_info_.warp_tile_.n, tensor_core_info_.warp_tile_.k,
     Simplify(buffer->elem_offset), data_for_load_.src, data_for_load_.stride, data_for_load_.major},
    Call::Intrinsic));
  return AttrStmt::make(node_, "buffer_bind_scope", tuple_, stmt);
}

Stmt EmitTensorCoreHelper::MakeStoreTransform() {
  key_ = air::ir::TensorKey{data_for_store_.call->func, data_for_store_.call->value_index};
  call_ = data_for_store_.call;
  buffer_node_ = data_for_store_.node;
  data_type_ = call_->type;

  PrepareDataCore();
  Buffer buffer = Downcast<Buffer>(node_[0]);
  Stmt stmt = Evaluate::make(Call::make(
    Handle(), air::ir::intrinsic::tvm_store_matrix_sync,
    {buffer->data, tensor_core_info_.warp_tile_.m, tensor_core_info_.warp_tile_.n, tensor_core_info_.warp_tile_.k,
     buffer->elem_offset, data_for_store_.dst, data_for_store_.stride, StringImm::make(ROW_MAJOR)},
    Call::Intrinsic));
  return AttrStmt::make(node_, "buffer_bind_scope", tuple_, stmt);
}

Stmt EmitTensorCoreHelper::MakeFillTransform() {
  key_ = air::ir::TensorKey{data_for_fill_.call->func, data_for_fill_.call->value_index};
  call_ = data_for_fill_.call;
  buffer_node_ = data_for_fill_.node;
  data_type_ = call_->type;

  PrepareDataCore();
  Buffer buffer = Downcast<Buffer>(node_[0]);
  Stmt stmt = Evaluate::make(Call::make(Handle(), air::ir::intrinsic::tvm_fill_fragment,
                                        {buffer->data, tensor_core_info_.warp_tile_.m, tensor_core_info_.warp_tile_.n,
                                         tensor_core_info_.warp_tile_.k, buffer->elem_offset, data_for_fill_.op->value},
                                        Call::Intrinsic));
  return AttrStmt::make(node_, "buffer_bind_scope", tuple_, stmt);
}

Stmt EmitTensorCoreHelper::MakeSyncTransform() {
  bool is_cast = false;
  if (data_for_sync_.a.as<Call>()) {
    auto call_a = data_for_sync_.a.as<Call>();
    key_ = air::ir::TensorKey{call_a->func, call_a->value_index};
    call_ = call_a;
    buffer_node_ = data_for_sync_.node_a;
    data_type_ = call_->type;
    is_cast = true;
  } else if (data_for_sync_.a.as<Cast>()) {
    auto cast_a = data_for_sync_.a.as<Cast>();
    auto call_a = cast_a->value.as<Call>();
    CHECK(call_a);
    key_ = air::ir::TensorKey{call_a->func, call_a->value_index};
    call_ = call_a;
    buffer_node_ = data_for_sync_.node_a;
    data_type_ = call_->type;
    is_cast = true;
  }

  PrepareDataCore();

  auto tuple_a = tuple_;
  auto node_a = node_;

  if (data_for_sync_.b.as<Call>()) {
    auto call_b = data_for_sync_.b.as<Call>();
    key_ = air::ir::TensorKey{call_b->func, call_b->value_index};
    call_ = call_b;
    buffer_node_ = data_for_sync_.node_b;
    data_type_ = call_->type;
    is_cast = true;
  } else if (data_for_sync_.b.as<Cast>()) {
    auto cast_b = data_for_sync_.b.as<Cast>();
    auto call_b = cast_b->value.as<Call>();
    CHECK(call_b);
    key_ = air::ir::TensorKey{call_b->func, call_b->value_index};
    call_ = call_b;
    buffer_node_ = data_for_sync_.node_b;
    data_type_ = call_->type;
    is_cast = true;
  }

  PrepareDataCore();

  auto tuple_b = tuple_;
  auto node_b = node_;

  auto call_c = data_for_sync_.c.as<Call>();
  CHECK(call_c);
  key_ = air::ir::TensorKey{call_c->func, call_c->value_index};
  call_ = call_c;
  buffer_node_ = data_for_sync_.node_c;
  data_type_ = call_->type;

  PrepareDataCore();

  auto tuple_c = tuple_;
  auto node_c = node_;

  Buffer buffer_a(data_for_sync_.node_a);
  Buffer buffer_b(data_for_sync_.node_b);
  Buffer buffer = Downcast<Buffer>(node_c[0]);

  Stmt stmt = Evaluate::make(Call::make(Handle(), air::ir::intrinsic::tvm_mma_sync,
                                        {buffer->data, buffer->elem_offset, buffer_a->data, buffer_a->elem_offset,
                                         buffer_b->data, buffer_b->elem_offset, buffer->data, buffer->elem_offset},
                                        Call::Intrinsic));

  stmt = AttrStmt::make(node_c, "buffer_bind_scope", tuple_c, stmt);
  stmt = AttrStmt::make(node_b, "buffer_bind_scope", tuple_b, stmt);
  stmt = AttrStmt::make(node_a, "buffer_bind_scope", tuple_a, stmt);

  std::string cast_mode = CAST_MODE_1;
  if (is_cast) {
    stmt = AttrStmt::make(Expr("INFO"), CAST_FLAG, StringImm::make(cast_mode), stmt);
  }

  return stmt;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
