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

#include "poly/gpu_isl_emitter.h"
#include "pass/utils.h"
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

Stmt GpuIslEmitter::EmitStmt(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  auto node_id = node.get_annotation();

  if (info_.IsRead(stmt_id)) {
    is_sync_before_ = false;
    auto s = EmitRead(node);
    if (PRINT_EMITTER) {
      LOG(INFO) << ">>>>>>>>>>>>INPUT AST_NODE[READ]<<<<<<<<<<<<<<\n" << node;
      LOG(INFO) << ">>>>>>>>>>>>OUTPUT STMT<<<<<<<<<<<<\n" << s;
    }
    return s;
  } else if (info_.IsWrite(stmt_id)) {
    auto s = Stmt();
    if (info_.IsGMWrite(stmt_id)) {
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
      s = EmitWrite(node);
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
    return EmitUserStmt(node);
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
  ret += direction_size;
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
        isl::id new_stmt_id = isl::id(stmt_id.ctx(), stmt_id.name().substr(8));
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
              stmts[p_index] = InsertRealize(stmts[p_index], new_stmt_id);
            }
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
  std::stringstream ss;
  ss << iter_expr_new;
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
    std::stringstream ss;
    ss << modify_iter;
  }

  if (isl_cond.as<isl::ast_expr_op_le>()) {
    cond_expr = Simplify(cond_expr + 1);
  }

  Stmt body_stmt = EmitAst(node.get_body());

  if (!body_stmt.defined()) {
    PopIter(iter_expr.get());
    return Stmt();
  }

  if (need_to_modify_inc_) {
    stride_modify_iter_map_.erase(iter_expr_new.get());
  }
  PopIter(iter_expr.get());
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
    return name == THREAD_IDX_X ? thread_cfg->GetX().second
                                : (name == THREAD_IDX_Y ? thread_cfg->GetY().second : thread_cfg->GetZ().second);
  }
  LOG(WARNING) << "Unrecognized thread name " << name;
  return 1;
}

Stmt GpuIslEmitter::Emit(const isl::ast_node &node) {
  Stmt stmt = EmitAst(node);

  // emit realize for temporary tensor
  stmt = EmitRealizeForGlobalTensor(stmt);

  // iter var node attr emit
  std::map<std::string, VarExpr>::iterator it;
  for (it = gpuiter_.begin(); it != gpuiter_.end(); it++) {
    IterVar axis = IterVarNode::make(Range(), it->second, air::kThreadIndex, it->second->name_hint);
    stmt = AttrStmt::make(axis, "thread_extent", Expr(GetThreadExtent(it->second->name_hint)), stmt);
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
  return EmitAst(node.get_node());
}

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

VarExpr GpuIslEmitter::IterNameAdaptor(std::string name) {
  if (iter_name_map_.find(name) != iter_name_map_.end()) {
    gpuiter_[name] = iter_name_map_[name];
    return iter_name_map_[name];
  } else {
    return VarExpr(name);
  }
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

}  // namespace poly
}  // namespace ir
}  // namespace akg
