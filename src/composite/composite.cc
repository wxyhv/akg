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
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include "build_module.h"
#include "composite/util.h"
#include "dmlc/logging.h"
#include "dmlc/common.h"
#include "picojson.h"
#include "topi/broadcast.h"

namespace akg {
using FuncRefMap = std::unordered_map<FunctionRef, FunctionRef, NodeHash, NodeEqual>;
using FuncRefSet = std::unordered_set<FunctionRef, NodeHash, NodeEqual>;
using FuncTensorMap = std::unordered_map<FunctionRef, Tensor, NodeHash, NodeEqual>;
struct BuildInfo {
  Array<Tensor> tensors;         // topi's output tensor, which should be compute node
  Array<NodeRef> args;           // the composite kernel's inputs and outputs
  Map<Tensor, Buffer> in_binds;  // the tensors which should be in bind
  std::string kernel_name;       // the composite kernel's name
};

struct BuildInfoOpt {
  FuncRefMap inplaces;           // the tensors which should be in bind
  FuncRefMap sames;              // the tensors which are same
  FuncRefSet fakeout;            // the tensors which are not output
  std::vector<Tensor> sch_only;  // the tensors which should only used in sch, not output
};

std::tuple<std::string, picojson::array, picojson::array, picojson::array> ParseInputJson(
  const picojson::value &input_json) {
  picojson::array input_desc;
  picojson::array output_desc;
  picojson::array op_desc;
  std::string kernel_name;
  const picojson::value::object &input_obj = input_json.get<picojson::object>();
  for (const auto &item : input_obj) {
    if (item.first == "op") {
      CHECK(item.second.is<std::string>());
      kernel_name = item.second.get<std::string>();
    } else if (item.first == "input_desc") {
      CHECK(item.second.is<picojson::array>());
      input_desc = item.second.get<picojson::array>();
    } else if (item.first == "output_desc") {
      CHECK(item.second.is<picojson::array>());
      output_desc = item.second.get<picojson::array>();
    } else if (item.first == "op_desc") {
      CHECK(item.second.is<picojson::array>());
      op_desc = item.second.get<picojson::array>();
    }
  }
  return std::make_tuple(kernel_name, input_desc, output_desc, op_desc);
}

struct OpDesc {
  std::string op_name;
  std::string fusion_op_name;
  Map<std::string, NodeRef> attrs;
  Array<NodeRef> input_descs;
  Array<NodeRef> output_descs;
};

class OpDescsParser {
 public:
  explicit OpDescsParser(picojson::array op_descs_json) : op_descs_json_(std::move(op_descs_json)) {}
  ~OpDescsParser() = default;

  void Parse() {
    for (const auto &item : op_descs_json_) {
      CHECK(item.is<picojson::object>());
      const picojson::object &op_desc = item.get<picojson::object>();
      ParseOpDesc(op_desc);
    }
  }

  void Dump() {
    LOG(INFO) << "========OP_DESCS========";
    for (const auto &item : op_descs_) {
      LOG(INFO) << "op_name: " << item.op_name;
      LOG(INFO) << "fusion_op_name: " << item.fusion_op_name;
      for (const auto &attr : item.attrs) {
        LOG(INFO) << "attrs: " << attr.first << ":" << attr.second;
      }
      for (const auto &input : item.input_descs) {
        LOG(INFO) << "input: " << input;
      }
      for (const auto &output : item.output_descs) {
        LOG(INFO) << "output: " << output;
      }
    }
  }

 public:
  std::vector<OpDesc> op_descs_;

 private:
  const picojson::array op_descs_json_;
  std::unordered_map<std::string, Tensor> tensor_map_;

 private:
  static void ParseTensorValue(const picojson::value &tensor_value, const std::string &tensor_name,
                               const Array<Expr> &shape, const Type &type, Array<NodeRef> &input_output) {
    CHECK_EQ(shape.size(), 1) << "We should not make a expr for a not const tensor.";
    CHECK(Equal(shape[0], Expr(1))) << "We should not make a expr for a not const tensor.";
    CHECK(!tensor_value.is<picojson::null>()) << "We should has default value of tensor(expr): " << tensor_name;
    if (tensor_value.is<double>()) {
      input_output.push_back(make_const(type, tensor_value.get<double>()));
    } else if (tensor_value.is<int64_t>()) {
      input_output.push_back(make_const(type, tensor_value.get<int64_t>()));
    } else {
      CHECK(0) << "Unknown value type of tensor: " << tensor_name;
    }
  }

  void ParseTensor(const picojson::object &tensor_desc, Array<NodeRef> &input_output) {
    std::string tensor_name;
    Array<Expr> shape;
    Type type;
    for (const auto &item : tensor_desc) {
      if (item.first == "tensor_name") {
        CHECK(item.second.is<std::string>());
        tensor_name = item.second.get<std::string>();
      } else if (item.first == "shape") {
        CHECK(item.second.is<picojson::array>());
        const picojson::array &dims = item.second.get<picojson::array>();
        for (const auto &dim : dims) {
          CHECK(dim.is<int64_t>());
          shape.push_back(Expr(static_cast<int>(dim.get<int64_t>())));
        }
      } else if (item.first == "data_type") {
        CHECK(item.second.is<std::string>());
        std::string dtype_str = item.second.get<std::string>();
        if (type_mapping.find(dtype_str) == type_mapping.end()) {
          LOG(FATAL) << "Not support dtype str " << dtype_str;
        }
        type = type_mapping[dtype_str];
      }
    }

    for (const auto &item : tensor_desc) {
      if (item.first == "value" && !item.second.is<picojson::null>()) {
        picojson::value tensor_value = item.second;
        ParseTensorValue(tensor_value, tensor_name, shape, type, input_output);
        return;
      }
    }

    if (tensor_map_.count(tensor_name) == 0) {
      Tensor t = placeholder(shape, type, tensor_name);
      tensor_map_[tensor_name] = t;
    }
    input_output.push_back(tensor_map_[tensor_name]);
  }

  void ParseInputTensors(const picojson::array &tensor_descs, Array<NodeRef> &input) {
    for (const auto &tensor_desc_l0 : tensor_descs) {
      CHECK(tensor_desc_l0.is<picojson::array>());
      const picojson::array &tensor_desc_l1 = tensor_desc_l0.get<picojson::array>();
      ParseTensors(tensor_desc_l1, input);
    }
  }

  void ParseTensors(const picojson::array &tensor_descs, Array<NodeRef> &tensors) {
    for (const auto &tensor_desc : tensor_descs) {
      CHECK(tensor_desc.is<picojson::object>());
      const picojson::object &tensor_desc_info = tensor_desc.get<picojson::object>();
      ParseTensor(tensor_desc_info, tensors);
    }
  }

  void ParseOpDesc(const picojson::object &op_desc) {
    OpDesc op_desc_info;
    auto it = op_desc.find("fusion");
    if (it != op_desc.end()) {
      op_desc_info.fusion_op_name = it->second.get<std::string>();
    }
    it = op_desc.find("name");
    if (it != op_desc.end()) {
      op_desc_info.op_name = it->second.get<std::string>();
    }
    it = op_desc.find("attr");
    if (it != op_desc.end() && it->second.is<picojson::array>()) {
      const picojson::array &attrs = it->second.get<picojson::array>();
      ParseAttrs(attrs, &op_desc_info.attrs);
    }
    it = op_desc.find("input_desc");
    if (it != op_desc.end() && it->second.is<picojson::array>()) {
      const picojson::array &input_descs = it->second.get<picojson::array>();
      ParseInputTensors(input_descs, op_desc_info.input_descs);
    }
    it = op_desc.find("output_desc");
    if (it != op_desc.end() && it->second.is<picojson::array>()) {
      const picojson::array &output_descs = it->second.get<picojson::array>();
      ParseTensors(output_descs, op_desc_info.output_descs);
    }
    op_descs_.emplace_back(op_desc_info);
  }

  static void ParseAttrs(const picojson::array &arr, Map<std::string, NodeRef> *op_attrs) {
    CHECK(op_attrs) << "input op_attrs is invalid.";
    for (const auto &item : arr) {
      CHECK(item.is<picojson::object>());
      const picojson::object &obj = item.get<picojson::object>();
      std::string name;
      NodeRef value;
      bool name_found = false;
      bool value_found = false;
      for (const auto &kv : obj) {
        // parse attr name
        if (kv.first == "name") {
          name = kv.second.get<std::string>();
          name_found = true;
          continue;
        }
        if (kv.first != "value") {
          continue;
        }
        // parse attr value
        value_found = true;
        if (kv.second.is<picojson::array>()) {
          Array<NodeRef> arr_v;
          const picojson::array &arr_s = kv.second.get<picojson::array>();
          for (const auto &v : arr_s) {
            if (v.is<int64_t>()) {
              arr_v.push_back(Integer(static_cast<int>(v.get<int64_t>())));
            } else if (v.is<std::string>()) {
              arr_v.push_back(StringImm::make(v.get<std::string>()));
            } else {
              LOG(FATAL) << "Not parsed type in array attr.";
            }
          }
          value = arr_v;
        } else if (kv.second.is<bool>()) {
          value = make_const(Int(1), kv.second.get<bool>());
        } else if (kv.second.is<int64_t>()) {
          value = Integer(static_cast<int>(kv.second.get<int64_t>()));
        } else if (kv.second.is<std::string>()) {
          value = StringImm::make(kv.second.get<std::string>());
        } else {
          LOG(FATAL) << "Not parsed type in op_attrs.";
        }
      }
      CHECK(name_found);
      CHECK(value_found);
      op_attrs->Set(name, value);
    }
  }
};

Stmt MakeStmt(const std::vector<OpDesc> &op_descs) {
  std::vector<Stmt> stmts;
  for (const auto &op_desc : op_descs) {
    Array<Expr> input;
    for (const auto &item : op_desc.input_descs) {
      if (item.as<TensorNode>()) {
        auto t = Downcast<Tensor>(item);
        input.push_back(Call::make(t->dtype, t->op->name, t->shape, Call::CallType::Halide, t->op));
      } else {
        input.push_back(Downcast<Expr>(item));
      }
    }
    Tensor output = Downcast<Tensor>(op_desc.output_descs[0]);
    auto op_name = op_desc.op_name;
    auto stmt =
      Provide::make(output->op, 0, Call::make(Int(32), op_name, input, Call::CallType::PureIntrinsic), output->shape);
    if (!op_desc.attrs.empty()) {
      stmt = AttrStmt::make(op_desc.attrs, "attrs", Expr(1), stmt);
    }
    if (!op_desc.fusion_op_name.empty()) {
      stmt = AttrStmt::make(make_zero(Int(32)), "fusion", op_desc.fusion_op_name, stmt);
    }
    stmts.emplace_back(stmt);
  }
  return Block::make(stmts);
}

class InplaceAssignMutator : public IRMutator {
 public:
  explicit InplaceAssignMutator(BuildInfoOpt &opt) : opt_(opt) {}

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "attrs") {
      op_attrs_ = Downcast<Map<std::string, NodeRef>>(op->node);
      auto stmt = IRMutator::Mutate_(op, s);
      op_attrs_ = {};
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    CHECK(op->value.as<Call>());
    auto call = op->value.as<Call>();
    auto op_name = call->name;
    if (op_name == "InplaceAssign") {
      if (op_attrs_.count("fake_output")) {
        auto fake_val = op_attrs_["fake_output"].as<IntImm>();
        if (fake_val && fake_val->value > 0) {
          opt_.fakeout.insert(op->func);
        }
      }
      auto inputs = call->args;
      opt_.sames[op->func] = inputs[2].as<Call>()->func;  // d = InplaceAssign(a, b, c)     d = c
      if (auto i1 = inputs[1].as<Call>()) {
        opt_.inplaces[i1->func] = inputs[0].as<Call>()->func;  // d = InplaceAssign(a, b, c)     a = b
        return Evaluate::make(0);
      } else {
        // d = Assign(dst, src)    d = dst   fake d, d should be InplaceAssigin's inputs[2]
        return Provide::make(op->func, op->value_index,
                             Call::make(call->type, "Assign", {inputs[0], inputs[1]}, call->call_type), op->args);
      }
    }
    return IRMutator::Mutate_(op, s);
  }
  BuildInfoOpt &opt_;
  Map<std::string, NodeRef> op_attrs_;
};

class FusionMutator : public IRMutator {
 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "fusion" && op->value.as<StringImm>()) {
      fusion_op_name_ = op->value.as<StringImm>()->value;
      return IRMutator::Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    if (!fusion_op_name_.empty()) {
      CHECK(op->value.as<Call>());
      auto call = op->value.as<Call>();
      if (fusion_op_name_.find("_end") == std::string::npos) {
        if (call->name == "ZerosLike") {  // ZerosLike directly transform to zero
          CHECK_EQ(call->args.size(), 1);
          CHECK(call->args[0].as<Call>());
          output_with_inputs_[op->func] = {make_zero(call->args[0].as<Call>()->type)};
        } else {
          output_with_inputs_[op->func] = call->args;
        }
        return Evaluate::make(0);
      } else {  // fusion end
        Array<Expr> fusion_inputs;
        GetFusionOpInputs(call->args, fusion_inputs);
        auto str_list = dmlc::Split(fusion_op_name_, '_');
        CHECK(!str_list.empty());
        auto stmt =
          Provide::make(op->func, op->value_index,
                        Call::make(Int(32), str_list[0], fusion_inputs, Call::CallType::PureIntrinsic), op->args);
        output_with_inputs_.clear();
        fusion_op_name_.clear();
        return stmt;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  void GetFusionOpInputs(const Array<Expr> &inputs, Array<Expr> &fusion_inputs) {
    for (const auto &item : inputs) {
      if (auto c = item.as<Call>()) {
        if (output_with_inputs_.count(c->func) != 0) {
          for (auto input : output_with_inputs_[c->func]) {
            fusion_inputs.push_back(input);
          }
          continue;
        }
      }
      fusion_inputs.push_back(item);
    }
  }

  std::unordered_map<FunctionRef, Array<Expr>, NodeHash, NodeEqual> output_with_inputs_;
  std::string fusion_op_name_;
};

Stmt Optimize(Stmt &s, BuildInfoOpt &opt) {
  // fusion
  s = FusionMutator().Mutate(s);
  // inplace_assign
  s = InplaceAssignMutator(opt).Mutate(s);
  return s;
}

class Emitter : public IRVisitor {
 public:
  Emitter(FuncTensorMap &tensor_map, BuildInfoOpt &opt) : tensor_map_(tensor_map), opt_(opt) {}
  void Visit_(const AttrStmt *op) override {
    if (op->attr_key == "attrs") {
      op_attrs_ = Downcast<Map<std::string, NodeRef>>(op->node);
      Visit(op->body);
      op_attrs_ = {};
    }
  }
  void Visit_(const Provide *op) override {
    CHECK(op->value.as<Call>());
    auto call = op->value.as<Call>();
    auto op_name = call->name;
    auto inputs = call->args;
    Array<NodeRef> real_input;
    for (const auto &input : inputs) {
      if (auto c = input.as<Call>()) {
        if (tensor_map_.count(c->func) == 0) {
          Tensor t = placeholder(c->args, c->type, c->name);
          tensor_map_[c->func] = t;
        }
        real_input.push_back(tensor_map_[c->func]);
      } else {
        real_input.push_back(input);
      }
    }
    const auto *topi_f = air::runtime::Registry::Get(op_name);
    CHECK(topi_f) << "Akg topi has no op: " << op_name;
    Tensor t = (*topi_f)(real_input, op_attrs_);
    if (op_name == "Assign") {
      EmitAssign(t, inputs[0]);
    }

    LOG(INFO) << op->func->func_name() << " = " << op_name << "(" << inputs << ")"
              << "\n>>>>>>>\n"
              << t->op->func_name() << " = " << op_name << "(" << real_input << ")";
    tensor_map_[op->func] = t;
  }

  void EmitAssign(Tensor &t, const Expr &input) {
    // copy out to bind_input, bind_input is used to bind input[0]
    // d = Assign(a, b), bind_input = d, input0 = bind_input
    auto bind_input = compute(t->shape, [&](const Array<Var> &indices) { return t(indices); });
    tensor_map_[bind_input->op] = bind_input;
    opt_.sch_only.emplace_back(bind_input);
    opt_.inplaces[bind_input->op] = input.as<Call>()->func;
  }

 private:
  FuncTensorMap &tensor_map_;
  BuildInfoOpt &opt_;
  Map<std::string, NodeRef> op_attrs_;
};

void ParseInputTensors(const picojson::array &input_descs, std::vector<std::string> &input_tensors) {
  for (auto input_desc = input_descs.begin(); input_desc != input_descs.end(); ++input_desc) {
    CHECK(input_desc->is<picojson::array>());
    const picojson::array &input_desc_array = input_desc->get<picojson::array>();
    CHECK(input_desc_array.begin()->is<picojson::object>());
    const picojson::object &input_desc_obj = input_desc_array.begin()->get<picojson::object>();
    for (const auto &item : input_desc_obj) {
      if (item.first != "tensor_name") continue;
      CHECK(item.second.is<std::string>());
      std::string tensor_name = item.second.get<std::string>();
      input_tensors.emplace_back(tensor_name);
    }
  }
}

void ParseOutputTensors(const picojson::array &output_descs, std::vector<std::string> &output_tensors) {
  for (auto output_desc = output_descs.begin(); output_desc != output_descs.end(); ++output_desc) {
    CHECK(output_desc->is<picojson::object>());
    const picojson::object &output_desc_obj = output_desc->get<picojson::object>();
    for (const auto &item : output_desc_obj) {
      if (item.first != "tensor_name") continue;
      CHECK(item.second.is<std::string>());
      std::string tensor_name = item.second.get<std::string>();
      output_tensors.emplace_back(tensor_name);
    }
  }
}

void CollectBinds(FuncTensorMap &tensor_map, BuildInfoOpt &opt, BuildInfo &info) {
  for (const auto &kv : opt.inplaces) {
    auto first = tensor_map[kv.first];
    auto second = tensor_map[kv.second];
    auto buf = decl_buffer(first->shape, first->dtype, first->op->name);
    info.in_binds.Set(first, buf);
    info.in_binds.Set(second, buf);
  }
}

void ProcessSames(FuncTensorMap &tensor_map, BuildInfoOpt &opt) {
  // b = func(a)
  // c = InplaceAssign(x, y, b)     c = b
  // d = InplaceAssign(i, j, c)     d = c
  while (!opt.sames.empty()) {
    for (auto it = opt.sames.begin(); it != opt.sames.end();) {
      if (tensor_map.count(it->second)) {
        tensor_map[it->first] = tensor_map[it->second];
        it = opt.sames.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void CollectInputs(const picojson::array &input_desc, FuncTensorMap &tensor_map, BuildInfo &info) {
  std::vector<std::string> input_tensors;
  ParseInputTensors(input_desc, input_tensors);
  for (const auto &input : input_tensors) {
    auto iter =
      std::find_if(tensor_map.begin(), tensor_map.end(),
                   [&input](const std::pair<const FunctionRef, Tensor> &kv) { return kv.first->func_name() == input; });
    CHECK(iter != tensor_map.end()) << "input Tensor " << input << " not built.";
    LOG(INFO) << "input: " << input << " " << iter->second;
    info.args.push_back(iter->second);
  }
}

void CollectOutputsAndComputes(const picojson::array &output_desc, FuncTensorMap &tensor_map, BuildInfoOpt &opt,
                               BuildInfo &info) {
  std::vector<std::string> output_tensors;
  ParseOutputTensors(output_desc, output_tensors);
  for (const auto &output : output_tensors) {
    auto iter = std::find_if(
      tensor_map.begin(), tensor_map.end(),
      [&output](const std::pair<const FunctionRef, Tensor> &kv) { return kv.first->func_name() == output; });
    CHECK(iter != tensor_map.end()) << "output Tensor " << output << " not built.";
    LOG(INFO) << "output: " << output << " " << iter->second;
    info.tensors.push_back(iter->second);
    if (!opt.fakeout.count(iter->first)) {
      info.args.push_back(iter->second);
    }
  }
}

void CollectSchOnlyComputes(BuildInfoOpt &opt, BuildInfo &info) {
  for (const auto &tensor : opt.sch_only) {
    info.tensors.push_back(tensor);
  }
}

void CollectBuildInfo(const picojson::array &input_desc, const picojson::array &output_desc, FuncTensorMap &tensor_map,
                      BuildInfoOpt &opt, BuildInfo &info) {
  CollectBinds(tensor_map, opt, info);
  ProcessSames(tensor_map, opt);
  CollectInputs(input_desc, tensor_map, info);
  CollectOutputsAndComputes(output_desc, tensor_map, opt, info);
  CollectSchOnlyComputes(opt, info);
}

void ExtractBuildInfo(const picojson::value &input_json, BuildInfo &info) {
  CHECK(input_json.is<picojson::object>());
  picojson::array input_desc;
  picojson::array output_desc;
  picojson::array op_descs;
  std::string kernelname;
  // 1. parse input json
  std::tie(kernelname, input_desc, output_desc, op_descs) = ParseInputJson(input_json);
  info.kernel_name = kernelname;
  // 2. parse op descs
  auto parser = OpDescsParser(op_descs);
  parser.Parse();
  // 3. make stmt by op descs
  auto stmt = MakeStmt(parser.op_descs_);
  LOG(INFO) << "\n========STMT START========\n" << stmt << "\n========STMT END========\n";
  // 4. optimize stmt
  BuildInfoOpt opt;
  stmt = Optimize(stmt, opt);
  LOG(INFO) << "\n========OPTIMIZED STMT START========\n" << stmt << "\n========OPTIMIZED STMT END========\n";
  // 5. emit stmt by topi
  FuncTensorMap tensor_map;
  Emitter(tensor_map, opt).Visit(stmt);
  // 6. collect build info: args, compute, binds
  CollectBuildInfo(input_desc, output_desc, tensor_map, opt, info);
}

int ExtractKernelNum(const picojson::value &v) {
  int kernel_num = 0;
  CHECK(kernel_num) << "input kernel_num is invalid.";
  CHECK(v.is<picojson::object>());
  const picojson::value::object &obj = v.get<picojson::object>();
  for (auto i = obj.begin(); i != obj.end(); ++i) {
    if (i->first == "core_num") {
      CHECK(i->second.is<int64_t>());
      kernel_num = i->second.is<int64_t>();
      break;
    }
  }
  return kernel_num;
}

NodeRef CompositeWithJsonToFunc(const std::string &json_str, Map<std::string, NodeRef> attrs) {
  picojson::value v;
  std::string err = picojson::parse(v, json_str);
  if (!err.empty()) {
    LOG(ERROR) << "json parse error, error message: " << err;
  }
  const char *akg_dump_pass_ir = getenv("MS_AKG_DUMP_IR");
  BuildInfo info;
  ExtractBuildInfo(v, info);
  Array<Operation> ops;
  std::for_each(info.tensors.begin(), info.tensors.end(), [&ops](const Tensor &t) { ops.push_back(t->op); });
  Schedule sch = create_schedule(ops);
  akg::BuildConfig config = akg::BuildConfig::Current();
  CHECK(config.defined());
  config->dump_pass_ir = akg_dump_pass_ir != nullptr;
  attrs.Set("pragma_reschedule", make_const(Int(32), 1));
  Array<NodeRef> shape_vars;
  auto build_rst =
    akg::BuildToFunc(sch, info.args, shape_vars, info.kernel_name, info.in_binds, attrs, true, "cce", config);
  CHECK(build_rst.defined());
  return std::move(build_rst);
}

std::string GetProcess(const std::string &json_str) {
  size_t pos = json_str.find("\"process\"");
  if (pos != std::string::npos && json_str.find("cuda", pos) != std::string::npos) {
    return "cuda";
  }
  return "aicore";
}

std::string GetSchedule(Array<Tensor> &outputs) {
  for (const Tensor &t : outputs) {
    if (t->op->tag == "comm_reduce" || t->op->tag == "comm_reduce_idx") {
      return "reduce";
    }
  }
  return "injective";
}

Module CompositeWithJsonGpu(const std::string &json_str, const Map<std::string, NodeRef> &attrs, bool poly) {
  picojson::value v;
  std::string err = picojson::parse(v, json_str);
  if (!err.empty()) {
    LOG(ERROR) << "json parse error, error message: " << err;
  }
  BuildInfo info;
  ExtractBuildInfo(v, info);
  const auto *build_func = air::runtime::Registry::Get("akg_build_gpu_module");
  CHECK(build_func != nullptr);
  std::string sch = GetSchedule(info.tensors);
  return (*build_func)(info.tensors, info.args, sch, info.kernel_name, attrs, poly, info.in_binds);
}

Module CompositeWithJson(const std::string &json_str, const Map<std::string, NodeRef> &attrs, bool poly) {
  if (GetProcess(json_str) == "cuda") {
    return CompositeWithJsonGpu(json_str, attrs, poly);
  }
  auto build_rst = CompositeWithJsonToFunc(json_str, attrs);
  return BuildToModule(build_rst);
}

NodeRef CompositeLower(const std::string &json_str, const Map<std::string, NodeRef> &attrs) {
  picojson::value v;
  std::string err = picojson::parse(v, json_str);
  if (!err.empty()) {
    LOG(ERROR) << "json parse error, error message: " << err;
  }
  BuildInfo info;
  ExtractBuildInfo(v, info);
  Array<Operation> ops;
  std::for_each(info.tensors.begin(), info.tensors.end(), [&ops](const Tensor &t) { ops.push_back(t->op); });
  Schedule sch = create_schedule(ops);
  akg::BuildConfig config = akg::BuildConfig::Current();
  CHECK(config.defined());
  bool tuning = attrs.find("tuning") != attrs.end();
  Array<NodeRef> shape_vars;
  return akg::Lower(sch, info.args, shape_vars, info.kernel_name, info.in_binds, attrs, false, true, tuning, "cce",
                    config);
}

Module CompositeWithJsonListGpu(const Array<NodeRef> &json_str_node, const Array<NodeRef> &args_list_node,
                                const Map<std::string, NodeRef> &attrs, bool poly) {
  std::vector<Stmt> all_irs;
  Array<NodeRef> all_args, ordered_args, input_args, output_args;
  std::vector<std::string> args_list_name;
  std::string merge_name;
  size_t idx = 0;
  const char *akg_dump_pass_ir = getenv("MS_AKG_DUMP_IR");

  // get origin json all input name
  for (auto arg : args_list_node) {
    auto arg_name = arg.as<StringImm>()->value;
    args_list_name.emplace_back(arg_name);
  }

  // traversal json_str_node and parse every subgraph json
  for (auto k : json_str_node) {
    ++idx;
    auto json_str = k.as<StringImm>()->value;
    picojson::value v;
    std::string err = picojson::parse(v, json_str);
    CHECK(err.empty()) << "json parse error, error message: " << err;
    Array<NodeRef> args, arg_list_0, shape_vars;
    Map<Tensor, Buffer> binds, binds_0;
    BuildInfo info;
    ExtractBuildInfo(v, info);

    // ensure merge_name is the same as original json name
    if (merge_name.empty()) {
      merge_name = info.kernel_name;
    }

    // use idx to distinct different subgraph
    std::string distinct_name = info.kernel_name + "_" + std::to_string(idx);
    std::string sch_name = GetSchedule(info.tensors);
    const auto *sch_create = air::runtime::Registry::Get("select_cuda_scheduler");
    CHECK(sch_create != nullptr);
    Schedule sch = (*sch_create)(info.tensors, sch_name, poly);

    akg::BuildConfig config = akg::BuildConfig::Current();
    CHECK(config.defined());
    config->dump_pass_ir = akg_dump_pass_ir != nullptr;
    Stmt s_ir = LowerStmt(sch, info.args, shape_vars, distinct_name, info.in_binds, attrs, false, poly, false, "cuda",
                          config, &args, &arg_list_0, &binds, &binds_0, true);
    for (const auto &x : arg_list_0) {
      all_args.push_back(x);
    }
    all_irs.emplace_back(s_ir);
  }
  // reorder args_list, now args_list satisfies: op1_input op2_input ... op1_output op2_output ...
  // suppose all args info from original json satisfies this order
  for (auto arg : all_args) {
    bool find = false;
    for (const auto &name : args_list_name) {
      auto buffer = arg.as<BufferNode>();
      CHECK(buffer) << "arg must be a BufferNode";
      if (buffer->name == name) {
        find = true;
        input_args.push_back(arg);
        break;
      }
    }
    if (!find) {
      output_args.push_back(arg);
    }
  }
  for (auto input : input_args) {
    ordered_args.push_back(input);
  }
  for (auto output : output_args) {
    ordered_args.push_back(output);
  }

  akg::BuildConfig final_config = akg::BuildConfig::Current();
  CHECK(final_config.defined());
  final_config->dump_pass_ir = akg_dump_pass_ir != nullptr;

  // use ordered_args to second stage build
  auto rst = LowerFunc(all_irs[0], merge_name, final_config, ordered_args);
  auto build_rst = BuildRstNode::make(rst, merge_name);
  return BuildToModule(build_rst, "cuda");
}
TVM_REGISTER_GLOBAL("composite_with_json_to_func").set_body_typed(CompositeWithJsonToFunc);
TVM_REGISTER_GLOBAL("composite_with_json").set_body_typed(CompositeWithJson);
TVM_REGISTER_GLOBAL("composite_with_json_list_gpu").set_body_typed(CompositeWithJsonListGpu);
TVM_REGISTER_GLOBAL("composite_lower").set_body_typed(CompositeLower);
}  // namespace akg
