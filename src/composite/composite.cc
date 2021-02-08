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
#include "dmlc/common.h"
#include "build_module.h"
#include "composite/block_fusion.h"
#include "composite/util.h"
#include "composite/optimize/optimize.h"
#include "composite/stitch_fusion.h"

namespace akg {
std::tuple<std::string, std::string, picojson::array, picojson::array, picojson::array> ParseInputJson(
  const picojson::value &input_json) {
  picojson::array input_desc;
  picojson::array output_desc;
  picojson::array op_desc;
  std::string kernel_name;
  std::string target;
  const picojson::value::object &input_obj = input_json.get<picojson::object>();
  for (const auto &item : input_obj) {
    if (item.first == "op") {
      CHECK(item.second.is<std::string>());
      kernel_name = item.second.get<std::string>();
    } else if (item.first == "process") {
      CHECK(item.second.is<std::string>());
      target = item.second.get<std::string>();
    } else if (item.first == "input_desc") {
      if (item.second.is<picojson::null>()) {
        continue;
      }
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
  return std::make_tuple(kernel_name, target, input_desc, output_desc, op_desc);
}

class OpDescsParser {
 public:
  OpDescsParser(picojson::array op_descs_json, const std::vector<std::string> &input_tensors,
                const std::vector<std::string> &output_tensors)
      : op_descs_json_(std::move(op_descs_json)), input_tensors_(input_tensors), output_tensors_(output_tensors) {}
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
      for (const auto &input_info : item.input_tensor_info) {
        LOG(INFO) << "input_info: ";
        LOG(INFO) << input_info.name_;
        LOG(INFO) << input_info.shape_;
      }
      for (const auto &output_info : item.output_tensor_info) {
        LOG(INFO) << "output_info: ";
        LOG(INFO) << output_info.name_;
        LOG(INFO) << output_info.shape_;
      }
    }
  }

 public:
  std::vector<OpDesc> op_descs_;
  FuncRefSet input_funcs_;
  FuncRefList output_funcs_;

 private:
  const picojson::array op_descs_json_;
  const std::vector<std::string> input_tensors_;
  const std::vector<std::string> output_tensors_;
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

  void MakeTensors(const std::vector<TensorInfo> &tensor_info, Array<NodeRef> &tensors) {
    for (const auto &info : tensor_info) {
      if (info.has_value_) {
        // In case when current tensor already has value information
        ParseTensorValue(info.value_, info.name_, info.shape_, info.dtype_, tensors);
        continue;
      }
      if (tensor_map_.count(info.name_) == 0) {
        Tensor t = placeholder(info.shape_, info.dtype_, info.name_);
        tensor_map_[info.name_] = t;
        if (std::find(input_tensors_.begin(), input_tensors_.end(), info.name_) != input_tensors_.end()) {
          input_funcs_.insert(t->op);
        }
        if (std::find(output_tensors_.begin(), output_tensors_.end(), info.name_) != output_tensors_.end()) {
          output_funcs_.emplace_back(t->op);
        }
      }
      tensors.push_back(tensor_map_[info.name_]);
    }
  }

  void ParseTensorInfo(const picojson::object &tensor_desc, std::vector<TensorInfo> &tensor_info) {
    TensorInfo info;
    for (const auto &item : tensor_desc) {
      if (item.first == "tensor_name") {
        CHECK(item.second.is<std::string>());
        info.name_ = item.second.get<std::string>();
      } else if (item.first == "format") {
        CHECK(item.second.is<std::string>());
        info.format_ = item.second.get<std::string>();
      } else if (item.first == "shape") {
        CHECK(item.second.is<picojson::array>());
        const picojson::array &dims = item.second.get<picojson::array>();
        for (const auto &dim : dims) {
          CHECK(dim.is<int64_t>());
          info.shape_.push_back(Expr(static_cast<int>(dim.get<int64_t>())));
        }
      } else if (item.first == "data_type") {
        CHECK(item.second.is<std::string>());
        std::string dtype_str = item.second.get<std::string>();
        if (type_mapping.find(dtype_str) == type_mapping.end()) {
          LOG(FATAL) << "Not support dtype str " << dtype_str;
        }
        info.dtype_ = type_mapping[dtype_str];
      } else if (item.first == "value" && !item.second.is<picojson::null>()) {
        info.has_value_ = true;
        info.value_ = item.second;
      }
    }

    tensor_info.emplace_back(info);
  }

  void ParseInputTensors(const picojson::array &tensor_descs, OpDesc &op_desc_info) {
    Map<std::string, NodeRef> &attrs = op_desc_info.attrs;
    std::vector<TensorInfo> tensor_info;
    for (const auto &tensor_desc_l0 : tensor_descs) {
      CHECK(tensor_desc_l0.is<picojson::array>());
      const picojson::array &tensor_desc_l1 = tensor_desc_l0.get<picojson::array>();
      for (const auto &tensor_desc : tensor_desc_l1) {
        CHECK(tensor_desc.is<picojson::object>());
        const picojson::object &tensor_desc_info = tensor_desc.get<picojson::object>();
        ParseTensorInfo(tensor_desc_info, tensor_info);
      }
    }

    // Gather data format information of input tensors
    for (const auto &info : tensor_info) {
      if (!info.format_.empty()) {
        auto key = CreateDataFormatKey(info.name_);
        auto format = StringImm::make(info.format_);
        if (attrs.find(key) != attrs.end()) {
          LOG(WARNING) << key << " already exists in attrs";
        }
        attrs.Set(key, format);
      }
    }
    op_desc_info.input_tensor_info = tensor_info;
    MakeTensors(tensor_info, op_desc_info.input_descs);
  }

  void ParseOutputTensors(const picojson::array &tensor_descs, OpDesc &op_desc_info) {
    std::vector<TensorInfo> tensor_info;
    for (const auto &tensor_desc : tensor_descs) {
      CHECK(tensor_desc.is<picojson::object>());
      const picojson::object &tensor_desc_info = tensor_desc.get<picojson::object>();
      ParseTensorInfo(tensor_desc_info, tensor_info);
    }
    op_desc_info.output_tensor_info = tensor_info;
    MakeTensors(tensor_info, op_desc_info.output_descs);
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
      ParseInputTensors(input_descs, op_desc_info);
    }
    it = op_desc.find("output_desc");
    if (it != op_desc.end() && it->second.is<picojson::array>()) {
      const picojson::array &output_descs = it->second.get<picojson::array>();
      ParseOutputTensors(output_descs, op_desc_info);
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
        opt_.inplaces[i1->func] = inputs[0];  // d = InplaceAssign(a, b, c)     a = b
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

class BroadcastInserter : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "attrs" && op->body.as<Provide>()) {
      const Provide *provide = op->body.as<Provide>();
      CHECK(provide);
      auto call = provide->value.as<Call>();
      CHECK(call);
      auto it = broadcast_ops_.find(call->name);
      if (it != broadcast_ops_.end()) {
        for (size_t i = 0; i < call->args.size(); ++i) {
          if (!(it->second & (1u << i))) {
            continue;
          }
          Expr e = call->args[i];
          if (e.as<IntImm>() || e.as<UIntImm>() || e.as<FloatImm>()) {
            Stmt first, second;
            std::string name = "broadcast_" + std::to_string(name_idx_++);
            Tensor t = placeholder(provide->args, call->type, name);
            first =
              Provide::make(t->op, 0, Call::make(Int(32), "BroadcastTo", {e}, Call::CallType::PureIntrinsic), t->shape);
            Map<std::string, NodeRef> attrs = Downcast<Map<std::string, NodeRef>>(op->node);
            attrs.Set("shape", t->shape);
            first = AttrStmt::make(attrs, "attrs", Expr(1), first);
            auto args = call->args;
            args.Set(i, Call::make(t->dtype, t->op->name, t->shape, Call::CallType::Halide, t->op));
            second = Provide::make(provide->func, provide->value_index,
                                   Call::make(call->type, call->name, args, call->call_type), provide->args);
            second = AttrStmt::make(op->node, op->attr_key, op->value, second);
            return Block::make(first, second);
          }
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  int name_idx_ = 0;
  std::unordered_map<std::string, unsigned> broadcast_ops_ = {{"Equal", -1}, {"Select", -1}, {"Cast", -1}};
};

class TypeCastInserter : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "attrs" && op->body.as<Provide>()) {
      const Provide *provide = op->body.as<Provide>();
      CHECK(provide);
      auto call = provide->value.as<Call>();
      CHECK(call);
      auto it = typecast_ops_.find(call->name);
      if (it != typecast_ops_.end() && call->type == Int(32)) {
        CHECK_EQ(call->args.size(), 2);
        auto input0 = call->args[0];
        auto input1 = call->args[1];
        Tensor t0 = placeholder(provide->args, Float(32), "equal_input1");
        Tensor t1 = placeholder(provide->args, Float(32), "equal_input2");
        Tensor t2 = placeholder(provide->args, Float(32), "equal_output");
        Map<std::string, NodeRef> attrs0, attrs1, attrs2, attrs3;
        attrs0.Set("dst_type", StringImm::make("float32"));
        attrs1.Set("dst_type", StringImm::make("float32"));
        attrs3.Set("dst_type", StringImm::make("float32"));

        auto arg0 = Call::make(t0->dtype, t0->op->name, t0->shape, Call::CallType::Halide, t0->op);
        auto arg1 = Call::make(t1->dtype, t1->op->name, t1->shape, Call::CallType::Halide, t1->op);
        auto arg2 = Call::make(t2->dtype, t2->op->name, t2->shape, Call::CallType::Halide, t2->op);
        auto cast0 = Call::make(Int(32), "Cast", {input0}, Call::CallType::Intrinsic);
        auto cast1 = Call::make(Int(32), "Cast", {input1}, Call::CallType::Intrinsic);
        auto equal_op = Call::make(Float(32), "Equal", {arg0, arg1}, Call::CallType::Intrinsic);
        auto assign_cast0 = Provide::make(t0->op, 0, cast0, provide->args);
        auto assign_cast1 = Provide::make(t1->op, 0, cast1, provide->args);
        auto assign_equal = Provide::make(t2->op, 0, equal_op, provide->args);
        auto value_int32 = Call::make(Float(32), "Cast", {arg2}, Call::CallType::Intrinsic);
        auto new_provide = Provide::make(provide->func, provide->value_index, value_int32, provide->args);
        auto new_attr0 = AttrStmt::make(attrs0, "attrs", Expr(1), assign_cast0);
        auto new_attr1 = AttrStmt::make(attrs1, "attrs", Expr(1), assign_cast1);
        auto new_attr2 = AttrStmt::make(attrs2, "attrs", Expr(1), assign_equal);
        auto new_attr3 = AttrStmt::make(attrs3, "attrs", Expr(1), new_provide);
        auto new_body = Block::make(Block::make(new_attr0, new_attr1), Block::make(new_attr2, new_attr3));
        auto new_attr = AttrStmt::make(op->node, op->attr_key, op->value, new_body);
        return new_attr;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::unordered_map<std::string, unsigned> typecast_ops_ = {
    {"Equal", -1},
  };
};

Stmt Optimize(Stmt &s, BuildInfoOpt &opt, const FuncRefSet &input_funcs, const FuncRefList &output_funcs,
              const std::string &target) {
  // reshape optimize
  s = ReshapeTensor(s);
  // fusion
  s = FusionMutator().Mutate(s);
  // elemwise opt
  s = ElimTransformOp(s, input_funcs, output_funcs, opt);
  // inplace_assign
  s = InplaceAssignMutator(opt).Mutate(s);
  // insert broadcast
  s = BroadcastInserter().Mutate(s);
  // insert cast for equal(int32) in ascend
  if (target == "aicore") {
    s = TypeCastInserter().Mutate(s);
  }
  return s;
}

class Emitter : public IRVisitor {
 public:
  Emitter(FuncTensorMap &tensor_map, BuildInfoOpt &opt) : tensor_map_(tensor_map), opt_(opt) {}

 private:
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
    if (op_name == "Reshape") {  // reshape's attr may have shape [-1], it will cause error.
      op_attrs_.Set("shape", op->args);
    }
    Tensor t = (*topi_f)(real_input, op_attrs_);
    if (op_name == "Assign") {
      EmitAssign(t, inputs[0]);
    }

    tensor_map_[op->func] = t;
  }

  void EmitAssign(Tensor &t, const Expr &input) {
    // copy out to bind_input, bind_input is used to bind input[0]
    // d = Assign(a, b), bind_input = d, input0 = bind_input
    auto bind_input = compute(
      t->shape, [&](const Array<Var> &indices) { return t(indices); },
      "assign_tensor_" + std::to_string(assign_count_));
    tensor_map_[bind_input->op] = bind_input;
    opt_.sch_only.emplace_back(bind_input);
    opt_.inplaces[bind_input->op] = input;
    assign_count_++;
  }

 private:
  FuncTensorMap &tensor_map_;
  BuildInfoOpt &opt_;
  Map<std::string, NodeRef> op_attrs_;
  int assign_count_{0};
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
    CHECK(tensor_map.count(kv.first)) << kv.first->func_name() << " not in tensor map";
    CHECK(tensor_map.count(kv.second.as<Call>()->func))
      << kv.second.as<Call>()->func->func_name() << " not in tensor map";
    auto first = tensor_map[kv.first];
    auto second = tensor_map[kv.second.as<Call>()->func];
    auto buf = decl_buffer(second->shape, second->dtype, second->op->name);
    info.in_binds.Set(first, buf);
    info.in_binds.Set(second, buf);
  }
}

void ProcessSames(FuncTensorMap &tensor_map, BuildInfoOpt &opt) {
  // b = func(a)
  // c = InplaceAssign(x, y, b)     c = b
  // d = InplaceAssign(i, j, c)     d = c
  bool changed = true;
  while (!opt.sames.empty() && changed) {
    changed = false;
    for (auto it = opt.sames.begin(); it != opt.sames.end();) {
      if (tensor_map.count(it->second)) {
        tensor_map[it->first] = tensor_map[it->second];
        it = opt.sames.erase(it);
        changed = true;
      } else {
        ++it;
      }
    }
  }
}

void CollectInputs(const std::vector<std::string> &input_tensors, FuncTensorMap &tensor_map, BuildInfo &info) {
  for (const auto &input : input_tensors) {
    auto iter =
      std::find_if(tensor_map.begin(), tensor_map.end(),
                   [&input](const std::pair<const FunctionRef, Tensor> &kv) { return kv.first->func_name() == input; });
    CHECK(iter != tensor_map.end()) << "input Tensor " << input << " not built.";
    LOG(INFO) << "input: " << input << " " << iter->second;
    info.args.push_back(iter->second);
  }
}

void CollectOutputsAndComputes(const std::vector<std::string> &output_tensors, FuncTensorMap &tensor_map,
                               BuildInfoOpt &opt, BuildInfo &info) {
  int count = 0;
  for (const auto &output : output_tensors) {
    auto iter = std::find_if(
      tensor_map.begin(), tensor_map.end(),
      [&output](const std::pair<const FunctionRef, Tensor> &kv) { return kv.first->func_name() == output; });
    CHECK(iter != tensor_map.end()) << "output Tensor " << output << " not built.";
    LOG(INFO) << "output: " << output << " " << iter->second;
    info.tensors.push_back(iter->second);
    if (!opt.fakeout.count(iter->first)) {
      info.args.push_back(iter->second);
    } else {
      auto name = "fake_" + std::to_string(count);
      count++;
      Tensor t = placeholder(iter->second->shape, iter->second->dtype, name);
      info.args.push_back(t);
    }
  }
  for (const auto &inplace_itr : opt.inplaces) {
    auto iter =
      std::find_if(tensor_map.begin(), tensor_map.end(), [&inplace_itr](std::pair<const FunctionRef, Tensor> &kv) {
        return kv.first->func_name() == inplace_itr.first->func_name();
      });
    if (std::find_if(info.tensors.begin(), info.tensors.end(),
                     [&iter](const Tensor &t) { return t == iter->second; }) == info.tensors.end()) {
      info.tensors.push_back(iter->second);
    }
  }
}

void CollectSchOnlyComputes(BuildInfoOpt &opt, BuildInfo &info) {
  for (const auto &tensor : opt.sch_only) {
    info.tensors.push_back(tensor);
  }
}

void CollectBuildInfo(const std::vector<std::string> &input_tensors, const std::vector<std::string> &output_tensors,
                      FuncTensorMap &tensor_map, BuildInfoOpt &opt, BuildInfo &info) {
  CollectBinds(tensor_map, opt, info);
  ProcessSames(tensor_map, opt);
  CollectInputs(input_tensors, tensor_map, info);
  CollectOutputsAndComputes(output_tensors, tensor_map, opt, info);
  CollectSchOnlyComputes(opt, info);
}

void EmitIsolatedInplaceTensor(BuildInfoOpt &opt, FuncTensorMap &tensor_map) {
  // tensors which have never be used before is isolated and not be created,
  // so we should create them after emit.
  for (const auto &kv : opt.inplaces) {
    auto c = kv.second.as<Call>();
    if (tensor_map.find(c->func) == tensor_map.end()) {
      tensor_map[c->func] = placeholder(c->args, c->type, c->name);
    }
  }
}

void ExtractBuildInfo(const picojson::value &input_json, BuildInfo &info, std::vector<std::string> &input_tensors,
                      std::vector<std::string> &output_tensors) {
  CHECK(input_json.is<picojson::object>());
  picojson::array input_desc;
  picojson::array output_desc;
  picojson::array op_descs;
  std::string kernelname;
  std::string target;
  // 1. parse input json
  std::tie(kernelname, target, input_desc, output_desc, op_descs) = ParseInputJson(input_json);
  info.kernel_name = kernelname;
  ParseInputTensors(input_desc, input_tensors);
  ParseOutputTensors(output_desc, output_tensors);
  // 2. parse op descs
  auto parser = OpDescsParser(op_descs, input_tensors, output_tensors);
  parser.Parse();
  // 3. make stmt by op descs
  auto stmt = MakeStmt(parser.op_descs_);
  LOG(INFO) << "\n========STMT START========\n" << stmt << "\n========STMT END========\n";
  // 4. optimize stmt
  BuildInfoOpt opt;
  stmt = Optimize(stmt, opt, parser.input_funcs_, parser.output_funcs_, target);
  LOG(INFO) << "\n========OPTIMIZED STMT START========\n" << stmt << "\n========OPTIMIZED STMT END========\n";
  // 5. emit stmt by topi
  FuncTensorMap tensor_map;
  Emitter(tensor_map, opt).Visit(stmt);
  EmitIsolatedInplaceTensor(opt, tensor_map);
  // 6. collect build info: args, compute, binds
  CollectBuildInfo(input_tensors, output_tensors, tensor_map, opt, info);
}

void ExtractBuildInfo(const picojson::value &input_json, BuildInfo &info) {
  std::vector<std::string> input_tensors;
  std::vector<std::string> output_tensors;
  ExtractBuildInfo(input_json, info, input_tensors, output_tensors);
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

Stmt String2LowerStmtSimple(const StringImm *json_str, const Map<std::string, NodeRef> &attrs, bool poly) {
  CHECK(json_str);
  picojson::value v = String2Json(json_str->value);
  BuildInfo info;
  std::vector<std::string> input_tensors;
  std::vector<std::string> output_tensors;
  ExtractBuildInfo(v, info, input_tensors, output_tensors);
  std::string sch_name = GetSchedule(info.tensors);
  const auto *sch_create = air::runtime::Registry::Get("select_cuda_scheduler");
  CHECK(sch_create != nullptr);
  Schedule sch = (*sch_create)(info.tensors, sch_name, poly);
  akg::BuildConfig config = akg::BuildConfig::Current();
  CHECK(config.defined());
  Array<NodeRef> args, shape_vars, arg_list_0;
  Map<Tensor, Buffer> binds, binds_0;
  auto stmt = LowerStmt(sch, info.args, shape_vars, info.kernel_name, info.in_binds, attrs, false, poly, false, "cuda",
                        config, &args, &arg_list_0, &binds, &binds_0, true);
  return Downcast<Stmt>(stmt);
}

Stmt String2LowerStmt(const StringImm *json_str, const Map<std::string, NodeRef> &attrs, bool poly,
                      Array<NodeRef> &all_args, std::unordered_map<std::string, NodeRef> &outputs2args,
                      std::string &merge_name, size_t &idx, int grid_dims, int block_dims, bool buffer_stitch = false) {
  CHECK(json_str);
  picojson::value v = String2Json(json_str->value);
  BuildInfo info;
  std::vector<std::string> input_tensors;
  std::vector<std::string> output_tensors;
  ExtractBuildInfo(v, info, input_tensors, output_tensors);
  // ensure merge_name is the same as original json name
  if (merge_name.empty()) merge_name = info.kernel_name;
  std::string sch_name = GetSchedule(info.tensors);
  const auto *sch_create = air::runtime::Registry::Get("select_cuda_scheduler");
  CHECK(sch_create != nullptr);
  Schedule sch = (*sch_create)(info.tensors, sch_name, poly, grid_dims, block_dims, buffer_stitch);
  akg::BuildConfig config = akg::BuildConfig::Current();
  CHECK(config.defined());
  config->dump_pass_ir = getenv("MS_AKG_DUMP_IR") != nullptr;
  // use idx to distinct different subgraph
  std::string distinct_name = info.kernel_name + "_" + std::to_string(idx);
  Array<NodeRef> args, shape_vars, arg_list_0;
  Map<Tensor, Buffer> binds, binds_0;
  auto stmt = LowerStmt(sch, info.args, shape_vars, distinct_name, info.in_binds, attrs, false, poly, false, "cuda",
                        config, &args, &arg_list_0, &binds, &binds_0, true);
  size_t count = 0;
  for (const auto &x : arg_list_0) {
    auto buffer = x.as<BufferNode>();
    CHECK(buffer) << "arg must be a BufferNode";
    if (std::find(input_tensors.begin(), input_tensors.end(), buffer->name) == std::end(input_tensors)) {
      CHECK(count < output_tensors.size());
      outputs2args[output_tensors[count]] = x;
      count++;
    }
    all_args.push_back(x);
  }
  return Downcast<Stmt>(stmt);
}

NodeRef CompositeWithJsonToFunc(const std::string &json_str, Map<std::string, NodeRef> attrs) {
  const char *akg_dump_pass_ir = getenv("MS_AKG_DUMP_IR");
  picojson::value v = String2Json(json_str);
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

Module CompositeWithJsonGpu(const std::string &json_str, const Map<std::string, NodeRef> &attrs, bool poly) {
  picojson::value v = String2Json(json_str);
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
  picojson::value v = String2Json(json_str);
  BuildInfo info;
  ExtractBuildInfo(v, info);
  Array<Operation> ops;
  std::for_each(info.tensors.begin(), info.tensors.end(), [&ops](const Tensor &t) { ops.push_back(t->op); });
  Schedule sch = create_schedule(ops);
  akg::BuildConfig config = akg::BuildConfig::Current();
  CHECK(config.defined());
  bool tuning = attrs.find("tuning") != attrs.end();
  std::string target = "cce";
  if (GetProcess(json_str) == "cuda") {
    target = "cuda";
  }
  Array<NodeRef> shape_vars;
  return akg::Lower(sch, info.args, shape_vars, info.kernel_name, info.in_binds, attrs, false, true, tuning, target,
                    config);
}
std::vector<std::string> GetNames(const Array<NodeRef> &io) {
  std::vector<std::string> names;
  for (const auto &arg : io) {
    CHECK(arg.as<StringImm>());
    auto arg_name = arg.as<StringImm>()->value;
    names.emplace_back(arg_name);
  }
  return names;
}
Array<NodeRef> ReorderArgs(const Array<NodeRef> &inputs, const Array<NodeRef> &outputs, const Array<NodeRef> &all_args,
                           std::unordered_map<std::string, NodeRef> &outputs2args) {
  // reorder args_list, now args_list satisfies: op1_input op2_input ... op1_output op2_output ...
  // suppose all args info from original json satisfies this order
  Array<NodeRef> input_args, ordered_args;
  std::map<std::string, std::vector<NodeRef>> vmap;
  std::vector<std::string> inputs_name = GetNames(inputs);
  std::vector<std::string> outputs_name = GetNames(outputs);
  for (auto arg : all_args) {
    auto buffer = arg.as<BufferNode>();
    CHECK(buffer) << "arg must be a BufferNode";
    if (std::find(inputs_name.begin(), inputs_name.end(), buffer->name) != std::end(inputs_name)) {
      if (vmap.find(buffer->name) == vmap.end()) {
        input_args.push_back(arg);
        vmap[buffer->name] = {};
      }
      vmap[buffer->name].push_back(arg);
    }
  }
  // input_args is not ordered as args list, should make it first.
  CHECK(inputs_name.size() == input_args.size());
  for (const auto &input : inputs_name) {
    for (const auto &arg : input_args) {
      if (arg.as<BufferNode>()->name == input) {
        ordered_args.push_back(arg);
        break;
      }
    }
  }
  // output args keep order as origin output
  for (const auto &output : outputs_name) {
    if (outputs2args.find(output) != outputs2args.end()) {
      ordered_args.push_back(outputs2args[output]);
    }
  }
  return ordered_args;
}

class ElimDuplicateInputs : public IRMutator {
 public:
  explicit ElimDuplicateInputs(const Array<NodeRef> &inputs) { names_ = GetNames(inputs); }
  Stmt Run(Stmt &stmt) {
    is_mutate_ = false;
    static_cast<void>(Mutate(stmt));
    is_mutate_ = true;
    return Mutate(stmt);
  }

 private:
  Expr Mutate_(const Load *op, const Expr &e) final {
    Var var = op->buffer_var;
    auto name = var->name_hint;
    if (std::find(names_.begin(), names_.end(), name) != names_.end()) {
      auto it = vars_.find(name);
      if (it != vars_.end()) {
        if (is_mutate_) return Load::make(op->type, it->second, op->index, op->predicate);
      } else {
        vars_[name] = var;
      }
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  bool is_mutate_{false};
  std::unordered_map<std::string, Var> vars_;
  std::vector<std::string> names_;
};

std::vector<OpDesc> ParseOpDesc(const std::string &json_str) {
  picojson::value v = String2Json(json_str);
  picojson::array op_desc;
  const picojson::value::object &input_obj = v.get<picojson::object>();
  for (const auto &item : input_obj) {
    if (item.first == "op_desc") {
      CHECK(item.second.is<picojson::array>());
      op_desc = item.second.get<picojson::array>();
    }
  }
  std::vector<std::string> input_tensors;
  std::vector<std::string> output_tensors;
  auto parser = OpDescsParser(op_desc, input_tensors, output_tensors);
  parser.Parse();
  return parser.op_descs_;
}

Map<std::string, NodeRef> BindBlockAndThread(GridBlockDims &dims, bool poly, const Map<std::string, NodeRef> &attrs) {
  Map<std::string, NodeRef> new_attrs;
  if (attrs.defined()) new_attrs = attrs;
  if (poly) {
    std::stringstream ss;
    ss << dims.griddim_x << " " << dims.griddim_y << " " << dims.griddim_z;
    new_attrs.Set("bind_block", Expr(ss.str()));
    ss.str("");
    ss << dims.blockdim_x << " " << dims.blockdim_y << " " << dims.blockdim_z;
    new_attrs.Set("bind_thread", Expr(ss.str()));
    LOG(INFO) << new_attrs;
    return new_attrs;
  }
  return attrs;
}

Stmt InsertSync(Stmt &s) {
  return Block::make(
    s, Evaluate::make(Call::make(Int(32), "tvm_storage_sync", {StringImm::make("shared")}, Call::Intrinsic)));
}

std::vector<Stmt> LowerStitchIRs(const NodeRef &block_json, StitchAttrInfo &stitch_attr,
                                 const Map<std::string, NodeRef> &attrs, bool poly, Array<NodeRef> &all_args,
                                 std::unordered_map<std::string, NodeRef> &outputs2args, std::string &merge_name,
                                 size_t &idx) {
  std::vector<Stmt> stitch_irs;
  std::vector<Expr> loop_extent_array;
  std::vector<GridBlockDims> dim_array;
  std::vector<StitchOpType> ir_type_array;
  for (auto &stitch_json : Downcast<Array<Expr>>(block_json)) {
    ++idx;
    std::vector<OpDesc> op_v = ParseOpDesc(stitch_json.as<StringImm>()->value);
    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    const std::function<Stmt(const StringImm *, const Map<std::string, NodeRef> &, bool)> f =
      std::bind(&String2LowerStmtSimple, _1, _2, _3);
    BufferStitchAttr stitch_attr_info(f);
    stitch_attr_info.GetBufferStitchAttr(stitch_json, op_v, attrs, poly);
    auto dims = stitch_attr_info.dims;
    auto stitch_type = stitch_attr_info.stitch_type_;
    if (idx == 1) loop_extent_array = stitch_attr_info.loop_extent;
    stitch_attr_info.loop_extent = loop_extent_array;  // only care about loop from first ir.
    dim_array.push_back(dims);                         // save current dims into array.
    IrAttrInfo ir_attr_info = GetIRAttr(stitch_type, stitch_attr_info, ir_type_array, dim_array, attrs);
    ir_type_array.push_back(stitch_type);  // Note this should be done AFTER GetIrAttr.
    stitch_attr.broadcast_size = ir_attr_info.broadcast_size;
    stitch_attr.switch_y_2_x = ir_attr_info.switch_y_2_x;
    auto new_attrs = BindBlockAndThread(ir_attr_info.dims, poly, ir_attr_info.attrs);
    auto single_ir = String2LowerStmt(stitch_json.as<StringImm>(), new_attrs, poly, all_args, outputs2args, merge_name,
                                      idx, ir_attr_info.grid_dims, ir_attr_info.block_dims, true);
    stitch_irs.emplace_back(InsertSync(single_ir));
  }
  stitch_attr.type_array = ir_type_array;
  return stitch_irs;
}

Module CompositeWithJsonListGpu(const Array<NodeRef> &json_str_node, const Array<NodeRef> &inputs,
                                const Array<NodeRef> &outputs, const Map<std::string, Array<NodeRef>> &alloc_map,
                                const Map<std::string, Array<NodeRef>> &reuse_map,
                                const Map<std::string, Array<NodeRef>> &clean_op_map,
                                const Map<std::string, NodeRef> &attrs, bool poly) {
  CHECK(!json_str_node.empty());
  std::vector<Stmt> block_irs;
  Array<NodeRef> all_args;
  std::unordered_map<std::string, NodeRef> outputs2args;
  std::string merge_name;
  size_t idx = 0;
  for (auto &block_json : json_str_node) {
    if (block_json.as<StringImm>()) {
      ++idx;
      auto single_ir =
        String2LowerStmt(block_json.as<StringImm>(), attrs, poly, all_args, outputs2args, merge_name, idx, 0, 0);
      block_irs.emplace_back(single_ir);
    } else {
      StitchAttrInfo stitch_attr;
      std::vector<Stmt> stitch_irs =
        LowerStitchIRs(block_json, stitch_attr, attrs, poly, all_args, outputs2args, merge_name, idx);
      StitchBufAlloc buf_manager;
      buf_manager.BufferAllocReuse(stitch_irs, alloc_map, reuse_map, clean_op_map, outputs2args);
      auto stitched_ir = StitchFusion(stitch_irs, stitch_attr, buf_manager.stitch_buffer_map,
                                      buf_manager.buf_within_op_map, buf_manager.allocate_revoke);
      stitched_ir = ElimDuplicateInputs(inputs).Run(stitched_ir);
      block_irs.emplace_back(stitched_ir);
    }
  }
  Array<NodeRef> ordered_args = ReorderArgs(inputs, outputs, all_args, outputs2args);
  auto merged_ir = block_irs.size() == 1 ? block_irs[0] : ir::BlockFusion(block_irs);
  merged_ir = ElimDuplicateInputs(inputs).Run(merged_ir);
  akg::BuildConfig final_config = akg::BuildConfig::Current();
  CHECK(final_config.defined());
  final_config->dump_pass_ir = getenv("MS_AKG_DUMP_IR") != nullptr;
  auto rst = LowerFunc(merged_ir, merge_name, final_config, ordered_args);
  auto build_rst = BuildRstNode::make(rst, merge_name);
  return BuildToModule(build_rst, "cuda");
}

TVM_REGISTER_GLOBAL("composite_with_json_to_func").set_body_typed(CompositeWithJsonToFunc);
TVM_REGISTER_GLOBAL("composite_with_json").set_body_typed(CompositeWithJson);
TVM_REGISTER_GLOBAL("composite_with_json_list_gpu").set_body_typed(CompositeWithJsonListGpu);
TVM_REGISTER_GLOBAL("composite_lower").set_body_typed(CompositeLower);
}  // namespace akg
