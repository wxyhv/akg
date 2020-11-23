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
#include "composite/optimize/optimize.h"

namespace akg {
class ReshapeTensorMutator : public IRMutator {
 public:
  explicit ReshapeTensorMutator() {}
  ~ReshapeTensorMutator() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    attr_level_++;
    Map<std::string, NodeRef> attr_map;
    // 1. Gather attribute map
    if (op->node.as<StrMapNode>() != nullptr) {
      attr_map = Downcast<Map<std::string, NodeRef>>(op->node);
      GatherAttrs(attr_map);
    }

    // 2. Traverse current AttrStmt
    auto stmt = this->Mutate(op->body);

    bool update_attr_map = false;
    Map<std::string, NodeRef> attr_map_new;
    // 3. Modify the attribute map if needed
    if (!reshape_.empty() && attr_map.defined()) {
      std::unordered_map<std::string, std::string> format_key;
      for (const auto &it : reshape_) {
        auto key_f = CreateDataFormatKey(it.first->op->name);
        auto key_s = CreateDataFormatKey(it.second->op->name);
        format_key[key_f] = key_s;
      }

      for (const auto &it : attr_map) {
        if (format_key.find(it.first) != format_key.end()) {
          update_attr_map = true;
          attr_map_new.Set(format_key[it.first], it.second);
        } else {
          attr_map_new.Set(it.first, it.second);
        }
      }
    }
    if (update_attr_map) {
      stmt = AttrStmt::make(attr_map_new, op->attr_key, op->value, stmt);
    } else {
      stmt = AttrStmt::make(op->node, op->attr_key, op->value, stmt);
    }

    // 4. Add reshape stmt before the top-level AttrStmt if needed
    if (attr_level_ == 1) {
      if (!reshape_.empty()) {
        for (const auto &it : reshape_) {
          auto arg =
            Call::make(it.first->dtype, it.first->op->name, it.first->shape, Call::CallType::Halide, it.first->op);
          auto reshape_stmt = Provide::make(
            it.second->op, 0, Call::make(it.first->dtype, "Reshape", {arg}, Call::CallType::PureIntrinsic),
            it.second->shape);
          Map<std::string, NodeRef> attrs;
          attrs.Set("shape", it.second->shape);
          auto reshape_attr = AttrStmt::make(attrs, "attrs", Expr(1), reshape_stmt);
          stmt = Block::make(reshape_attr, stmt);
        }
      }

      // Reset status of current top-level AttrStmt
      attr_level_ = 0;
      op_attrs_ = {};
      reshape_ = {};
    }

    return stmt;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    static std::unordered_set<std::string> check_list = {"TensorAdd", "RealDiv", "Mul", "Minimum", "Maximum", "Sub"};
    auto call = op->value.as<Call>();
    if (call == nullptr || check_list.find(call->name) == check_list.end()) {
      return IRMutator::Mutate_(op, s);
    }

    auto op_name = call->name;
    auto args = call->args;
    CHECK_EQ(args.size(), 2);

    // Recover tensor from args to get information: shape, type, tensor name
    Tensor tensor0 = RecoverTensor(args[0]);
    Tensor tensor1 = RecoverTensor(args[1]);
    if (!tensor0.defined() || !tensor1.defined()) {
      LOG(INFO) << "[" << op_name << "] not all inputs are tensors";
      return IRMutator::Mutate_(op, s);
    }
    LOG(INFO) << "[" << op_name << "] input shapes: " << tensor0->shape << ", " << tensor1->shape;

    if (!tensor0->op.defined() || !tensor1->op.defined()) {
      LOG(INFO) << "[" << op_name << "] not all input tensors have operation";
      return IRMutator::Mutate_(op, s);
    }

    // Get format of input tensors
    std::string format0 = GetDataFormat(tensor0->op->name);
    std::string format1 = GetDataFormat(tensor1->op->name);
    LOG(INFO) << "[" << op_name << "] input formats: " << format0 << ", " << format1;
    if (format0 == format1) {
      return IRMutator::Mutate_(op, s);
    }

    bool need_swap = false;
    Array<Expr> shape_fractal;
    Array<Expr> shape_default;
    if (format0 == "FRACTAL_NZ" && format1 == "DefaultFormat") {
      shape_fractal = tensor0->shape;
      shape_default = tensor1->shape;
    } else if (format0 == "DefaultFormat" && format1 == "FRACTAL_NZ") {
      shape_fractal = tensor1->shape;
      shape_default = tensor0->shape;
      need_swap = true;
    } else {
      LOG(WARNING) << "[" << op_name << "], check whether input formats are valid";
      return IRMutator::Mutate_(op, s);
    }

    Array<Expr> shape0;
    Array<Expr> shape1;
    Array<Expr> shape_out;
    auto orig_shape = InferShapeFromFractalNz(shape_fractal);
    std::tie(shape0, shape1, shape_out) = ProduceShapes(orig_shape, shape_default);

    auto dims = shape_out.size();
    auto batch = dims - 2;
    Array<Expr> shape_new;
    CHECK(dims >= 2);
    for (size_t i = 0; i < batch; ++i) {
      shape_new.push_back(shape_out[i]);
    }
    if (is_one(shape1[dims - 2]) && is_one(shape1[dims - 1])) {
      // (bs, 1, 1) --> (bs, 1, 1, 1, 1)
      shape_new.push_back(Expr(1));
      shape_new.push_back(Expr(1));
      shape_new.push_back(Expr(1));
      shape_new.push_back(Expr(1));
    } else if (is_one(shape1[dims - 2]) && GetDim(shape1, dims - 1) == GetDim(shape0, dims - 1)) {
      // (bs, 1, n) --> (bs, n1, 1, 1, n0), where n = n1 * n0
      shape_new.push_back(shape_fractal[shape_fractal.size() - 4]);
      shape_new.push_back(Expr(1));
      shape_new.push_back(Expr(1));
      shape_new.push_back(shape_fractal[shape_fractal.size() - 1]);
    } else if (GetDim(shape1, dims - 2) == GetDim(shape0, dims - 2) && is_one(shape1[dims - 1])) {
      // (bs, m, 1) --> (bs, 1, m1, m0, 1), where m = m1 * m0
      shape_new.push_back(Expr(1));
      shape_new.push_back(shape_fractal[shape_fractal.size() - 3]);
      shape_new.push_back(shape_fractal[shape_fractal.size() - 2]);
      shape_new.push_back(Expr(1));
    } else {
      // (bs, m, n), in this case, data format transformation is needed
      LOG(FATAL) << "[" << op_name << "] " << shape_fractal << " (FRACTAL_NZ) and " << shape_default
                 << " (DefaultFormat) may need data format transformation for ";
    }

    if (!EqualShape(shape_default, shape_new)) {
      LOG(INFO) << "[" << op_name << "] " << shape_default << " will be reshaped to " << shape_new;
      Array<Tensor> input_tensors;
      if (need_swap) {
        auto reshape_tensor = placeholder(shape_new, tensor0->dtype, tensor0->op->name + "_reshape");
        reshape_.Set(tensor0, reshape_tensor);
        input_tensors.push_back(reshape_tensor);
        input_tensors.push_back(tensor1);
      } else {
        auto reshape_tensor = placeholder(shape_new, tensor1->dtype, tensor1->op->name + "_reshape");
        reshape_.Set(tensor1, reshape_tensor);
        input_tensors.push_back(tensor0);
        input_tensors.push_back(reshape_tensor);
      }

      // Rewrite current provide stmt
      Array<Expr> input;
      for (const auto &ele : input_tensors) {
        input.push_back(Call::make(ele->dtype, ele->op->name, ele->shape, Call::CallType::Halide, ele->op));
      }
      return Provide::make(op->func, 0, Call::make(op->value.type(), op_name, input, Call::CallType::PureIntrinsic),
                           op->args);
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  int attr_level_{0};
  Map<std::string, NodeRef> op_attrs_;
  Map<Tensor, Tensor> reshape_;

  void GatherAttrs(const Map<std::string, NodeRef> &attr_map) {
    for (const auto &it : attr_map) {
      if (op_attrs_.find(it.first) != op_attrs_.end()) {
        LOG(WARNING) << it.first << " already exist in attribute map, original value: " << op_attrs_[it.first]
                     << ", overridden value: " << it.second;
      }
      op_attrs_.Set(it.first, it.second);
    }
  }

  Tensor RecoverTensor(const Expr &e) {
    Tensor ret;
    auto call = e.as<Call>();
    if (call == nullptr || call->call_type != Call::CallType::Halide) {
      return ret;
    }

    auto func = call->func;
    if (!func.defined() || func.as<OperationNode>() == nullptr) {
      return ret;
    }

    ret = Downcast<Operation>(func).output(0);
    return ret;
  }

  std::string GetDataFormat(const std::string &name) {
    std::string format;
    auto key = CreateDataFormatKey(name);
    if (op_attrs_.find(key) != op_attrs_.end()) {
      auto value = op_attrs_[key].as<StringImm>();
      if (value != nullptr) {
        format = value->value;
      }
    }
    return format;
  }

  int64_t GetDim(const Array<Expr> &shape, size_t idx) {
    CHECK(idx >= 0 && idx < shape.size());
    auto dim = shape[idx].as<IntImm>();
    CHECK(dim != nullptr);
    return dim->value;
  }

  std::tuple<Array<Expr>, Array<Expr>, Array<Expr>> ProduceShapes(const Array<Expr> &shape1,
                                                                  const Array<Expr> &shape2) {
    bool need_swap = false;
    Array<Expr> shape_tmp;
    Array<Expr> shape_out;
    Array<Expr> shape_long = shape1;
    Array<Expr> shape_short = shape2;
    if (shape1.size() < shape2.size()) {
      shape_long = shape2;
      shape_short = shape1;
      need_swap = true;
    }

    size_t diff_dims = shape_long.size() - shape_short.size();
    for (size_t i = 0; i < diff_dims; ++i) {
      shape_tmp.push_back(Expr(1));
      shape_out.push_back(shape_long[i]);
    }
    for (size_t i = 0; i < shape_short.size(); ++i) {
      if (GetDim(shape_long, i + diff_dims) != GetDim(shape_short, i) && !is_one(shape_long[i + diff_dims]) &&
          !is_one(shape_short[i])) {
        LOG(FATAL) << "Can not broadcast with dimension " << shape_long[i + diff_dims] << " and " << shape_short[i];
      }

      shape_tmp.push_back(shape_short[i]);

      if (is_one(shape_long[i + diff_dims])) {
        shape_out.push_back(shape_short[i]);
      } else {
        shape_out.push_back(shape_long[i + diff_dims]);
      }
    }

    if (need_swap) {
      return std::make_tuple(shape_tmp, shape_long, shape_out);
    }
    return std::make_tuple(shape_long, shape_tmp, shape_out);
  }

  Array<Expr> InferShapeFromFractalNz(const Array<Expr> &fractal) {
    CHECK(fractal.size() >= 4);

    Array<Expr> shape;
    size_t dims = fractal.size();
    size_t batch = dims - 4;
    for (size_t i = 0; i < batch; ++i) {
      shape.push_back(fractal[i]);
    }
    int64_t m = GetDim(fractal, dims - 3) * GetDim(fractal, dims - 2);
    int64_t n = GetDim(fractal, dims - 4) * GetDim(fractal, dims - 1);
    shape.push_back(Expr(m));
    shape.push_back(Expr(n));

    return shape;
  }
};

/* Use reshape to deal with FRACTAL_NZ and DefaultFormat in element-wise and broadcast supported operators. e.g.
 *
 *  output_0_1(256, 256, 16, 16) = TensorAdd(input_1(256, 256, 16, 16), output_0_0(4096))
 *  ------------>
 *  output_0_0_reshape(256, 1, 1, 16) = Reshape(output_0_0(4096))
 *  output_0_1(256, 256, 16, 16) = TensorAdd(input_1(256, 256, 16, 16), output_0_0_reshape(256, 1, 1, 16))
 *
 *  where input_1 is FRACTAL_NZ, output_0_0 is DefaultFormat
 */
Stmt ReshapeTensor(const Stmt &stmt) { return ReshapeTensorMutator().Mutate(stmt); }
}  // namespace akg