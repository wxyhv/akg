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
#ifndef COMPOSITE_UTIL_H_
#define COMPOSITE_UTIL_H_
#include "tvm.h"

namespace akg {
constexpr auto kMsDavinciKernelPath = "./kernel_meta/";
constexpr auto kMsGpuKernelPath = "./cuda_meta";
static std::unordered_map<std::string, air::Type> type_mapping = {
  {"float32", air::Float(32)}, {"float16", air::Float(16)}, {"int32", air::Int(32)}, {"bool", air::Bool()}};

bool IsReduce(const std::string &op_name);
bool IsTransform(const std::string &op_name);
bool IsOtherOp(const std::string &op_name);
bool IsElemwise(const std::string &op_name);
bool EqualShape(const Array<Expr> &shape1, const Array<Expr> &shape2);
bool ShapeIsOne(const Array<Expr> &shape);
std::string GetOpName(const Provide *p);
std::string CreateDataFormatKey(const std::string &tensor_name);

using FuncRefMap = std::unordered_map<FunctionRef, FunctionRef, NodeHash, NodeEqual>;
using FuncRefSet = std::unordered_set<FunctionRef, NodeHash, NodeEqual>;
using FuncRefGraph = std::unordered_map<FunctionRef, FuncRefSet, NodeHash, NodeEqual>;
using FuncTensorMap = std::unordered_map<FunctionRef, Tensor, NodeHash, NodeEqual>;
using FuncStmtMap = std::unordered_map<FunctionRef, const Provide *, NodeHash, NodeEqual>;
using FuncShape = std::unordered_map<FunctionRef, Array<Expr>, NodeHash, NodeEqual>;

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

struct Graph {
  FuncRefGraph pre_graph;
  FuncRefGraph post_graph;
  FuncStmtMap func_stmts;
  FuncRefSet input_funcs;
  FuncRefSet output_funcs;
  FuncRefSet visited_funcs;
  FuncShape func_shape;
  bool CanChangeElem(const FunctionRef &output) {
    // if all input shape same as output shape, it can be changed.
    // consider special case: if elemwise input tensor shape is [1], can auto broadcast
    auto inputs = pre_graph[output];
    for (const auto &input : inputs) {
      if (!EqualShape(func_shape[input], func_shape[output]) && !ShapeIsOne(func_shape[input])) {
        return false;
      }
    }
    return true;
  }
};

class StmtToGraph : public IRVisitor {
 public:
  StmtToGraph(const FuncRefSet &input_funcs, const FuncRefSet &output_funcs) {
    g_.input_funcs = input_funcs;
    g_.output_funcs = output_funcs;
  };

 private:
  void Visit_(const Provide *op) override {
    auto call = op->value.as<Call>();
    CHECK(call);
    FuncRefSet inputs = GetInputsFunc(call->args);
    FunctionRef output = op->func;
    g_.pre_graph[output] = inputs;
    for (const auto &input : inputs) {
      g_.post_graph[input].insert(output);
    }
    g_.func_stmts[op->func] = op;
    g_.func_shape[op->func] = op->args;
  }
  FuncRefSet GetInputsFunc(const Array<Expr> &inputs) {
    FuncRefSet set;
    for (const auto &item : inputs) {
      if (auto call = item.as<Call>()) {
        set.insert(call->func);
        g_.func_shape[call->func] = call->args;
      }
    }
    return set;
  }

 public:
  Graph g_;
};

struct NeedReshape {
  FunctionRef func;
  Array<Expr> origin_shape;
};

struct AnalysisResult {
  FuncRefMap to_be_replaced;
  std::unordered_set<const Provide *> to_be_removed;
  std::unordered_map<FunctionRef, Array<Expr>, NodeHash, NodeEqual> changed_shapes;
  std::unordered_map<const Provide *, std::vector<NeedReshape>> need_reshape_map;
  bool ShapeChanged(const FunctionRef &tensor) { return changed_shapes.find(tensor) != changed_shapes.end(); }
  void CollectReshape(const Provide *op, const FunctionRef &func, const Array<Expr> &origin_shape,
                      const Array<Expr> &changed_shape) {
    if (EqualShape(origin_shape, changed_shape)) return;
    NeedReshape nr;
    nr.func = func;
    nr.origin_shape = origin_shape;
    need_reshape_map[op].emplace_back(nr);
  }
  void Dump() {
    LOG(INFO) << "\n=======to_be_replaced=======\n";
    for (const auto &item : to_be_replaced) {
      LOG(INFO) << item.first->func_name() << " -> " << item.second->func_name() << "\n";
    }
    LOG(INFO) << "\n=======to_be_removed=======\n";
    for (const auto &item : to_be_removed) {
      LOG(INFO) << item->func->func_name() << " " << item->value.as<Call>()->name << "\n";
    }
    LOG(INFO) << "\n=======changed_shapes=======\n";
    for (const auto &item : changed_shapes) {
      LOG(INFO) << item.first->func_name() << " -> " << item.second << "\n";
    }
    LOG(INFO) << "\n=======need_reshape_map=======\n";
    for (const auto &item : need_reshape_map) {
      LOG(INFO) << item.first->func->func_name() << " -> \n";
      for (const auto &j : item.second) {
        LOG(INFO) << j.func->func_name() << " -> " << j.origin_shape << "\n";
      }
    }
  }
};

class DoAnalysis : public IRMutator {
 public:
  explicit DoAnalysis(AnalysisResult result) : result_(std::move(result)){};

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
  Expr UpdateInputs(const Call *call) {
    CHECK(call);
    Array<Expr> args;
    for (const auto &arg : call->args) {
      if (auto tensor = arg.as<Call>()) {
        auto shape = tensor->args;
        auto tensor_func = tensor->func;
        if (result_.changed_shapes.find(tensor->func) != result_.changed_shapes.end()) {
          shape = result_.changed_shapes[tensor->func];
        }
        if (result_.to_be_replaced.find(tensor->func) != result_.to_be_replaced.end()) {
          tensor_func = result_.to_be_replaced[tensor->func];
        }
        args.push_back(Call::make(tensor->type, tensor_func->func_name(), shape, tensor->call_type, tensor_func));
      } else {
        args.push_back(arg);
      }
    }
    return Call::make(call->type, call->name, args, call->call_type, call->func);
  }
  Stmt UpdateOutput(const Provide *op, Expr &new_call) {
    Array<Expr> output_shape = op->args;
    if (result_.changed_shapes.count(op->func)) {
      output_shape = result_.changed_shapes[op->func];
    }
    return Provide::make(op->func, op->value_index, new_call, output_shape);
  }
  Expr InputsTryAddReshape(const Expr &new_call, const std::vector<NeedReshape> &nr_vec, std::vector<Stmt> &stmts) {
    auto new_call_p = new_call.as<Call>();
    // input need reshape
    Array<Expr> new_args;
    for (auto &arg : new_call_p->args) {
      auto tmp_arg = arg;
      if (auto input = arg.as<Call>()) {
        for (const auto &nr : nr_vec) {
          if (nr.func == input->func) {
            // if input shape changed, input need reshape
            // b = reduce(a) -> t = trans(a); b = reduce(t)
            auto tensor = NewTensor(nr.origin_shape);
            auto reshape = AddReshape(input->func, tensor->op, input->args, tensor->shape);
            stmts.emplace_back(reshape);
            tmp_arg =
              Call::make(input->type, tensor->op->func_name(), tensor->shape, new_call_p->call_type, tensor->op);
            break;
          }
        }
      }
      new_args.push_back(tmp_arg);
    }
    return Call::make(new_call_p->type, new_call_p->name, new_args, new_call_p->call_type, new_call_p->func);
  }
  void OutputTryAddReshape(const FunctionRef &output, const Provide *provide, const std::vector<NeedReshape> &nr_vec,
                           std::vector<Stmt> &stmts) {
    for (const auto &nr : nr_vec) {
      if (nr.func == output) {
        // if output shape changed, output need reshape
        // b = reduce(a) -> t = reduce(a); b = trans(t)
        auto tensor = NewTensor(nr.origin_shape);
        auto reshape = AddReshape(tensor->op, provide->func, tensor->shape, provide->args);
        auto stmt = Provide::make(tensor->op, provide->value_index, provide->value, tensor->shape);
        if (!op_attrs_.empty()) {
          stmt = AttrStmt::make(op_attrs_, "attrs", Expr(1), stmt);
        }
        stmts.pop_back();  // provide need update
        stmts.emplace_back(stmt);
        stmts.emplace_back(reshape);
        break;
      }
    }
  }
  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    if (result_.to_be_removed.count(op)) {
      return Evaluate::make(0);
    }
    auto call = UpdateInputs(op->value.as<Call>());
    auto provide = UpdateOutput(op, call);
    if (result_.need_reshape_map.count(op)) {
      std::vector<Stmt> stmts;
      auto new_call = InputsTryAddReshape(call, result_.need_reshape_map[op], stmts);
      auto provide_p = provide.as<Provide>();
      auto new_provide = Provide::make(provide_p->func, provide_p->value_index, new_call, provide_p->args);
      auto stmt = new_provide;
      if (!op_attrs_.empty()) {
        stmt = AttrStmt::make(op_attrs_, "attrs", Expr(1), new_provide);
      }
      stmts.emplace_back(stmt);
      OutputTryAddReshape(op->func, new_provide.as<Provide>(), result_.need_reshape_map[op], stmts);
      return Block::make(stmts);
    }
    return provide;
  }

  static Stmt AddReshape(const FunctionRef &input_func, const FunctionRef &output_func, const Array<Expr> &input_shape,
                         const Array<Expr> &output_shape) {
    Array<Expr> input;
    input.push_back(Call::make(Int(32), input_func->func_name(), input_shape, Call::CallType::Halide, input_func));
    auto stmt =
      Provide::make(output_func, 0, Call::make(Int(32), "Reshape", input, Call::CallType::PureIntrinsic), output_shape);
    Map<std::string, NodeRef> attrs;
    attrs.Set("shape", output_shape);
    stmt = AttrStmt::make(attrs, "attrs", Expr(1), stmt);
    return stmt;
  }

  Tensor NewTensor(const Array<Expr> &shape) {
    std::stringstream ss;
    ss << "tmp_" << count_++;
    return placeholder(shape, Int(1), ss.str());
  }

 private:
  AnalysisResult result_;
  int count_{0};
  Map<std::string, NodeRef> op_attrs_;
};

}  // namespace akg

#endif  // COMPOSITE_UTIL_H_
