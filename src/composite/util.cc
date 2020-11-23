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
#include "util.h"

namespace akg {
bool IsReduce(const std::string &op_name) {
  // if topi support more, add to this list
  std::unordered_set<std::string> elems = {"ReduceSum", "ReduceMax", "ReduceMin"};
  return elems.find(op_name) != elems.end();
}
bool IsTransform(const std::string &op_name) {
  // if topi support more, add to this list
  std::unordered_set<std::string> elems = {"Reshape", "ExpandDims", "Squeeze", "Flatten", "ProccessNode"};
  return elems.find(op_name) != elems.end();
}
bool IsOtherOp(const std::string &op_name) {
  // if topi support more, add to this list
  std::unordered_set<std::string> elems = {"Matmul", "BatchMatmul",   "Conv",        "Transpose", "Tile",
                                           "Assign", "InplaceAssign", "EquivFormat", "TransData", "AddMinValue"};
  return elems.find(op_name) != elems.end();
}
bool IsElemwise(const std::string &op_name) {
  return !IsReduce(op_name) && !IsTransform(op_name) && !IsOtherOp(op_name);
}
bool EqualShape(const Array<Expr> &shape1, const Array<Expr> &shape2) {
  if (shape1.size() != shape2.size()) return false;
  for (size_t i = 0; i < shape1.size(); ++i) {
    if (!Equal(shape1[i], shape2[i])) {
      return false;
    }
  }
  return true;
}

bool ShapeIsOne(const Array<Expr> &shape) { return shape.size() == 1 && Equal(shape[0], 1); }
std::string GetOpName(const Provide *p) {
  auto call = p->value.as<Call>();
  CHECK(call);
  auto op_name = call->name;
  return op_name;
}
std::string CreateDataFormatKey(const std::string &tensor_name) {
  std::string key = tensor_name + "_format";
  return key;
}

}  // namespace akg
