/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <tvm/api_registry.h>

namespace akg {
namespace ir {
using air::ir::LoopPartitionCCE;
using air::runtime::TVMArgs;
using air::runtime::TVMRetValue;

TVM_REGISTER_API("ir_pass.LoopPartitionCCE").set_body([](const TVMArgs args, TVMRetValue *ret) {
  if (args.size() == 2) {
    *ret = LoopPartitionCCE(args[0], args[1]);
  } else if (args.size() == 3) {
    *ret = LoopPartitionCCE(args[0], args[1], args[2]);
  } else {
    CHECK_EQ(args.size(), 4);
    *ret = LoopPartitionCCE(args[0], args[1], args[2], args[3]);
  }
});

#define REGISTER_PASS(PassName) TVM_REGISTER_API("ir_pass." #PassName).set_body_typed(PassName);

REGISTER_PASS(AutoPoly);
REGISTER_PASS(GenTuningSpace);
REGISTER_PASS(RewriteMultiValueFunc);
REGISTER_PASS(RenameRealize);
REGISTER_PASS(ElementwiseFlatten);
REGISTER_PASS(TestInferBoundWithCond);
REGISTER_PASS(TestReduceInequality);
REGISTER_PASS(TestSimplify);
REGISTER_PASS(TestCanProveWithPosParam);
}  // namespace ir
}  // namespace akg
