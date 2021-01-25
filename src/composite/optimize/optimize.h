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
#ifndef COMPOSITE_OPTIMIZE_H_
#define COMPOSITE_OPTIMIZE_H_
#include "composite/util.h"

namespace akg {
Stmt ElimTransformOp(Stmt &s, const FuncRefSet &input_funcs, const FuncRefList &output_funcs, BuildInfoOpt &opt);
Stmt ReshapeTensor(const Stmt &stmt);
}  // namespace akg
#endif  // COMPOSITE_OPTIMIZE_H_
