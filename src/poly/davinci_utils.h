
/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef DAVINCI_UTILS_H_
#define DAVINCI_UTILS_H_

namespace akg {
namespace ir {
namespace poly {

constexpr auto LOCAL_BUF = "_local_UB";
constexpr auto LOCAL_C1 = "_local_L1";
constexpr auto FRACTAL_C1 = "_fractal_L1";
constexpr auto LOCAL_C0C = "_local_L0C";
constexpr auto LOCAL_C1_LOCAL_C0A = "_local_L1_local_L0A";
constexpr auto LOCAL_C1_LOCAL_C0B = "_local_L1_local_L0B";
constexpr auto LOCAL_BUF_LOCAL_C0C = "_local_UB_local_L0C";
constexpr auto FRACTAL_C1_LOCAL_C0A = "_fractal_L1_local_L0A";

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // DAVINCI_UTILS_H_
