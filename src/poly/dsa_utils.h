
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

#ifndef DSA_UTILS_H_
#define DSA_UTILS_H_

namespace akg {
namespace ir {
namespace poly {

constexpr auto LOCAL_BUF = "LOCAL_BUF";
constexpr auto LOCAL_C1 = "LOCAL_C1";
constexpr auto _FRACTAL_C1 = "_FRACTAL_C1";
constexpr auto FRACTAL_C1 = "FRACTAL_C1";
constexpr auto LOCAL_C0C = "LOCAL_C0C";
constexpr auto LOCAL_C1_LOCAL_C0A = "LOCAL_C1_LOCAL_C0A";
constexpr auto LOCAL_C1_LOCAL_C0B = "LOCAL_C1_LOCAL_C0B";
constexpr auto LOCAL_BUF_LOCAL_C0C = "LOCAL_BUF_LOCAL_C0C";
constexpr auto FRACTAL_C1_LOCAL_C0A = "FRACTAL_C1_LOCAL_C0A";
constexpr auto FRACTAL_C1_LOCAL_C0B = "FRACTAL_C1_LOCAL_C0B";

constexpr auto C1_LOCAL_C0A = "C1_LOCAL_C0A";
constexpr auto C1_LOCAL_C0B = "C1_LOCAL_C0B";
constexpr auto LOCAL_C0B = "LOCAL_C0B";

constexpr auto BUF = "BUF";
constexpr auto C1 = "C1";
constexpr auto C0 = "C0";
constexpr auto C0A = "C0A";
constexpr auto C0B = "C0B";
constexpr auto C0C = "C0C";
constexpr auto REG = "REG";

constexpr auto DOT_LOCAL_BUF = "DOT_LOCAL_BUF";
constexpr auto DOT_LOCAL_C1 = "DOT_LOCAL_C1";
constexpr auto DOT_LOCAL_C1_TMP = "DOT_LOCAL_C1_TMP";
constexpr auto DOT_LOCAL_C0A = "DOT_LOCAL_C0A";
constexpr auto DOT_LOCAL_C0B = "DOT_LOCAL_C0B";
constexpr auto DOT_LOCAL_C0C = "DOT_LOCAL_C0C";

constexpr auto LOAD_IM2COL = "LOAD_IM2COL";
constexpr auto REALIZE_C1 = "REALIZE_C1";
constexpr auto REALIZE_C0 = "REALIZE_C0";
constexpr auto REALIZE_BUF = "REALIZE_BUF";
constexpr auto REALIZE_BUFC0 = "REALIZE_BUFC0";
constexpr auto REALIZE_BUFC1 = "REALIZE_BUFC1";
constexpr auto REALIZE_C1BUFC1 = "REALIZE_C1BUFC1";
constexpr auto PRAGMA_BYPATH_FILTER_C0 = "PRAGMA_BYPATH_FILTER_C0";
constexpr auto PRAGMA_BYPATH_FILTER_C1 = "PRAGMA_BYPATH_FILTER_C1";

constexpr auto FLOW_S = 0;
constexpr auto FLOW_V = 1;
constexpr auto FLOW_M = 2;
constexpr auto FLOW_DMA1 = 0;
constexpr auto FLOW_DMA2 = 1;
constexpr auto FLOW_DMA3 = 2;

constexpr auto PRAGMA_MMU_C0WRITE = "PRAGMA_MMU_C0WRITE";
constexpr auto PRAGMA_MMU_C1WRITE = "PRAGMA_MMU_C1WRITE";
constexpr auto K_C1 = "K_C1";
constexpr auto PRAGMA_GEMM_C0 = "PRAGMA_GEMM_C0";

inline int GetCoreValue(const std::string &name) {
  if (name == "Core_num") {
    return 32;
  }
  return -1;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // DSA_UTILS_H_
