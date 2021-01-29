
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
constexpr auto _FRACTAL_C1 = "_fractal_L1";
constexpr auto FRACTAL_C1 = "fractal_L1";
constexpr auto LOCAL_C0C = "_local_L0C";
constexpr auto LOCAL_C1_LOCAL_C0A = "_local_L1_local_L0A";
constexpr auto LOCAL_C1_LOCAL_C0B = "_local_L1_local_L0B";
constexpr auto LOCAL_BUF_LOCAL_C0C = "_local_UB_local_L0C";
constexpr auto FRACTAL_C1_LOCAL_C0A = "_fractal_L1_local_L0A";
constexpr auto FRACTAL_C1_LOCAL_C0B = "_fractal_L1_local_L0B";

constexpr auto C1_LOCAL_C0A = "_L1_local_L0A";
constexpr auto C1_LOCAL_C0B = "_L1_local_L0B";
constexpr auto LOCAL_C0B = "_local_L0B";

constexpr auto BUF = "UB";
constexpr auto C1 = "L1";
constexpr auto C0 = "L0";
constexpr auto C0A = "L0A";
constexpr auto C0B = "L0B";
constexpr auto C0C = "L0C";
constexpr auto REG = "REG";

constexpr auto DOT_LOCAL_BUF = "local.UB";
constexpr auto DOT_LOCAL_C1 = "local.L1";
constexpr auto DOT_LOCAL_C1_TMP = "local.L1_tmp";
constexpr auto DOT_LOCAL_C0A = "local.L0A";
constexpr auto DOT_LOCAL_C0B = "local.L0B";
constexpr auto DOT_LOCAL_C0C = "local.L0C";

constexpr auto LOAD_IM2COL = "load_3d";
constexpr auto REALIZE_C1 = "realize_L1";
constexpr auto REALIZE_C0 = "realize_L0";
constexpr auto REALIZE_BUF = "realize_UB";
constexpr auto REALIZE_BUFC0 = "realize_UBL0";
constexpr auto REALIZE_BUFC1 = "realize_UBL1";
constexpr auto REALIZE_C1BUFC1 = "realize_L1UBL1";
constexpr auto PRAGMA_BYPATH_FILTER_C0 = "pragma_bypass_filter_l0";
constexpr auto PRAGMA_BYPATH_FILTER_C1 = "pragma_bypass_filter_l1";

constexpr auto FLOW_S = 1;
constexpr auto FLOW_V = 2;
constexpr auto FLOW_M = 3;
constexpr auto FLOW_DMA1 = 4;
constexpr auto FLOW_DMA2 = 5;
constexpr auto FLOW_DMA3 = 6;

constexpr auto PRAGMA_MMU_C0WRITE = "pragma_cube_l0write";
constexpr auto PRAGMA_MMU_C1WRITE = "pragma_cube_l1write";
constexpr auto K_C1 = "k_l1";
constexpr auto PRAGMA_GEMM_C0 = "pragma_gemm_l0";
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // DAVINCI_UTILS_H_
