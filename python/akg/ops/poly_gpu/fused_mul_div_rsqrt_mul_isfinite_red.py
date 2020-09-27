# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""fused operator dsl function"""
from __future__ import absolute_import
import akg
from akg.topi.cuda.injective_single_kernel import schedule_injective
from akg.ops.fused_gpu.fused_mul_div_rsqrt_mul_isfinite_red import fused_mul_div_rsqrt_mul_isfinite_red

@akg.schedule(schedule_injective)
def fused_mul_div_rsqrt_mul_isfinite_red_manual(input1, input2, out_dtype="float32"):
    """fused_mul_div_rsqrt_mul_isfinite_red, with manual schedule"""
    return fused_mul_div_rsqrt_mul_isfinite_red(input1, input2, out_dtype)

def fused_mul_div_rsqrt_mul_isfinite_red_auto(input1, input2, out_dtype="float32"):
    """fused_mul_div_rsqrt_mul_isfinite_red, with auto schedule"""
    return fused_mul_div_rsqrt_mul_isfinite_red(input1, input2, out_dtype)