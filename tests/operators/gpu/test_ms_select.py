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
import numpy as np
from akg.ops.poly_gpu import select_manual, select_auto
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tensorio import compare_tensor

def gen_data(shape_cond, shape_x, dtype_cond, dtype_x):
    support_list = {"float16": np.float16, "float32": np.float32, "int32": np.int32, "int8": np.int8}
    cond = np.random.randint(0, 2, shape_cond).astype(support_list[dtype_cond])
    x1 = random_gaussian(shape_x, miu=1, sigma=0.1).astype(support_list[dtype_x])
    x2 = random_gaussian(shape_x, miu=1, sigma=0.1).astype(support_list[dtype_x])
    expect = np.where(cond, x1, x2)
    output = np.full(shape_x, np.nan, dtype_x)
    return expect, cond, x1, x2, output


def test_ms_select(shape_cond, shape_x, dtype_cond, dtype_x, poly_sch=False):
    if poly_sch:
        mod = utils.op_build(select_auto, [shape_cond, shape_x, shape_x], [dtype_cond, dtype_x, dtype_x], attrs={"target": "cuda"})
    else:
        mod = utils.op_build(select_manual, [shape_cond, shape_x, shape_x], [dtype_cond, dtype_x, dtype_x])
    expect, cond, x1, x2, output = gen_data(shape_cond, shape_x, dtype_cond, dtype_x)
    output = utils.mod_launch(mod, (cond, x1, x2, output), expect=expect)
    res = compare_tensor(output, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    x1, x2, expect = to_tvm_nd_array([x1, x2, expect])
    gpu_profiling(mod, x1, x2, expect, 400)
