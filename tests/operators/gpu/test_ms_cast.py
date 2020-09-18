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
from akg.ops.poly_gpu import cast_manual, cast_auto
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array


def gen_data(shape, srcType, dstType):
    # Result_Numpy
    if srcType == 'int8':
        low_bound = -128
        high_bound = 127
    elif srcType == 'int32':
        low_bound = -1000
        high_bound = 1000
    else:
        low_bound = -1.0
        high_bound = 1.0

    inputs = np.random.uniform(
        low=low_bound, high=high_bound, size=tuple(shape)).astype(srcType)
    expect = inputs.astype(dstType, copy=True)

    # inputs and output to hold the data
    output = np.full(expect.shape, np.nan, dstType)
    return output, expect, inputs


def test_ms_cast(shape, srcType, dstType, poly_sch=False):
    if poly_sch:
        mod = utils.op_build(cast_auto, [shape], [
            srcType], [dstType], attrs={"target": "cuda"})
    else:
        mod = utils.op_build(cast_manual, [shape], [srcType], [dstType])
    output, expect, inputs = gen_data(shape, srcType, dstType)
    output = utils.mod_launch(mod, (inputs, output), expect=expect)
    res = np.allclose(output, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    inputs, expect = to_tvm_nd_array([inputs, expect])
    gpu_profiling(mod, inputs, expect, 400)
