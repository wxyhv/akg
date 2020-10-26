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
from akg.ops.poly_gpu import reshape_manual, reshape_auto
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array


def gen_data(dtype, in_shape, out_shape):
    inputs = np.random.randint(100, size=in_shape).astype(dtype)
    expect = np.reshape(inputs, out_shape)
    output = np.full(expect.shape, np.nan, dtype)
    return output, expect, inputs


def test_ms_reshape(dtype, in_shape, out_shape, poly_sch=False):
    if poly_sch:
        mod = utils.op_build_test(reshape_auto, [in_shape], [
                             dtype], [out_shape], attrs={"target": "cuda"}, kernel_name="reshape_auto")
    else:
        mod = utils.op_build_test(reshape_manual, [in_shape], [dtype], [out_shape], kernel_name="reshape_manual")

    output, expect, inputs = gen_data(dtype, in_shape, out_shape)
    output = utils.mod_launch(mod, (inputs, output), expect=expect)
    res = np.allclose(output, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    inputs, expect = to_tvm_nd_array([inputs, expect])
    return True
