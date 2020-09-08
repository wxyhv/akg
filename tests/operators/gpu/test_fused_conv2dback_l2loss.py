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
from akg.ops.poly_gpu import fused_conv2dback_l2loss_manual, fused_conv2dback_l2loss_auto
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tensorio import compare_tensor

def gen_data(data_shape, dtype):
    data = random_gaussian(data_shape, miu=1, sigma=0.1).astype(dtype)
    return data

def compute_py(data_f16, data_f32, layout_f16, layout_f32, dtype):
    if layout_f16 == "NCHW":
        data_f16 = np.transpose(data_f16, axes=(0,2,3,1))
    if layout_f32 == "NCHW":
        data_f32 = np.transpose(data_f32, axes=(0,2,3,1))

    data_f16 = data_f16.astype(dtype)

    fill_data = 0.00004
    data_constant = np.array([float(fill_data)])

    expect = np.multiply(data_constant, data_f32)
    expect = np.add(expect, data_f16)

    output = np.full(np.shape(expect), np.nan, dtype)
    return expect, output

def test_fused_conv2dback_l2loss(data_f16, data_f32, layout_f16, layout_f32, dtype, poly_sch=False):
    data_1 = gen_data(data_f16, 'float16')
    data_2 = gen_data(data_f32, 'float32')

    if (layout_f16 != "NHWC" and layout_f16 != "NCHW") or (layout_f32 != "NHWC" and layout_f32 != "NCHW"):
        raise NotImplementedError('Layout not supported')

    expect, output = compute_py(data_1, data_2, layout_f16, layout_f32, dtype)

    if poly_sch:
        mod = utils.op_build(fused_conv2dback_l2loss_auto, (data_f16, data_f32), ('float16', 'float32'), op_attrs=[layout_f16, layout_f32, dtype], attrs={"target": "cuda"})
    else:
        mod = utils.op_build(fused_conv2dback_l2loss_manual, (data_f16, data_f32), ('float16', 'float32'), op_attrs=[layout_f16, layout_f32, dtype])
    
    output = utils.mod_launch(mod, (data_1, data_2, output), expect = expect)
    res = compare_tensor(output, expect, rtol=5e-03, atol=1e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")
    
    data_1, data_2, expect = to_tvm_nd_array([data_1, data_2, expect])
    gpu_profiling(mod, data_1, data_2, expect, 400)

