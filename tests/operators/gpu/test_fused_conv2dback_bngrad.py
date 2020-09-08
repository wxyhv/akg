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
from akg.ops.poly_gpu import fused_conv2dback_bngrad_manual, fused_conv2dback_bngrad_auto
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tensorio import compare_tensor

def gen_data(data_shape, dtype):
    data = random_gaussian(data_shape, miu=1, sigma=0.1).astype(dtype)
    return data

def compute_py(data_1, data_2, data_3, data_4, data_5, data_6, data_7, const_1, const_2, const_3):
    #data_tmp1 = topi.full_like(data_1, const_1)
    data_tmp1 = np.array([const_1]).astype('float16')
    data_tmp2 = np.greater(data_1, data_tmp1)
    data_tmp3 = np.add(data_2, data_3)
    data_tmp4 = np.where(data_tmp2, data_tmp3, data_tmp1)
    data_tmp5 = data_tmp4.astype('float32')
    n,h,w,c = data_tmp5.shape
    data_tmp6 = np.reshape(data_tmp5, (n*h*w,c))
    data_tmp7 = np.sum(data_tmp6, axis=0)

    data_tmp8 = data_4.astype('float32')
    #data_tmp9 = topi.full_like(data_tmp7, const_3)
    data_tmp9 = np.array([const_3]).astype('float32')
    data_tmp10 = np.multiply(data_5, data_tmp9)
    data_tmp11 = np.broadcast_to(data_tmp10, data_tmp8.shape)
    data_tmp12 = np.subtract(data_tmp8, data_tmp11)
    data_tmp13 = np.multiply(data_tmp5, data_tmp12)
    n,h,w,c = data_tmp13.shape
    data_tmp14 = np.reshape(data_tmp13, (n*h*w,c))
    data_tmp15 = np.sum(data_tmp14, axis=0)

    data_tmp16 = data_6.astype('float32')
    data_tmp17 = np.multiply(data_7, data_tmp9)
    data_tmp18 = np.broadcast_to(data_tmp17, data_tmp16.shape)
    data_tmp19 = np.subtract(data_tmp16, data_tmp18)
    data_tmp20 = np.multiply(data_tmp5, data_tmp19)
    n,h,w,c = data_tmp20.shape
    data_tmp21 = np.reshape(data_tmp20, (n*h*w,c))
    data_tmp22 = np.sum(data_tmp21, axis=0)

    n,h,w,c = data_1.shape
    out_shape = [c]

    return data_tmp7, data_tmp15, data_tmp22, out_shape

def test_fused_conv2dback_bngrad(data_1_shape, data_2_shape, data_3_shape, data_4_shape, data_5_shape, data_6_shape, data_7_shape, const_1, const_2, const_3, poly_sch=False):
    data_1 = gen_data(data_1_shape, 'float16')
    data_2 = gen_data(data_2_shape, 'float16')
    data_3 = gen_data(data_3_shape, 'float16')
    data_4 = gen_data(data_4_shape, 'float16')
    data_5 = gen_data(data_5_shape, 'float32')
    data_6 = gen_data(data_6_shape, 'float16')
    data_7 = gen_data(data_7_shape, 'float32')

    data_tmp7, data_tmp15, data_tmp22, out_shape = compute_py(data_1, data_2, data_3, data_4, data_5, data_6, data_7, const_1, const_2, const_3)
    expect = [data_tmp7, data_tmp15, data_tmp22]
    output = np.full(out_shape, np.nan, 'float32')
    output = [output, output, output]
    if poly_sch:
        mod = utils.op_build(fused_conv2dback_bngrad_auto, (data_1_shape, data_2_shape, data_3_shape, data_4_shape, data_5_shape, data_6_shape, data_7_shape),
                ('float16', 'float16', 'float16', 'float16', 'float32', 'float16', 'float32'), op_attrs=[const_1, const_2, const_3], attrs={"target": "cuda", "use_shared_memory":False})
    else:
        mod = utils.op_build(fused_conv2dback_bngrad_manual, (data_1_shape, data_2_shape, data_3_shape, data_4_shape, data_5_shape, data_6_shape, data_7_shape), 
            ('float16', 'float16', 'float16', 'float16', 'float32', 'float16', 'float32'), op_attrs=[const_1, const_2, const_3])

    output = utils.mod_launch(mod, (data_1, data_2, data_3, data_4, data_5, data_6, data_7, *output), outputs=tuple(range(-len(output),0)), expect = expect)

    res = True
    res &= compare_tensor(output[0], expect[0], rtol=5e-03, atol=1e-8)
    res &= compare_tensor(output[1], expect[1], rtol=5e-03, atol=1e-8)
    res &= compare_tensor(output[2], expect[2], rtol=5e-03, atol=1e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")
    
    data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_tmp7, data_tmp15, data_tmp22 = to_tvm_nd_array([data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_tmp7, data_tmp15, data_tmp22])
    gpu_profiling(mod, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_tmp7, data_tmp15, data_tmp22, 400)

