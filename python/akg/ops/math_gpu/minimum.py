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
# limitations under the License.

"""operator dsl function: minimum"""
import akg.topi as topi
import akg.tvm as tvm
from akg.utils import validation_check as vc_util


@vc_util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor)
def minimum(data1, data2):
    """
    Take element-wise minimum of two tensors with auto-broadcasting.

    Args:
        data1: tvm.tensor.Tensor
        data2: tvm.tensor.Tensor

    Returns:
        tvm.tensor.Tensor of minimum of two tensors.
    """
    shape1 = [x.value for x in data1.shape]
    shape2 = [x.value for x in data2.shape]
    vc_util.check_shape(shape1)
    vc_util.check_shape(shape2)
    vc_util.auto_broadcast_check(shape1, shape2)
    vc_util.elemwise_dtype_check(data1.dtype, data2.dtype)

    res = topi.minimum(data1, data2)
    return res
