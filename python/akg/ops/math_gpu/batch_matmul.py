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

"""operator dsl function: batch_matmul"""
import akg.topi
import akg.tvm
from akg.utils import validation_check as vc_util


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (akg.tvm.tensor.Tensor, type(None)))
def batch_matmul(data1, data2, bias=None):
    """
    Multiplies two tensors in batches and adds bias to the output.

    Args:
        data1 (tvm.tensor.Tensor): 3D tensor of type float16 or float32 with shape (Batch, M, K).
        data2 (tvm.tensor.Tensor): 3D tensor of type float16 or float32 with shape (Batch, N, K).
        bias (tvm.tensor.Tensor): The bias tensor added to the result of data1 * data2.
                                        Should be of same type as a_value, broadcast is allowed.

    Returns:
        tvm.tensor.Tensor of same type as a_value with shape (Batch, M, N).
    """

    res = akg.topi.nn.batch_matmul(data1, data2)
    if bias is not None:
        res = akg.topi.add(res, bias)
    return res
