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

""" fused operator dsl function: fused_pad ResNet50 fused_computation. 957 in XLA patterns  """
from __future__ import absolute_import
import akg.tvm as tvm
import akg.topi as topi
from akg.topi.util import equal_const_int

def fused_pad(input, pad_before, pad_after, layout='NHWC', pad_value=0.0):
    """
    fused_pad.
 
    Args:
        input : tvm.Tensor or Expr
        pad_before : list / tuple of n ints. (Pad width on each dimension to pad the before the axis begin.)
        pad_after : list / tuple of n ints. (Pad width each dimension to pad the after the axis end.)
        pad_value : float. (The value to be padded.)

    Returns
        tvm.Tensor
    """
    if layout == "NCHW":
        data = topi.transpose(data, axes=(0, 2, 3, 1))
    elif layout != "NHWC":
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))
            
    cast_after = topi.cast(input, 'float16')
    n = len(cast_after.shape)
    pad_after = pad_after if pad_after else pad_before
    if len(pad_before) != n:
        raise ValueError("Input dimension and pad_before dismatch : %d vs %d" % (
             n, len(pad_before)))
    if len(pad_after) != n:
        raise ValueError("Input dimension and pad_after dismatch : %d vs %d" % (
             n, len(pad_after)))
    out_shape = tuple(
         tvm.ir_pass.Simplify(
            (cast_after.shape[i] + pad_before[i] + pad_after[i])) for i in range(n))
    pad_value = (pad_value if isinstance(pad_value, tvm.expr.Expr)
                else tvm.const(pad_value, cast_after.dtype))

    def _pad(*indices):
         not_zero = []
         index_tuple = []
         for i in range(n):
             if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
                 index_tuple.append(indices[i])
             else:
                 index_tuple.append(indices[i] - pad_before[i])
                 not_zero.append(indices[i] >= pad_before[i])
                 not_zero.append(indices[i] < cast_after.shape[i] + pad_before[i])
         if not_zero:
             not_zero = tvm.all(*not_zero)
             return tvm.if_then_else(not_zero, cast_after(*index_tuple), pad_value)
         return cast_after(*index_tuple)
    return tvm.compute(out_shape, _pad)
