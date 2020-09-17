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

"""fused_relu_grad_bn_reduce_grad"""
import akg
from akg.topi.cuda.injective_single_kernel import schedule_injective
from akg.ops.fused_gpu import fused_relu_grad_bn_reduce_grad

@akg.schedule(schedule_injective)
def fused_relu_grad_bn_reduce_grad_manual(data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, layout='NHWC'):
    """Operater fusion: fused_relu_grad_bn_reduce_grad"""
    return fused_relu_grad_bn_reduce_grad.fused_relu_grad_bn_reduce_grad(data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, layout)

def fused_relu_grad_bn_reduce_grad_auto(data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, layout='NHWC'):
    """Operater fusion: fused_relu_grad_bn_reduce_grad"""
    return fused_relu_grad_bn_reduce_grad.fused_relu_grad_bn_reduce_grad(data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, layout)
