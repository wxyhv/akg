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

"""fused_l2loss_grad"""
import akg
from akg.topi.cuda.injective_single_kernel import schedule_injective
from akg.ops.fused_gpu import fused_l2loss_grad

@akg.schedule(schedule_injective)
def fused_l2loss_grad_manual(data_f16, data_f32, layout='NHWC', fill_data=4e-05):
    """Operator fusion: fused_l2loss_grad"""
    return fused_l2loss_grad.fused_l2loss_grad(data_f16, data_f32, layout, fill_data)

def fused_l2loss_grad_auto(data_f16, data_f32, layout='NHWC', fill_data=4e-05):
    """Operator fusion: fused_l2loss_grad"""
    return fused_l2loss_grad.fused_l2loss_grad(data_f16, data_f32, layout, fill_data)