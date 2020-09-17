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

"""resize_nearest_neighbor_grad"""
import akg
import akg.topi as topi
from akg.topi.cuda.injective_single_kernel import schedule_injective
from akg.ops.gpu import resize_nearest_neighbor_grad

@akg.schedule(schedule_injective)
def resize_nearest_neighbor_grad_manual(grad, size, align_corners=True):
    """Resize_nearest_neighbor_grad with manual schedule."""
    return resize_nearest_neighbor_grad.resize_nearest_neighbor_grad(grad, size, align_corners=align_corners)

def resize_nearest_neighbor_grad_auto(grad, size, align_corners=True):
    """Resize_nearest_neighbor_grad with auto schedule."""
    return resize_nearest_neighbor_grad.resize_nearest_neighbor_grad(grad, size, align_corners=align_corners)
