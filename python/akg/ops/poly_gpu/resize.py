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

"""resize"""
import akg
from akg.topi.cuda.injective_single_kernel import schedule_injective
import akg.tvm as tvm
import akg.topi as topi

@akg.schedule(schedule_injective)
def resize_manual(data, size, layout='NCHW', align_corners=False, method='nearest_neighbor'):
    """resize with manual schedule."""
    return topi.image.resize(data, size, layout=layout,method=method, align_corners=align_corners)

def resize_auto(data, size, layout='NCHW', align_corners=False, method='nearest_neighbor'):
    """resize with auto schedule."""
    return topi.image.resize(data, size, layout=layout,method=method, align_corners=align_corners)