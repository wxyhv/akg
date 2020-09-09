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

"""trans_data"""
from akg.ops.array_gpu import trans_data
from akg.topi.cuda.injective_single_kernel import schedule_injective
import akg

@akg.schedule(schedule_injective)
def trans_data_manual(x, axes):
    """trans data"""
    return trans_data.trans_data(x, axes)

def trans_data_auto(x, axes):
    """trans data"""
    return trans_data.trans_data(x, axes)